"""
PTZ Optics camera control for Lakeshore Church attendance scanner.
Camera: 10.10.140.140, ceiling center-stage, looking out at congregation.
Scan pattern: presets (100-131) in a zigzag grid, left-to-right.
Captures frames continuously during camera movement for denser coverage.
Uses VISCA over TCP (port 5678) for PTZ control.
"""
import asyncio
import logging
import math
import socket
import time
from typing import List, Optional, Callable, Awaitable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Default / fallback values (overridden by DB settings at runtime)
CAMERA_IP  = "10.10.140.140"
VISCA_PORT = 5678
RTSP_URL   = f"rtsp://{CAMERA_IP}:554/1"

# ── Scan presets — zigzag grid 100–131 ───────────────────────────────────────
SCAN_PRESETS = list(range(100, 132))


# ── Runtime-configurable helpers ─────────────────────────────────────────────

def _get_camera_ip() -> str:
    try:
        import database as db
        s = db.get_config("app_settings", {})
        return s.get("camera_ip") or CAMERA_IP
    except Exception:
        return CAMERA_IP


def _get_rtsp_url() -> str:
    ip = _get_camera_ip()
    return f"rtsp://{ip}:554/1"

HOME_PRESET = 0

# ── Timing constants ──────────────────────────────────────────────────────────
TRAVEL_TIME      = 1.5   # seconds camera takes to travel between adjacent presets
SETTLE_TIME      = 0     # extra seconds to wait after travel before final capture
CAPTURE_INTERVAL = 0.25  # seconds between frame captures during movement


# ── VISCA TCP helpers ─────────────────────────────────────────────────────────

def _visca_connect() -> socket.socket:
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(4.0)
    s.connect((_get_camera_ip(), VISCA_PORT))
    return s


def _visca_send(s: socket.socket, cmd: bytes, read_timeout: float = 2.0) -> bytes:
    """Send a VISCA command and return the response bytes."""
    s.sendall(cmd)
    s.settimeout(read_timeout)
    try:
        return s.recv(64)
    except socket.timeout:
        return b''


def _decode_visca_pos(data: bytes, offset: int) -> int:
    """Decode 4 VISCA nibble bytes at offset into a signed 16-bit int."""
    v = (
        (data[offset]     & 0xF) << 12 |
        (data[offset + 1] & 0xF) << 8  |
        (data[offset + 2] & 0xF) << 4  |
        (data[offset + 3] & 0xF)
    )
    return v - 0x10000 if v > 0x7FFF else v


# ── Preset helper ─────────────────────────────────────────────────────────────

def _call_preset_sync(preset: int):
    """Call a VISCA preset recall command (blocking)."""
    # VISCA: 8x 01 04 3F 02 pp FF  (preset recall)
    cmd = bytes([0x81, 0x01, 0x04, 0x3F, 0x02, preset & 0xFF, 0xFF])
    try:
        s = _visca_connect()
        _visca_send(s, cmd)
        s.close()
    except Exception as exc:
        logger.warning(f"VISCA preset {preset} failed: {exc}")


async def call_preset(preset: int):
    """Call a saved VISCA preset."""
    await asyncio.to_thread(_call_preset_sync, preset)
    logger.debug(f"VISCA preset recall preset={preset}")


# ── Home position ─────────────────────────────────────────────────────────────

def _go_home_sync():
    """Send VISCA Home command (blocking)."""
    # VISCA: 81 01 06 04 FF  (pan/tilt home)
    cmd = b'\x81\x01\x06\x04\xFF'
    try:
        s = _visca_connect()
        _visca_send(s, cmd)
        s.close()
    except Exception as exc:
        logger.warning(f"VISCA home failed: {exc}")


async def go_home():
    """Return camera to home position via VISCA."""
    await asyncio.to_thread(_go_home_sync)
    await asyncio.sleep(1.0)
    logger.info("Camera returned to home position")


# ── Absolute PTZ move (VISCA) ─────────────────────────────────────────────────

def _encode_visca_pos(v: int) -> list:
    """Encode a signed 16-bit integer into 4 VISCA nibble bytes."""
    u = v & 0xFFFF
    return [
        0x00 | ((u >> 12) & 0x0F),
        0x00 | ((u >>  8) & 0x0F),
        0x00 | ((u >>  4) & 0x0F),
        0x00 | ((u      ) & 0x0F),
    ]


def _move_abs_sync(pan: int, tilt: int, pan_speed: int = 6, tilt_speed: int = 6):
    """VISCA 8x 01 06 02 VV WW [pan×4] [tilt×4] FF — blocking."""
    p = _encode_visca_pos(pan)
    t = _encode_visca_pos(tilt)
    cmd = bytes([0x81, 0x01, 0x06, 0x02, pan_speed & 0x1F, tilt_speed & 0x1F,
                 p[0], p[1], p[2], p[3], t[0], t[1], t[2], t[3], 0xFF])
    try:
        s = _visca_connect()
        _visca_send(s, cmd)
        s.close()
    except Exception as exc:
        logger.warning(f"VISCA absolute move failed: {exc}")


async def move_abs(pan: int, tilt: int, pan_speed: int = 6, tilt_speed: int = 6):
    """Move camera to absolute pan/tilt position."""
    await asyncio.to_thread(_move_abs_sync, pan, tilt, pan_speed, tilt_speed)
    logger.debug(f"VISCA absolute move pan={pan} tilt={tilt}")


# ── Position query ─────────────────────────────────────────────────────────────

def _get_position_sync() -> dict:
    """Query pan/tilt/zoom via VISCA inquiry (blocking)."""
    result = {"pan": None, "tilt": None, "zoom": None}
    try:
        s = _visca_connect()

        # Pan/Tilt position inquiry: 81 09 06 12 FF
        # Response: 90 50 0p 0q 0r 0s 0t 0u 0v 0w FF
        s.sendall(b'\x81\x09\x06\x12\xFF')
        time.sleep(0.1)
        try:
            data = s.recv(64)
            if len(data) >= 11 and data[0] == 0x90 and data[1] == 0x50:
                result["pan"]  = _decode_visca_pos(data, 2)
                result["tilt"] = _decode_visca_pos(data, 6)
        except socket.timeout:
            pass

        # Zoom position inquiry: 81 09 04 47 FF
        # Response: 90 50 0p 0q 0r 0s FF
        s.sendall(b'\x81\x09\x04\x47\xFF')
        time.sleep(0.1)
        try:
            data = s.recv(64)
            if len(data) >= 7 and data[0] == 0x90 and data[1] == 0x50:
                result["zoom"] = _decode_visca_pos(data, 2) & 0xFFFF
        except socket.timeout:
            pass

        s.close()
    except Exception as exc:
        logger.debug(f"get_position failed: {exc}")
    return result


async def get_position() -> dict:
    """Query the camera's current pan/tilt/zoom position via VISCA."""
    return await asyncio.to_thread(_get_position_sync)


# ── Manual pan/tilt/zoom control ─────────────────────────────────────────────

def _pan_tilt_sync(pan_dir: int, tilt_dir: int, pan_speed: int = 8, tilt_speed: int = 8):
    """Send a VISCA Pan-Tilt Drive command (blocking).
    pan_dir:  0x01=left, 0x02=right, 0x03=stop
    tilt_dir: 0x01=up,   0x02=down,  0x03=stop
    """
    pan_speed  = max(1, min(24, pan_speed))
    tilt_speed = max(1, min(20, tilt_speed))
    cmd = bytes([0x81, 0x01, 0x06, 0x01,
                 pan_speed, tilt_speed,
                 pan_dir, tilt_dir, 0xFF])
    try:
        s = _visca_connect()
        _visca_send(s, cmd)
        s.close()
    except Exception as exc:
        logger.warning(f"VISCA pan/tilt failed: {exc}")


async def pan_left(speed: int = 8):
    await asyncio.to_thread(_pan_tilt_sync, 0x01, 0x03, speed, speed)

async def pan_right(speed: int = 8):
    await asyncio.to_thread(_pan_tilt_sync, 0x02, 0x03, speed, speed)

async def tilt_up(speed: int = 8):
    await asyncio.to_thread(_pan_tilt_sync, 0x03, 0x01, speed, speed)

async def tilt_down(speed: int = 8):
    await asyncio.to_thread(_pan_tilt_sync, 0x03, 0x02, speed, speed)

async def stop():
    """Stop all pan/tilt movement."""
    await asyncio.to_thread(_pan_tilt_sync, 0x03, 0x03)


def _zoom_sync(direction: int, speed: int = 4):
    """Send a VISCA Zoom command (blocking).
    direction: 0x20=tele (in), 0x30=wide (out), 0x00=stop
    """
    speed = max(0, min(7, speed))
    zoom_byte = direction | (speed if direction != 0x00 else 0)
    cmd = bytes([0x81, 0x01, 0x04, 0x07, zoom_byte, 0xFF])
    try:
        s = _visca_connect()
        _visca_send(s, cmd)
        s.close()
    except Exception as exc:
        logger.warning(f"VISCA zoom failed: {exc}")


async def zoom_in(speed: int = 4):
    await asyncio.to_thread(_zoom_sync, 0x20, speed)

async def zoom_out(speed: int = 4):
    await asyncio.to_thread(_zoom_sync, 0x30, speed)

async def zoom_stop():
    await asyncio.to_thread(_zoom_sync, 0x00, 0)


# ── Frame capture ─────────────────────────────────────────────────────────────
def capture_frame(url: str = None) -> Optional[np.ndarray]:
    """Open RTSP stream, flush buffer, return latest frame."""
    if url is None:
        url = _get_rtsp_url()
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    frame = None
    for _ in range(4):  # fewer flushes since we want near-realtime frames
        ret, f = cap.read()
        if ret:
            frame = f
    cap.release()
    if frame is None:
        logger.warning("Failed to capture frame from RTSP stream")
    return frame


# ── Auto-scan ─────────────────────────────────────────────────────────────────
ProgressCB = Callable[[str, int], Awaitable[None]]


async def _preset_scan(
    presets: List[int],
    progress_callback: Optional[ProgressCB] = None,
) -> List[np.ndarray]:
    """Visit a list of VISCA presets, capturing frames continuously during each move."""
    frames: List[np.ndarray] = []
    total = len(presets)
    capture_window = TRAVEL_TIME + SETTLE_TIME

    async def prog(msg: str, pct: int):
        logger.info(f"[{pct:3d}%] {msg}")
        if progress_callback:
            await progress_callback(msg, pct)

    await prog("Starting preset scan…", 0)

    for i, preset_id in enumerate(presets):
        pct = int((i / total) * 88)
        await call_preset(preset_id)

        deadline = time.monotonic() + capture_window
        frames_this = 0
        while time.monotonic() < deadline:
            f = await asyncio.to_thread(capture_frame)
            if f is not None:
                frames.append(f)
                frames_this += 1
            await asyncio.sleep(CAPTURE_INTERVAL)

        await prog(
            f"Preset {preset_id} ({i+1}/{total}) — {frames_this} frames, {len(frames)} total",
            pct,
        )

    await prog("Returning camera to home…", 90)
    await go_home()
    await prog(f"Capture complete — {len(frames)} frames", 92)
    logger.info(f"_preset_scan done: {len(frames)} frames across {total} presets")
    return frames


async def _calibrated_scan(
    scan_positions: int = 24,
    progress_callback: Optional[ProgressCB] = None,
) -> List[np.ndarray]:
    """
    Move camera through a grid of absolute PTZ positions between the saved
    top-left and bottom-right bounds, capturing frames at each stop.
    """
    import database as db
    bounds = db.get_config("camera_bounds", {})
    tl = bounds.get("top_left")
    br = bounds.get("bottom_right")
    if not tl or not br:
        logger.error("No camera bounds saved — cannot do calibrated scan. Set bounds in Calibration.")
        return []

    pan_tl, tilt_tl = int(tl["pan"]), int(tl["tilt"])
    pan_br, tilt_br = int(br["pan"]), int(br["tilt"])

    # Build grid: aspect ratio ~1.78:1 (pan:tilt) for a typical wide sanctuary
    n = max(4, scan_positions)
    cols = max(1, round(math.sqrt(n * 1.78)))
    rows = max(1, round(n / cols))

    # Build zigzag position list (left→right on even rows, right→left on odd)
    positions: List[tuple] = []
    for row in range(rows):
        tilt_frac = row / max(rows - 1, 1)
        tilt = int(tilt_tl + (tilt_br - tilt_tl) * tilt_frac)
        col_range = range(cols) if row % 2 == 0 else range(cols - 1, -1, -1)
        for col in col_range:
            pan_frac = col / max(cols - 1, 1)
            pan = int(pan_tl + (pan_br - pan_tl) * pan_frac)
            positions.append((pan, tilt))

    frames: List[np.ndarray] = []
    total = len(positions)

    async def prog(msg: str, pct: int):
        logger.info(f"[{pct:3d}%] {msg}")
        if progress_callback:
            await progress_callback(msg, pct)

    await prog(f"Starting calibrated scan ({rows}×{cols} grid, {total} positions)…", 0)

    for i, (pan, tilt) in enumerate(positions):
        pct = int((i / total) * 88)
        await move_abs(pan, tilt)
        await asyncio.sleep(TRAVEL_TIME + SETTLE_TIME)

        deadline = time.monotonic() + 0.6
        frames_this = 0
        while time.monotonic() < deadline:
            f = await asyncio.to_thread(capture_frame)
            if f is not None:
                frames.append(f)
                frames_this += 1
            await asyncio.sleep(CAPTURE_INTERVAL)

        await prog(
            f"Position {i+1}/{total} (pan={pan}, tilt={tilt}) — {frames_this} frames, {len(frames)} total",
            pct,
        )

    await prog("Returning camera to home…", 90)
    await go_home()
    await prog(f"Capture complete — {len(frames)} frames", 92)
    logger.info(f"_calibrated_scan done: {len(frames)} frames across {total} positions")
    return frames


async def auto_scan(progress_callback: Optional[ProgressCB] = None) -> List[np.ndarray]:
    """
    Dispatch to either preset or calibrated scan based on saved app settings.
    """
    import database as db
    settings = db.get_config("app_settings", {})
    mode = settings.get("scan_mode", "preset")

    if mode == "calibrated":
        scan_positions = int(settings.get("scan_positions", 24))
        return await _calibrated_scan(scan_positions=scan_positions, progress_callback=progress_callback)
    else:
        preset_start = int(settings.get("preset_start", 100))
        preset_end   = int(settings.get("preset_end",   131))
        presets = list(range(preset_start, preset_end + 1))
        return await _preset_scan(presets, progress_callback=progress_callback)
