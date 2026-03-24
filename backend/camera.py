"""
PTZ Optics camera control for Lakeshore Church attendance scanner.
Camera: 10.10.140.140, ceiling center-stage, looking out at congregation.
Scan pattern: presets (100-131) in a vertical-S grid, top-to-bottom per column.
Captures frames continuously during camera movement for denser coverage.
Uses VISCA over TCP (port 5678) for PTZ control.
"""
import asyncio
import logging
import math
import socket
import threading
import time
from typing import List, Optional, Callable, Awaitable

import cv2
import numpy as np

logger = logging.getLogger(__name__)

ProgressCB = Callable[[str, int], Awaitable[None]]

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


# ── Preset position learning ───────────────────────────────────────────────────

async def learn_preset_positions(
    progress_callback: Optional[ProgressCB] = None,
) -> dict:
    """Visit each scan preset, wait for the camera to settle, then record its
    pan/tilt position.  Results are stored in the DB under 'preset_positions'
    so that _preset_scan can drive them with move_abs at maximum speed instead
    of relying on the camera's internal preset-recall speed.

    Returns a dict mapping str(preset_id) -> {"pan": int, "tilt": int}.
    """
    import database as db

    positions: dict = {}
    total = len(SCAN_PRESETS)

    async def prog(msg: str, pct: int):
        logger.info(f"[{pct:3d}%] {msg}")
        if progress_callback:
            await progress_callback(msg, pct)

    await prog("Starting preset position learning…", 0)

    for idx, preset_id in enumerate(SCAN_PRESETS):
        pct = int((idx / total) * 95)
        await prog(f"Learning preset {preset_id} ({idx + 1}/{total})…", pct)
        await call_preset(preset_id)
        # Wait for the camera to finish moving before querying position
        await asyncio.sleep(TRAVEL_TIME + SETTLE_TIME + 0.3)
        pos = await asyncio.to_thread(_get_position_sync)
        if pos["pan"] is not None and pos["tilt"] is not None:
            positions[str(preset_id)] = {"pan": pos["pan"], "tilt": pos["tilt"]}
            logger.debug(f"  preset {preset_id}: pan={pos['pan']}, tilt={pos['tilt']}")
        else:
            logger.warning(f"  Could not read position for preset {preset_id}")

    db.set_config("preset_positions", positions)
    await go_home()
    await prog(f"Done — learned {len(positions)}/{total} preset positions", 100)
    logger.info(f"learn_preset_positions: stored {len(positions)} positions")
    return positions


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


def _move_abs_sync(pan: int, tilt: int, pan_speed: int = 24, tilt_speed: int = 20):
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


async def move_abs(pan: int, tilt: int, pan_speed: int = 24, tilt_speed: int = 20):
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


def _zoom_abs_sync(zoom: int):
    """Set absolute zoom position via VISCA 8x 01 04 47 0p 0q 0r 0s FF (blocking)."""
    z = _encode_visca_pos(zoom)
    cmd = bytes([0x81, 0x01, 0x04, 0x47, z[0], z[1], z[2], z[3], 0xFF])
    try:
        s = _visca_connect()
        _visca_send(s, cmd)
        s.close()
    except Exception as exc:
        logger.warning(f"VISCA absolute zoom failed: {exc}")


async def zoom_abs(zoom: int):
    """Set camera zoom to an absolute VISCA position."""
    await asyncio.to_thread(_zoom_abs_sync, zoom)
    logger.debug(f"VISCA absolute zoom={zoom}")


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


# ── Persistent live-view capture (10 fps) ─────────────────────────────────────
_live_lock   = threading.Lock()
_live_latest: Optional[np.ndarray] = None
_live_stop   = threading.Event()
_live_thread: Optional[threading.Thread] = None


def _live_reader_thread(url: str, stop: threading.Event) -> None:
    """Background thread: keep RTSP stream open and update _live_latest."""
    global _live_latest
    cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    while not stop.is_set():
        ret, frame = cap.read()
        if ret:
            with _live_lock:
                _live_latest = frame
        else:
            # Reconnect on failure
            cap.release()
            if stop.is_set():
                break
            time.sleep(0.5)
            cap = cv2.VideoCapture(url, cv2.CAP_FFMPEG)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.release()


def start_live_capture() -> None:
    """Start (or restart) the persistent background live-capture thread."""
    global _live_thread, _live_stop, _live_latest
    # Stop any existing thread
    _live_stop.set()
    if _live_thread and _live_thread.is_alive():
        _live_thread.join(timeout=3)
    with _live_lock:
        _live_latest = None
    _live_stop = threading.Event()
    url = _get_rtsp_url()
    _live_thread = threading.Thread(
        target=_live_reader_thread,
        args=(url, _live_stop),
        daemon=True,
        name="live-capture",
    )
    _live_thread.start()
    logger.info("Live-capture thread started")


def stop_live_capture() -> None:
    """Stop the persistent live-capture thread."""
    global _live_latest
    _live_stop.set()
    if _live_thread and _live_thread.is_alive():
        _live_thread.join(timeout=3)
    with _live_lock:
        _live_latest = None
    logger.info("Live-capture thread stopped")


def get_live_frame() -> Optional[np.ndarray]:
    """Return the most-recently captured live frame (non-blocking)."""
    with _live_lock:
        return _live_latest.copy() if _live_latest is not None else None


# ── Auto-scan ─────────────────────────────────────────────────────────────────

async def _preset_scan(
    presets: List[int],
    progress_callback: Optional[ProgressCB] = None,
    cancel_event=None,
    camera_ip: str = None,
) -> List[np.ndarray]:
    """Visit a list of VISCA presets, capturing one frame immediately after each move.
    Navigates to the first preset and waits 1.5s to settle before scanning.
    If preset positions have been learned (via learn_preset_positions), uses
    move_abs at maximum speed instead of preset recall for faster traversal."""
    import database as db
    preset_positions = db.get_config("preset_positions", {})
    using_abs = bool(preset_positions)
    if using_abs:
        logger.info("Preset scan: using learned positions with move_abs (max speed)")
    else:
        logger.info("Preset scan: using preset recall (run learn-presets for faster movement)")

    frames: List[np.ndarray] = []
    total = len(presets)

    async def prog(msg: str, pct: int):
        logger.info(f"[{pct:3d}%] {msg}")
        if progress_callback:
            await progress_callback(msg, pct)

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    async def goto_preset(preset_id: int):
        """Move to a preset using absolute positioning if known, else preset recall."""
        pos = preset_positions.get(str(preset_id))
        if pos:
            await move_abs(pos["pan"], pos["tilt"])
        else:
            await call_preset(preset_id)

    await prog("Starting preset scan…", 0)

    if not presets:
        return frames

    # Go to first preset and wait for camera to settle
    await prog(f"Moving to first preset {presets[0]}…", 0)
    await goto_preset(presets[0])
    await asyncio.sleep(3.0)

    if cancelled():
        return frames

    # Capture frames during the settle window at first preset
    f = await asyncio.to_thread(capture_frame)
    if f is not None:
        frames.append(f)
    await prog(f"Preset {presets[0]} (1/{total}) — {len(frames)} total", 0)

    # Move through remaining presets; wait 100ms after each move, then capture
    for i, preset_id in enumerate(presets[1:], 1):
        if cancelled():
            return frames
        pct = int((i / total) * 88)
        await goto_preset(preset_id)
        await asyncio.sleep(0.1)
        f = await asyncio.to_thread(capture_frame)
        frames_this = 0
        if f is not None:
            frames.append(f)
            frames_this += 1

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
    progress_callback: Optional[ProgressCB] = None,
    cancel_event=None,
    camera_ip: str = None,
) -> List[np.ndarray]:
    """
    Move camera through a grid of absolute PTZ positions between the saved
    top-left and bottom-right bounds, capturing frames at each stop.
    Grid density is calculated from the saved zoom level: at zoom=10000,
    each photo covers ~25 pan units and ~25 tilt units.
    """
    import database as db
    bounds = db.get_config("camera_bounds", {})

    # New format: individual left/right/top/bottom edges
    left   = bounds.get("left")
    right  = bounds.get("right")
    top    = bounds.get("top")
    bottom = bounds.get("bottom")

    # Fallback: migrate from old {top_left, bottom_right} format
    if left is None or right is None or top is None or bottom is None:
        tl = bounds.get("top_left")
        br = bounds.get("bottom_right")
        if not tl or not br:
            logger.error("No camera bounds saved — cannot do calibrated scan. Set bounds in Calibration.")
            return []
        left   = int(tl["pan"])
        right  = int(br["pan"])
        top    = int(tl["tilt"])
        bottom = int(br["tilt"])

    pan_tl,  tilt_tl = int(left),  int(top)
    pan_br,  tilt_br = int(right), int(bottom)

    # Use the override zoom if set, otherwise default to 10000
    zoom = int(bounds.get("zoom") or 10000)
    zoom = max(1, zoom)

    # Step size derived from two calibrated data points:
    #   zoom=10000 → step=75,  zoom=5000 → step=200
    # Fitting step = a/zoom + b gives a=1_250_000, b=-50
    # i.e.  step = 1_250_000 / zoom - 50  (minimum 25)
    pan_step  = max(25, int(1_250_000 / zoom) - 50)
    tilt_step = max(25, int(1_250_000 / zoom) - 50)
    pan_range  = abs(pan_br  - pan_tl)
    tilt_range = abs(tilt_br - tilt_tl)
    cols = max(1, math.ceil(pan_range  / pan_step)  + 1)
    rows = max(1, math.ceil(tilt_range / tilt_step) + 1)

    logger.info(
        f"Grid: zoom={zoom} → step={pan_step}, {cols} cols × {rows} rows "
        f"({cols * rows} positions)"
    )

    # Build position list in a vertical-S (column-major boustrophedon) pattern:
    # matches the preset scan order — start top-left, go top-to-bottom down each
    # column, shift right to the next column, then alternate direction.
    # Even columns go top→bottom, odd columns go bottom→top.
    positions: List[tuple] = []
    for col in range(cols):
        pan_frac = col / max(cols - 1, 1)
        pan = int(pan_tl + (pan_br - pan_tl) * pan_frac)
        row_range = range(rows) if col % 2 == 0 else range(rows - 1, -1, -1)
        for row in row_range:
            tilt_frac = row / max(rows - 1, 1)
            tilt = int(tilt_tl + (tilt_br - tilt_tl) * tilt_frac)
            positions.append((pan, tilt))

    frames: List[np.ndarray] = []
    total = len(positions)

    async def prog(msg: str, pct: int):
        logger.info(f"[{pct:3d}%] {msg}")
        if progress_callback:
            await progress_callback(msg, pct)

    def cancelled() -> bool:
        return cancel_event is not None and cancel_event.is_set()

    await prog(f"Starting calibrated scan ({rows}×{cols} grid, {total} positions)…", 0)

    if not positions:
        return frames

    # Go to top-left (first position), zoom to saved level, and wait for camera to settle
    pan0, tilt0 = positions[0]
    await prog(f"Moving to top-left (pan={pan0}, tilt={tilt0}) and zooming to {zoom}…", 0)
    await move_abs(pan0, tilt0)
    await zoom_abs(zoom)
    await asyncio.sleep(3.0)

    if cancelled():
        return frames

    # Capture frames during the settle window at top-left
    f = await asyncio.to_thread(capture_frame)
    if f is not None:
        frames.append(f)
    await prog(f"Position 1/{total} (pan={pan0}, tilt={tilt0}) — {len(frames)} total", 0)

    # Move through remaining positions; wait 25ms after each move for vibration to settle, then capture
    for i, (pan, tilt) in enumerate(positions[1:], 1):
        if cancelled():
            return frames
        pct = int((i / total) * 88)
        await move_abs(pan, tilt, pan_speed=24, tilt_speed=20)
        await asyncio.sleep(0.025)
        f = await asyncio.to_thread(capture_frame)
        frames_this = 0
        if f is not None:
            frames.append(f)
            frames_this += 1

        await prog(
            f"Position {i+1}/{total} (pan={pan}, tilt={tilt}) — {frames_this} frames, {len(frames)} total",
            pct,
        )

    await prog("Returning camera to home…", 90)
    await go_home()
    await prog(f"Capture complete — {len(frames)} frames", 92)
    logger.info(f"_calibrated_scan done: {len(frames)} frames across {total} positions")
    return frames


async def auto_scan(progress_callback: Optional[ProgressCB] = None, cancel_event=None, room_config: dict = None) -> List[np.ndarray]:
    """
    Dispatch to either preset or calibrated scan based on saved app settings,
    or use room_config if provided to determine camera type and settings.
    """
    import database as db

    # If a room config is provided, use it to determine scan behaviour
    if room_config:
        camera_type = room_config.get("camera_type", "ptz_optics")

        if camera_type == "rtsp":
            # Generic RTSP: just capture a single frame from the URL
            rtsp_url = room_config.get("rtsp_url", "")
            if not rtsp_url:
                logger.error("No RTSP URL configured for room")
                return []
            if progress_callback:
                await progress_callback("Capturing RTSP frame…", 10)
            frame = await asyncio.to_thread(capture_frame, rtsp_url)
            if frame is None:
                logger.error(f"Failed to capture frame from RTSP URL: {rtsp_url}")
                return []
            if progress_callback:
                await progress_callback("Frame captured", 90)
            return [frame]

        # PTZ Optics: use room-specific settings
        mode = room_config.get("scan_mode", "preset")
        if mode == "calibrated":
            return await _calibrated_scan(
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                camera_ip=room_config.get("camera_ip"),
            )
        else:
            preset_start = int(room_config.get("preset_start", 100))
            preset_end   = int(room_config.get("preset_end", 131))
            presets = list(range(preset_start, preset_end + 1))
            return await _preset_scan(
                presets,
                progress_callback=progress_callback,
                cancel_event=cancel_event,
                camera_ip=room_config.get("camera_ip"),
            )

    # Fallback: use global settings (backward compat)
    settings = db.get_config("app_settings", {})
    mode = settings.get("scan_mode", "preset")

    if mode == "calibrated":
        return await _calibrated_scan(progress_callback=progress_callback, cancel_event=cancel_event)
    else:
        preset_start = int(settings.get("preset_start", 100))
        preset_end   = int(settings.get("preset_end",   131))
        presets = list(range(preset_start, preset_end + 1))
        return await _preset_scan(presets, progress_callback=progress_callback, cancel_event=cancel_event)
