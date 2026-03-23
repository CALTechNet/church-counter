"""
PTZ Optics camera control for Lakeshore Church attendance scanner.
Camera: 10.10.140.140, ceiling center-stage, looking out at congregation.
Scan pattern: 50 presets (100-149) in a zigzag grid, left-to-right.
Captures frames continuously during camera movement for denser coverage.
"""
import asyncio
import logging
import socket
import time
from typing import List, Optional, Callable, Awaitable

import httpx
import cv2
import numpy as np

logger = logging.getLogger(__name__)

CAMERA_IP   = "10.10.140.140"
PTZ_BASE    = f"http://{CAMERA_IP}/cgi-bin/ptzctrl.cgi"
RTSP_URL    = f"rtsp://{CAMERA_IP}:554/1"
VISCA_PORT  = 5678

# ── Scan presets — zigzag grid 100–149 ───────────────────────────────────────
SCAN_PRESETS = list(range(100, 132))  # 100, 101, 102, ... 149

HOME_PRESET = 0

# ── Timing constants ──────────────────────────────────────────────────────────
TRAVEL_TIME     = 1.5   # seconds camera takes to travel between adjacent presets
SETTLE_TIME     = 0   # extra seconds to wait after travel before final capture
CAPTURE_INTERVAL = 0.25  # seconds between frame captures during movement


# ── Low-level PTZ command ─────────────────────────────────────────────────────
async def _ptz(action: str, s1: int = 0, s2: int = 0, timeout: float = 4.0):
    url = f"{PTZ_BASE}?ptzcmd&{action}&{s1}&{s2}"
    async with httpx.AsyncClient(timeout=timeout) as client:
        try:
            resp = await client.get(url)
            logger.debug(f"PTZ {action} s1={s1} s2={s2} → {resp.status_code}")
        except Exception as exc:
            logger.warning(f"PTZ command failed ({action}): {exc}")


# ── Preset helper ─────────────────────────────────────────────────────────────
async def call_preset(preset: int):
    """Call a saved preset — matches exact URL format the camera expects."""
    url = f"{PTZ_BASE}?ptzcmd&poscall&{preset}"
    async with httpx.AsyncClient(timeout=4.0) as client:
        try:
            resp = await client.get(url)
            logger.debug(f"PTZ poscall preset={preset} → {resp.status_code}")
        except Exception as exc:
            logger.warning(f"Preset call failed (preset {preset}): {exc}")


async def go_home():
    """Return camera to home preset after scanning."""
    await call_preset(HOME_PRESET)
    await asyncio.sleep(1.0)
    logger.info("Camera returned to home position")


# ── Position query (VISCA over TCP port 5678) ─────────────────────────────────
def _decode_nibbles(b: bytes) -> int:
    """Decode 4 VISCA nibble bytes (0x0N) into a signed 16-bit integer."""
    val = ((b[0] & 0x0F) << 12 | (b[1] & 0x0F) << 8 |
           (b[2] & 0x0F) << 4  | (b[3] & 0x0F))
    return val - 0x10000 if val >= 0x8000 else val


async def get_position() -> dict:
    """Query pan/tilt/zoom via VISCA over TCP (port 5678).

    Sends two VISCA inquiry commands on one connection and decodes the
    4-nibble signed 16-bit position values from the responses.
    Returns a dict with keys pan, tilt, zoom (int or None on error).
    """
    def _query():
        try:
            with socket.create_connection((CAMERA_IP, VISCA_PORT), timeout=2.0) as sock:
                sock.settimeout(1.0)
                sock.sendall(b'\x81\x09\x06\x12\xFF')   # Pan/Tilt inquiry
                pt = sock.recv(64)
                sock.sendall(b'\x81\x09\x04\x47\xFF')   # Zoom inquiry
                z  = sock.recv(64)
                return pt, z
        except Exception as exc:
            logger.debug(f"get_position failed: {exc}")
            return None, None

    pt_data, z_data = await asyncio.to_thread(_query)

    pan = tilt = zoom = None
    if pt_data and len(pt_data) >= 10 and pt_data[0] == 0x90 and pt_data[1] == 0x50:
        pan  = _decode_nibbles(pt_data[2:6])
        tilt = _decode_nibbles(pt_data[6:10])
    if z_data and len(z_data) >= 6 and z_data[0] == 0x90 and z_data[1] == 0x50:
        zoom = _decode_nibbles(z_data[2:6])

    return {"pan": pan, "tilt": tilt, "zoom": zoom}


# ── Frame capture ─────────────────────────────────────────────────────────────
def capture_frame(url: str = RTSP_URL) -> Optional[np.ndarray]:
    """Open RTSP stream, flush buffer, return latest frame."""
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


async def auto_scan(progress_callback: Optional[ProgressCB] = None) -> List[np.ndarray]:
    """
    Visit all presets in zigzag order.
    For each preset: send move command, then capture frames continuously
    during travel + settle, giving the stitcher dense overlapping coverage.
    Returns list of all captured frames for stitching.
    """
    frames: List[np.ndarray] = []
    total = len(SCAN_PRESETS)
    capture_window = TRAVEL_TIME + SETTLE_TIME  # total seconds to capture per preset

    async def prog(msg: str, pct: int):
        logger.info(f"[{pct:3d}%] {msg}")
        if progress_callback:
            await progress_callback(msg, pct)

    await prog("Starting grid scan…", 0)

    for i, preset_id in enumerate(SCAN_PRESETS):
        pct = int((i / total) * 88)

        # Send move command (don't await movement — start capturing immediately)
        await call_preset(preset_id)

        # Capture frames continuously while camera is moving and settling
        deadline = time.monotonic() + capture_window
        frames_this_preset = 0
        while time.monotonic() < deadline:
            f = await asyncio.to_thread(capture_frame)
            if f is not None:
                frames.append(f)
                frames_this_preset += 1
            await asyncio.sleep(CAPTURE_INTERVAL)

        await prog(
            f"Preset {preset_id} ({i+1}/{total}) — {frames_this_preset} frames captured, {len(frames)} total",
            pct,
        )

    # ── Return home ───────────────────────────────────────────────────────────
    await prog("Returning camera to home position…", 90)
    await go_home()

    await prog(f"Capture complete — {len(frames)} frames", 92)
    logger.info(f"auto_scan finished: {len(frames)} frames captured across {total} presets")
    return frames
