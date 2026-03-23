"""
PTZ Optics camera control for Lakeshore Church attendance scanner.
Camera: 10.10.140.140, ceiling center-stage, looking out at congregation.
Scan pattern: 50 presets (100-149) in a zigzag grid, left-to-right.
Captures frames continuously during camera movement for denser coverage.
"""
import asyncio
import logging
import re
import time
from typing import List, Optional, Callable, Awaitable

import httpx
import cv2
import numpy as np

logger = logging.getLogger(__name__)

CAMERA_IP = "10.10.140.140"
PTZ_BASE = f"http://{CAMERA_IP}/cgi-bin/ptzctrl.cgi"
RTSP_URL = f"rtsp://{CAMERA_IP}:554/1"

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


# ── Position query ─────────────────────────────────────────────────────────────
def _parse_var(text: str, name: str) -> Optional[int]:
    m = re.search(rf"(?:var\s+)?{name}\s*[=:]\s*(-?\d+)", text)
    return int(m.group(1)) if m else None


async def get_position() -> dict:
    """Query the camera's current pan/tilt/zoom position.

    PTZ Optics cameras expose current position via param.cgi.
    Returns a dict with keys pan, tilt, zoom (int or None on error).
    """
    url = f"http://{CAMERA_IP}/cgi-bin/param.cgi?get_pan_tilt_zoom"
    async with httpx.AsyncClient(timeout=2.0) as client:
        try:
            resp = await client.get(url)
            text = resp.text
            return {
                "pan":  _parse_var(text, "pan"),
                "tilt": _parse_var(text, "tilt"),
                "zoom": _parse_var(text, "zoom"),
            }
        except Exception as exc:
            logger.debug(f"get_position failed: {exc}")
            return {"pan": None, "tilt": None, "zoom": None}


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
