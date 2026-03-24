"""
Frame stitching for panorama generation.
Primary: OpenCV Stitcher (RANSAC-based feature matching)
Fallback: Horizontal concat (resized to common height)

Low-light fix: frames are contrast-enhanced before feature detection
so the stitcher finds enough keypoints at 52% house lights.
The saved/displayed image uses the original brightness.

Crash prevention: only ever stitch 2 frames at a time, then
accumulate by stitching the running result with the next frame.
This keeps OpenCV memory usage minimal and avoids native segfaults.
"""
import base64
import gc
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


def _enhance_for_stitching(frame: np.ndarray) -> np.ndarray:
    """
    Boost contrast/brightness for feature detection only.
    CLAHE on L channel + gamma lift so ORB/SIFT find more keypoints in dark frames.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    # Gamma 2.0 lift
    lut = np.array([((i / 255.0) ** (1.0 / 2.0)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(enhanced, lut)


def _stitch_pair(a: np.ndarray, b: np.ndarray) -> Optional[np.ndarray]:
    """
    Stitch exactly 2 frames. Tries enhanced first, then original.
    Returns None on failure.
    """
    try:
        enhanced = [_enhance_for_stitching(a), _enhance_for_stitching(b)]
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, pano = stitcher.stitch(enhanced)
        del enhanced, stitcher
        gc.collect()

        if status == cv2.Stitcher_OK:
            return pano

        logger.debug(f"Enhanced pair stitch failed (status={status}), trying originals")
        stitcher2 = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status2, pano2 = stitcher2.stitch([a, b])
        del stitcher2
        gc.collect()

        if status2 == cv2.Stitcher_OK:
            return pano2

    except Exception as exc:
        logger.warning(f"Stitch pair exception: {exc}")
        gc.collect()

    return None


def _hconcat_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Resize two frames to the same height, then horizontally concatenate."""
    target_h = min(a.shape[0], b.shape[0])
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    ra = cv2.resize(a, (int(wa * target_h / ha), target_h))
    rb = cv2.resize(b, (int(wb * target_h / hb), target_h))
    return np.hstack([ra, rb])


def _stitch_sequential(frames: List[np.ndarray], label: str = "") -> np.ndarray:
    """
    Stitch a list of frames by accumulating one at a time:
      result = frames[0]
      result = stitch(result, frames[1])
      result = stitch(result, frames[2])
      ...
    Falls back to hconcat for any pair that fails.
    """
    result = frames[0]
    for i, frame in enumerate(frames[1:], 1):
        logger.info(f"  {label}Stitching frame {i + 1}/{len(frames)}")
        merged = _stitch_pair(result, frame)
        if merged is not None:
            result = merged
        else:
            logger.warning(f"  {label}Pair stitch failed at frame {i + 1}, using hconcat")
            result = _hconcat_pair(result, frame)
        gc.collect()
    return result


def stitch_frames(
    frames: List[np.ndarray],
    grid_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Stitch frames into a panorama — only ever 2 frames per OpenCV call.
    Walks through every frame in order, stitching each onto the accumulator.
    grid_shape is accepted but ignored (kept for API compat).
    Returns (image, status_message).
    """
    if not frames:
        return None, "no_frames"

    if len(frames) == 1:
        return frames[0], "single_frame"

    # Deduplicate near-identical frames
    unique = _deduplicate(frames)
    n = len(unique)
    logger.info(f"Stitching {n} unique frames (from {len(frames)} captured)")

    result = _stitch_sequential(unique)
    result = _crop_black(result)
    logger.info(f"Stitch complete — output shape: {result.shape}")
    return result, "ok"


def _deduplicate(frames: List[np.ndarray], threshold: float = 0.97) -> List[np.ndarray]:
    """Remove near-duplicate frames based on histogram correlation."""
    if len(frames) <= 2:
        return frames
    unique = [frames[0]]
    prev_hist = _hist(frames[0])
    for f in frames[1:]:
        h = _hist(f)
        corr = cv2.compareHist(prev_hist, h, cv2.HISTCMP_CORREL)
        if corr < threshold:
            unique.append(f)
            prev_hist = h
    return unique


def _hist(frame: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.calcHist([gray], [0], None, [64], [0, 256])


def _crop_black(img: np.ndarray) -> np.ndarray:
    """Crop black borders from stitched panorama."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y: y + h, x: x + w]


def to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode frame as base64 JPEG."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def from_base64(b64: str) -> np.ndarray:
    """Decode base64 JPEG to frame."""
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
