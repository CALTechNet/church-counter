"""
Frame stitching for panorama generation.
Primary: OpenCV Stitcher (RANSAC-based feature matching), batched
Fallback: Horizontal concat (resized to common height)

Low-light fix: frames are contrast-enhanced before feature detection
so the stitcher finds enough keypoints at 52% house lights.
The saved/displayed image uses the original brightness.

Batching fix: frames are split into small batches before stitching
to prevent OpenCV's bundle adjuster from consuming excessive memory
and crashing the machine.
"""
import base64
import logging
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BATCH_SIZE = 6  # max frames per stitch batch — keeps OpenCV's bundle adjuster stable


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


def stitch_frames(frames: List[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    """
    Stitch frames into a panorama using batched stitching.
    Splits frames into batches of BATCH_SIZE, stitches each batch,
    then stitches the resulting batch panoramas together.
    Returns (image, status_message).
    """
    if not frames:
        return None, "no_frames"

    if len(frames) == 1:
        return frames[0], "single_frame"

    # Deduplicate near-identical frames
    unique = _deduplicate(frames)
    logger.info(f"Stitching {len(unique)} unique frames (from {len(frames)} captured)")

    if len(unique) <= BATCH_SIZE:
        return _stitch_batch(unique)

    # Split into batches and stitch each
    batches = [unique[i:i + BATCH_SIZE] for i in range(0, len(unique), BATCH_SIZE)]
    logger.info(f"Stitching in {len(batches)} batches of up to {BATCH_SIZE} frames")

    batch_panoramas = []
    for i, batch in enumerate(batches):
        logger.info(f"Stitching batch {i + 1}/{len(batches)} ({len(batch)} frames)…")
        pano, status = _stitch_batch(batch)
        if pano is not None:
            batch_panoramas.append(pano)
        else:
            logger.warning(f"Batch {i + 1} stitch failed — using concat fallback for this batch")
            batch_panoramas.append(_concat_fallback(batch))

    if len(batch_panoramas) == 1:
        return batch_panoramas[0], "ok"

    # Stitch the batch panoramas together
    logger.info(f"Stitching {len(batch_panoramas)} batch panoramas together…")
    final, status = _stitch_batch(batch_panoramas)
    if final is not None:
        return final, "ok_batched"

    # Final fallback — concat all batch panoramas
    logger.warning("Final stitch failed — concatenating batch panoramas")
    return _concat_fallback(batch_panoramas), "fallback_concat"


def _stitch_batch(frames: List[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    """Stitch a small batch of frames using OpenCV stitcher with low-light enhancement."""
    if len(frames) == 1:
        return frames[0], "single_frame"

    # Try stitching with enhanced frames first (better keypoint detection in low light)
    with ThreadPoolExecutor() as pool:
        enhanced = list(pool.map(_enhance_for_stitching, frames))

    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, pano = stitcher.stitch(enhanced)
    if status == cv2.Stitcher_OK:
        pano = _crop_black(pano)
        logger.info(f"Batch stitch OK (enhanced) — shape {pano.shape}")
        return pano, "ok"

    logger.warning(f"Enhanced batch stitch failed (status={status}), trying original frames…")

    # Retry with original frames
    status2, pano2 = stitcher.stitch(frames)
    if status2 == cv2.Stitcher_OK:
        pano2 = _crop_black(pano2)
        logger.info(f"Batch stitch OK (original) — shape {pano2.shape}")
        return pano2, "ok_original"

    logger.warning(f"Batch stitch failed (status={status2})")
    return None, f"failed_{status2}"


def _concat_fallback(frames: List[np.ndarray]) -> np.ndarray:
    """Resize all frames to common height and horizontally concatenate."""
    target_h = min(f.shape[0] for f in frames)
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        new_w = int(w * target_h / h)
        resized.append(cv2.resize(f, (new_w, target_h)))
    return np.hstack(resized)


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
