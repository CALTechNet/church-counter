"""
Frame stitching for panorama generation.
Primary: OpenCV Stitcher (RANSAC-based feature matching)
Fallback: Horizontal concat (resized to common height)

Low-light fix: frames are contrast-enhanced before feature detection
so the stitcher finds enough keypoints at 52% house lights.
The saved/displayed image uses the original brightness.

Crash prevention: when grid_shape is available, stitch row-by-row
(max ~16 frames per stitch call) instead of all 100+ frames at once.
OpenCV's Stitcher does O(n²) feature matching which causes native
segfaults/OOM with large frame counts.
"""
import base64
import gc
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# Maximum frames to feed into a single cv2.Stitcher.stitch() call.
# Beyond this, OpenCV's O(n²) feature matching can segfault.
_MAX_STITCH_BATCH = 15


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


def _stitch_small_batch(frames: List[np.ndarray]) -> Optional[np.ndarray]:
    """
    Stitch a small batch of frames (should be <= _MAX_STITCH_BATCH).
    Tries enhanced first, then original, returns None on failure.
    """
    if len(frames) == 1:
        return frames[0]

    try:
        enhanced = [_enhance_for_stitching(f) for f in frames]
        stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status, pano = stitcher.stitch(enhanced)
        del enhanced
        gc.collect()

        if status == cv2.Stitcher_OK:
            return pano

        logger.debug(f"Enhanced batch stitch failed (status={status}), trying originals")
        stitcher2 = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        status2, pano2 = stitcher2.stitch(frames)
        del stitcher, stitcher2
        gc.collect()

        if status2 == cv2.Stitcher_OK:
            return pano2

    except Exception as exc:
        logger.warning(f"Stitch batch exception: {exc}")
        gc.collect()

    return None


def _hconcat_fallback(frames: List[np.ndarray]) -> np.ndarray:
    """Resize all frames to the same height, then horizontally concatenate."""
    target_h = min(f.shape[0] for f in frames)
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        new_w = int(w * target_h / h)
        resized.append(cv2.resize(f, (new_w, target_h)))
    return np.hstack(resized)


def _vconcat_fallback(strips: List[np.ndarray]) -> np.ndarray:
    """Resize all strips to the same width, then vertically concatenate."""
    target_w = min(s.shape[1] for s in strips)
    resized = []
    for s in strips:
        h, w = s.shape[:2]
        new_h = int(h * target_w / w)
        resized.append(cv2.resize(s, (target_w, new_h)))
    return np.vstack(resized)


def _stitch_row(row_frames: List[np.ndarray], row_idx: int) -> np.ndarray:
    """Stitch a single row of frames. Falls back to hconcat if stitching fails."""
    logger.info(f"  Row {row_idx}: stitching {len(row_frames)} frames")

    if len(row_frames) <= _MAX_STITCH_BATCH:
        result = _stitch_small_batch(row_frames)
        if result is not None:
            logger.info(f"  Row {row_idx}: stitch OK — shape {result.shape}")
            return result
    else:
        # Row is larger than max batch — stitch in overlapping chunks
        chunks = []
        step = _MAX_STITCH_BATCH - 2  # overlap of 2 frames between chunks
        for start in range(0, len(row_frames), step):
            chunk = row_frames[start:start + _MAX_STITCH_BATCH]
            if len(chunk) < 2:
                chunks.append(chunk[0])
                continue
            result = _stitch_small_batch(chunk)
            if result is not None:
                chunks.append(result)
            else:
                chunks.append(_hconcat_fallback(chunk))
            gc.collect()

        if len(chunks) == 1:
            logger.info(f"  Row {row_idx}: stitch OK (chunked) — shape {chunks[0].shape}")
            return chunks[0]

        # Try to stitch the chunks together
        merged = _stitch_small_batch(chunks)
        if merged is not None:
            logger.info(f"  Row {row_idx}: stitch OK (merged chunks) — shape {merged.shape}")
            return merged

    # Fallback
    logger.warning(f"  Row {row_idx}: stitch failed, using hconcat fallback")
    return _hconcat_fallback(row_frames)


def stitch_frames(
    frames: List[np.ndarray],
    grid_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Stitch frames into a panorama.
    When grid_shape (rows, cols) is provided, stitches row-by-row to avoid
    feeding too many frames to OpenCV at once (which causes native crashes).
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

    # --- Small enough to stitch all at once ---
    if n <= _MAX_STITCH_BATCH:
        result = _stitch_small_batch(unique)
        if result is not None:
            result = _crop_black(result)
            logger.info(f"Stitch OK (single batch) — output shape: {result.shape}")
            return result, "ok"
        logger.warning("Single-batch stitch failed, using fallback concat")
        return _hconcat_fallback(unique), "fallback_concat"

    # --- Grid-aware row-by-row stitching ---
    if grid_shape is not None:
        rows, cols = grid_shape
        logger.info(f"Grid-aware stitch: {rows} rows × {cols} cols ({n} unique frames)")

        # Map unique frames back into the grid.
        # After dedup, we may have fewer frames than rows*cols.
        # Distribute frames into rows as evenly as possible.
        if n >= rows * cols:
            # No frames were dropped — use exact grid layout
            row_strips = []
            for r in range(rows):
                start = r * cols
                row_frames = unique[start:start + cols]
                row_strips.append(_stitch_row(row_frames, r))
                gc.collect()
        else:
            # Some dupes removed — distribute evenly into rows
            frames_per_row = max(1, n // rows)
            row_strips = []
            idx = 0
            for r in range(rows):
                end = idx + frames_per_row
                if r == rows - 1:
                    end = n  # last row gets remainder
                row_frames = unique[idx:end]
                if row_frames:
                    row_strips.append(_stitch_row(row_frames, r))
                    gc.collect()
                idx = end

        if not row_strips:
            return _hconcat_fallback(unique), "fallback_concat"

        if len(row_strips) == 1:
            result = _crop_black(row_strips[0])
            return result, "ok_grid"

        # Combine row strips vertically
        logger.info(f"Combining {len(row_strips)} row strips vertically")
        result = _vconcat_fallback(row_strips)
        result = _crop_black(result)
        logger.info(f"Grid stitch complete — output shape: {result.shape}")
        return result, "ok_grid"

    # --- No grid shape: batch sequentially ---
    logger.info(f"Sequential batch stitch: {n} frames in batches of {_MAX_STITCH_BATCH}")
    strips = []
    step = _MAX_STITCH_BATCH - 2  # overlap for continuity
    for start in range(0, n, step):
        batch = unique[start:start + _MAX_STITCH_BATCH]
        if len(batch) < 2:
            strips.append(batch[0])
            continue
        result = _stitch_small_batch(batch)
        if result is not None:
            strips.append(result)
        else:
            strips.append(_hconcat_fallback(batch))
        gc.collect()
        logger.info(f"  Batch {start//step + 1}: done — shape {strips[-1].shape}")

    if len(strips) == 1:
        result = _crop_black(strips[0])
        return result, "ok_batched"

    # Try stitching the strips together
    merged = _stitch_small_batch(strips)
    if merged is not None:
        merged = _crop_black(merged)
        logger.info(f"Batch stitch complete — output shape: {merged.shape}")
        return merged, "ok_batched"

    # Final fallback: just vconcat the strips
    result = _vconcat_fallback(strips)
    result = _crop_black(result)
    logger.info(f"Batch stitch (vconcat fallback) — output shape: {result.shape}")
    return result, "fallback_vconcat"


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
