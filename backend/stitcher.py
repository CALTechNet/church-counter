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


def _find_horizontal_overlap(a: np.ndarray, b: np.ndarray, max_overlap: float = 0.75) -> int:
    """
    Find horizontal overlap between two frames using template matching.
    Searches the right portion of `a` for the left edge strip of `b`.
    Returns the overlap in pixels (0 if none found).
    """
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb)

    # Downscale for speed
    scale = min(1.0, 400.0 / h)
    sh = int(h * scale)
    swa = int(wa * scale)
    swb = int(wb * scale)
    if sh < 16 or swa < 16 or swb < 16:
        return 0

    ga = cv2.cvtColor(cv2.resize(a[:h], (swa, sh)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b[:h], (swb, sh)), cv2.COLOR_BGR2GRAY)

    # Template: left strip of b (15% of its width)
    strip_w = max(8, int(swb * 0.15))
    template = gb[:, :strip_w]

    # Search region: right portion of a (up to max_overlap of a's width)
    search_w = int(swa * max_overlap)
    search_region = ga[:, swa - search_w:]

    if search_region.shape[1] < template.shape[1]:
        return 0

    res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.3:
        return 0

    # Overlap = distance from match position to right edge of search region
    overlap_scaled = search_w - max_loc[0]
    overlap = int(overlap_scaled / scale)
    return max(0, min(overlap, int(wa * max_overlap), wb))


def _blend_horizontal(a: np.ndarray, b: np.ndarray, overlap: int) -> np.ndarray:
    """Merge two frames with a linear blend across the overlap zone."""
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    h = min(ha, hb)
    a = a[:h]
    b = b[:h]

    out_w = wa + wb - overlap
    result = np.zeros((h, out_w, 3), dtype=np.uint8)

    # Left part (non-overlapping portion of a)
    result[:, :wa - overlap] = a[:, :wa - overlap]
    # Right part (non-overlapping portion of b)
    result[:, wa:] = b[:, overlap:]

    # Blend the overlap region
    if overlap > 0:
        alpha = np.linspace(1, 0, overlap, dtype=np.float32).reshape(1, -1, 1)
        region_a = a[:, wa - overlap:].astype(np.float32)
        region_b = b[:, :overlap].astype(np.float32)
        result[:, wa - overlap:wa] = (region_a * alpha + region_b * (1 - alpha)).astype(np.uint8)

    return result


def _hconcat_pair(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Join two frames, detecting and blending any horizontal overlap."""
    target_h = min(a.shape[0], b.shape[0])
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    ra = cv2.resize(a, (int(wa * target_h / ha), target_h))
    rb = cv2.resize(b, (int(wb * target_h / hb), target_h))

    overlap = _find_horizontal_overlap(ra, rb)
    if overlap > 0:
        logger.info(f"    Found {overlap}px horizontal overlap, blending")
        return _blend_horizontal(ra, rb, overlap)

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
    Stitch frames into a panorama.

    When grid_shape (rows, cols) is provided (calibrated PTZ scan), frames are
    assumed to be in column-major boustrophedon order (down col 0, up col 1, …).
    They are rearranged into rows, each row is stitched independently, then
    rows are vertically concatenated.  This avoids the accumulator-growth
    problem where a huge panorama can't match features with a small new frame.

    Without grid_shape, falls back to sequential pair-wise stitching with
    deduplication.

    Returns (image, status_message).
    """
    if not frames:
        return None, "no_frames"

    if len(frames) == 1:
        return frames[0], "single_frame"

    if grid_shape and len(grid_shape) == 2:
        rows, cols = grid_shape
        if rows * cols == len(frames):
            return _stitch_grid(frames, rows, cols)
        logger.warning(
            f"grid_shape {grid_shape} doesn't match frame count {len(frames)}, "
            "falling back to sequential stitch"
        )

    # Fallback: sequential stitch with deduplication
    unique = _deduplicate(frames)
    n = len(unique)
    logger.info(f"Stitching {n} unique frames (from {len(frames)} captured)")

    result = _stitch_sequential(unique)
    result = _crop_black(result)
    logger.info(f"Stitch complete — output shape: {result.shape}")
    return result, "ok"


def _stitch_grid(
    frames: List[np.ndarray], rows: int, cols: int,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Stitch frames captured in column-major boustrophedon order into a panorama.

    1. Rearrange into a row-major grid (grid[row][col]).
    2. Stitch each row left-to-right using pair-wise stitching.
    3. Vertically concatenate all row strips.
    """
    logger.info(f"Grid stitch: {rows}r × {cols}c = {rows * cols} frames")

    # Build grid[row][col] from column-major boustrophedon order
    grid: List[List[np.ndarray]] = [[None] * cols for _ in range(rows)]
    idx = 0
    for col in range(cols):
        row_range = range(rows) if col % 2 == 0 else range(rows - 1, -1, -1)
        for row in row_range:
            grid[row][col] = frames[idx]
            idx += 1

    # Stitch each row independently
    row_strips = []
    for r in range(rows):
        row_frames = [f for f in grid[r] if f is not None]
        if not row_frames:
            continue
        logger.info(f"  Stitching row {r + 1}/{rows} ({len(row_frames)} frames)")
        strip = _stitch_sequential(row_frames, label=f"row{r + 1}: ")
        row_strips.append(strip)
        gc.collect()

    if not row_strips:
        return None, "grid_empty"

    if len(row_strips) == 1:
        result = row_strips[0]
    else:
        # Vertically concatenate rows (resize to common width)
        result = _vconcat_strips(row_strips)

    result = _crop_black(result)
    logger.info(f"Grid stitch complete — output shape: {result.shape}")
    return result, "ok_grid"


def _find_vertical_overlap(a: np.ndarray, b: np.ndarray, max_overlap: float = 0.75) -> int:
    """
    Find vertical overlap between two strips using template matching.
    Searches the bottom portion of `a` for the top edge strip of `b`.
    Returns the overlap in pixels (0 if none found).
    """
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    w = min(wa, wb)

    scale = min(1.0, 400.0 / w)
    sw = int(w * scale)
    sha = int(ha * scale)
    shb = int(hb * scale)
    if sw < 16 or sha < 16 or shb < 16:
        return 0

    ga = cv2.cvtColor(cv2.resize(a[:, :w], (sw, sha)), cv2.COLOR_BGR2GRAY)
    gb = cv2.cvtColor(cv2.resize(b[:, :w], (sw, shb)), cv2.COLOR_BGR2GRAY)

    # Template: top strip of b (15% of its height)
    strip_h = max(8, int(shb * 0.15))
    template = gb[:strip_h, :]

    # Search region: bottom portion of a
    search_h = int(sha * max_overlap)
    search_region = ga[sha - search_h:, :]

    if search_region.shape[0] < template.shape[0]:
        return 0

    res = cv2.matchTemplate(search_region, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.3:
        return 0

    overlap_scaled = search_h - max_loc[1]
    overlap = int(overlap_scaled / scale)
    return max(0, min(overlap, int(ha * max_overlap)))


def _blend_vertical(a: np.ndarray, b: np.ndarray, overlap: int) -> np.ndarray:
    """Merge two strips with a linear blend across the vertical overlap zone."""
    ha, wa = a.shape[:2]
    hb, wb = b.shape[:2]
    w = min(wa, wb)
    a = a[:, :w]
    b = b[:, :w]

    out_h = ha + hb - overlap
    result = np.zeros((out_h, w, 3), dtype=np.uint8)

    result[:ha - overlap] = a[:ha - overlap]
    result[ha:] = b[overlap:]

    if overlap > 0:
        alpha = np.linspace(1, 0, overlap, dtype=np.float32).reshape(-1, 1, 1)
        region_a = a[ha - overlap:].astype(np.float32)
        region_b = b[:overlap].astype(np.float32)
        result[ha - overlap:ha] = (region_a * alpha + region_b * (1 - alpha)).astype(np.uint8)

    return result


def _vconcat_strips(strips: List[np.ndarray]) -> np.ndarray:
    """Resize strips to the same width, then vertically concatenate with overlap blending."""
    target_w = max(s.shape[1] for s in strips)
    resized = []
    for s in strips:
        h, w = s.shape[:2]
        if w != target_w:
            new_h = int(h * target_w / w)
            s = cv2.resize(s, (target_w, new_h))
        resized.append(s)

    result = resized[0]
    for i, strip in enumerate(resized[1:], 1):
        overlap = _find_vertical_overlap(result, strip)
        if overlap > 0:
            logger.info(f"  Row {i+1}: found {overlap}px vertical overlap, blending")
            result = _blend_vertical(result, strip, overlap)
        else:
            result = np.vstack([result, strip])
    return result


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
