"""
Frame stitching for panorama generation.
Primary: OpenCV Stitcher (RANSAC-based feature matching), batched
Fallback: Horizontal concat (resized to common height)

Grid-aware stitching: when the camera scan provides grid dimensions
(rows × cols from a calibrated scan), frames are stitched column-by-column
first, then the column strips are stitched horizontally.  This matches
the physical scan pattern (vertical-S / boustrophedon) and is far more
robust than treating the frames as a flat sequence.

Sequential fallback: for preset scans (no grid info), the legacy
coverage-based batching is used.

Low-light fix: frames are contrast-enhanced (CLAHE + gamma lift) on the
raw numpy arrays before any JPEG compression.

Memory fix: waiting frames are JPEG-compressed in memory (~600 KB each
vs ~6 MB raw) while batches are processed.
"""
import base64
import gc
import logging
import math
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

BATCH_SIZE = 10   # max frames per stitch batch — keeps OpenCV's bundle adjuster stable
BATCH_OVERLAP = 3  # frames shared between adjacent batches for second-level stitch anchoring


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


def stitch_frames(
    frames: List[np.ndarray],
    grid_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Stitch frames into a panorama.

    When *grid_shape* ``(rows, cols)`` is provided (calibrated scan), uses
    grid-aware stitching: each column is stitched into a vertical strip,
    then columns are stitched horizontally.  This matches the physical
    boustrophedon scan pattern and avoids the meta-batch overlap problem.

    Without *grid_shape* (preset scan), falls back to sequential
    coverage-based batching.

    Returns (image, status_message).
    """
    if not frames:
        return None, "no_frames"

    if len(frames) == 1:
        return frames[0], "single_frame"

    if grid_shape is not None:
        rows, cols = grid_shape
        if len(frames) >= rows * cols:
            # Use exactly rows*cols frames (ignore any extras)
            return _stitch_grid(frames[: rows * cols], rows, cols)
        if len(frames) >= cols:
            # Fewer frames than expected — reduce rows to fit
            actual_rows = len(frames) // cols
            if actual_rows >= 1:
                logger.warning(
                    f"Grid expects {rows}×{cols}={rows*cols} frames but got "
                    f"{len(frames)} — using {actual_rows}×{cols} instead"
                )
                return _stitch_grid(frames[: actual_rows * cols], actual_rows, cols)
        logger.warning(
            f"Grid expects {rows}×{cols}={rows*cols} frames but got "
            f"{len(frames)} — falling back to sequential stitch"
        )

    return _stitch_sequential(frames)


# ---------------------------------------------------------------------------
# Grid-aware stitching (calibrated scan)
# ---------------------------------------------------------------------------

def _stitch_grid(
    frames: List[np.ndarray], rows: int, cols: int,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Stitch a grid of frames captured in column-major boustrophedon order.

    Strategy:
      1. Enhance all frames (CLAHE + gamma).
      2. Reorganise into columns; un-reverse odd columns so every column
         runs top→bottom.
      3. Stitch each column into a vertical strip (≤ BATCH_SIZE frames).
         If a column has more than BATCH_SIZE frames, batch within the
         column with overlap.
      4. Stitch column strips together horizontally, progressively
         (pair-wise left-to-right) if there are too many for one shot.
    """
    logger.info(f"Grid stitch: {rows} rows × {cols} cols = {len(frames)} frames")

    # 1. Enhance on raw numpy arrays
    with ThreadPoolExecutor() as pool:
        enhanced = list(pool.map(_enhance_for_stitching, frames))
    del frames
    logger.info(f"Enhanced {len(enhanced)} frames on raw data")

    # 2. JPEG-compress for memory efficiency
    compressed = [
        cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
        for f in enhanced
    ]
    del enhanced
    gc.collect()

    # 3. Stitch each column into a vertical strip
    column_strips: List[np.ndarray] = []
    for col in range(cols):
        start = col * rows
        # Decompress this column's frames
        col_compressed = compressed[start : start + rows]
        col_frames = [
            cv2.imdecode(np.frombuffer(c, np.uint8), cv2.IMREAD_COLOR)
            for c in col_compressed
        ]
        # Un-boustrophedon: odd columns were scanned bottom→top, reverse
        # so every column is top→bottom for consistent spatial ordering.
        if col % 2 == 1:
            col_frames = col_frames[::-1]

        logger.info(f"Stitching column {col + 1}/{cols} ({len(col_frames)} frames)…")

        if len(col_frames) == 1:
            column_strips.append(col_frames[0])
        elif len(col_frames) <= BATCH_SIZE:
            pano, _ = _stitch_batch(col_frames)
            if pano is not None:
                column_strips.append(pano)
            else:
                logger.warning(f"Column {col + 1} stitch failed — vertical concat")
                column_strips.append(_vconcat_fallback(col_frames))
        else:
            # Batch within the column with overlap
            pano = _stitch_long_column(col_frames)
            column_strips.append(pano)

        del col_frames
        gc.collect()

    del compressed
    gc.collect()

    # 4. Stitch column strips horizontally
    logger.info(f"Stitching {len(column_strips)} column strips horizontally…")
    return _stitch_strips(column_strips)


def _stitch_long_column(frames: List[np.ndarray]) -> np.ndarray:
    """Stitch a column that has more frames than BATCH_SIZE by batching with overlap."""
    n = len(frames)
    step = BATCH_SIZE - BATCH_OVERLAP
    parts: List[np.ndarray] = []
    for i in range(0, n, step):
        batch = frames[i : i + BATCH_SIZE]
        if len(batch) == 1:
            parts.append(batch[0])
        else:
            pano, _ = _stitch_batch(batch)
            parts.append(pano if pano is not None else _vconcat_fallback(batch))
        gc.collect()

    if len(parts) == 1:
        return parts[0]

    # Progressively stitch the column parts
    result = parts[0]
    for p in parts[1:]:
        merged, _ = _stitch_batch([result, p])
        result = merged if merged is not None else _vconcat_fallback([result, p])
        gc.collect()
    return result


def _stitch_strips(strips: List[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    """
    Stitch column strips horizontally using tree reduction.

    Instead of accumulating left-to-right (which makes one side huge and
    causes OpenCV feature-matching to fail once the aspect ratios diverge),
    we stitch adjacent pairs, then pairs of pairs, etc.  This keeps image
    sizes balanced at every level of the tree.
    """
    if len(strips) == 1:
        return strips[0], "ok_grid"

    # Try stitching all strips at once if within batch size
    if len(strips) <= BATCH_SIZE:
        final, status = _stitch_batch(strips, try_panorama_mode=True)
        if final is not None:
            return final, "ok_grid"
        logger.warning("All-at-once strip stitch failed — trying tree reduction")

    # Tree reduction: stitch adjacent pairs, then pairs of pairs, …
    level = strips
    tree_level = 0
    while len(level) > 1:
        tree_level += 1
        next_level: List[np.ndarray] = []
        for j in range(0, len(level), 2):
            if j + 1 >= len(level):
                # Odd one out — carry forward
                next_level.append(level[j])
                continue
            logger.info(
                f"Tree stitch level {tree_level}, pair {j // 2 + 1}/"
                f"{math.ceil(len(level) / 2)}…"
            )
            merged, _ = _stitch_batch(
                [level[j], level[j + 1]], try_panorama_mode=True,
            )
            if merged is not None:
                next_level.append(merged)
            else:
                logger.warning(
                    f"Tree stitch failed at level {tree_level} pair {j // 2 + 1} "
                    f"— horizontal concat"
                )
                next_level.append(_concat_fallback([level[j], level[j + 1]]))
            gc.collect()
        level = next_level

    return level[0], "ok_grid_tree"


def _vconcat_fallback(frames: List[np.ndarray]) -> np.ndarray:
    """Resize all frames to common width and vertically concatenate."""
    target_w = min(f.shape[1] for f in frames)
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        new_h = int(h * target_w / w)
        resized.append(cv2.resize(f, (target_w, new_h)))
    return np.vstack(resized)


# ---------------------------------------------------------------------------
# Sequential stitching (preset scan / fallback)
# ---------------------------------------------------------------------------

def _stitch_sequential(frames: List[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    """
    Legacy sequential stitching for preset scans (no grid info).
    Uses coverage-based batching with overlap.
    """
    # Deduplicate near-identical frames
    unique = _deduplicate(frames)
    logger.info(f"Stitching {len(unique)} unique frames (from {len(frames)} captured)")

    with ThreadPoolExecutor() as pool:
        unique = list(pool.map(_enhance_for_stitching, unique))
    logger.info(f"Enhanced {len(unique)} frames on raw data")

    if len(unique) <= BATCH_SIZE:
        return _stitch_batch(unique)

    frame_groups = _group_indices_by_coverage(unique, batch_size=BATCH_SIZE, overlap=BATCH_OVERLAP)
    n_batches = len(frame_groups)
    logger.info(
        f"Stitching in {n_batches} coverage-based batches "
        f"(sizes: {[len(g) for g in frame_groups]})"
    )

    compressed = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, 95])[1].tobytes()
                  for f in unique]
    del unique
    gc.collect()

    batch_panoramas: List[np.ndarray] = []
    for batch_num, indices in enumerate(frame_groups, 1):
        batch = [cv2.imdecode(np.frombuffer(compressed[i], np.uint8), cv2.IMREAD_COLOR)
                 for i in indices]
        logger.info(f"Stitching batch {batch_num}/{n_batches} ({len(batch)} frames)…")
        pano, _status = _stitch_batch(batch)
        if pano is not None:
            batch_panoramas.append(pano)
        else:
            logger.warning(f"Batch {batch_num} stitch failed — using concat fallback")
            batch_panoramas.append(_concat_fallback(batch))
        del batch
        gc.collect()
    del compressed
    gc.collect()

    if len(batch_panoramas) == 1:
        return batch_panoramas[0], "ok"

    # Stitch batch panoramas together
    logger.info(f"Stitching {len(batch_panoramas)} batch panoramas together…")
    return _stitch_strips(batch_panoramas)


def _stitch_batch(
    frames: List[np.ndarray],
    try_panorama_mode: bool = False,
) -> Tuple[Optional[np.ndarray], str]:
    """Stitch a small batch of pre-enhanced frames using OpenCV stitcher.

    When *try_panorama_mode* is True, a failed SCANS-mode attempt is retried
    with PANORAMA mode at a lower confidence threshold.  This helps when
    merging already-stitched batch panoramas whose perspectives diverge.
    """
    if len(frames) == 1:
        return frames[0], "single_frame"

    # Frames are already enhanced (CLAHE+gamma applied on raw data upstream).
    stitcher = cv2.Stitcher_create(cv2.Stitcher_SCANS)
    status, pano = stitcher.stitch(frames)
    if status == cv2.Stitcher_OK:
        pano = _crop_black(pano)
        logger.info(f"Batch stitch OK — shape {pano.shape}")
        return pano, "ok"

    logger.warning(f"Batch stitch failed (status={status})")

    if try_panorama_mode:
        # Retry with PANORAMA mode + lower confidence — better for images
        # captured from different PTZ positions with varying perspectives.
        logger.info("Retrying with PANORAMA mode (confidence=0.3)…")
        stitcher2 = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
        stitcher2.setPanoConfidenceThresh(0.3)
        status2, pano2 = stitcher2.stitch(frames)
        if status2 == cv2.Stitcher_OK:
            pano2 = _crop_black(pano2)
            logger.info(f"PANORAMA-mode stitch OK — shape {pano2.shape}")
            return pano2, "ok_panorama"
        logger.warning(f"PANORAMA-mode stitch also failed (status={status2})")

    return None, f"failed_{status}"


def _concat_fallback(frames: List[np.ndarray]) -> np.ndarray:
    """Resize all frames to common height and horizontally concatenate."""
    target_h = min(f.shape[0] for f in frames)
    resized = []
    for f in frames:
        h, w = f.shape[:2]
        new_w = int(w * target_h / h)
        resized.append(cv2.resize(f, (new_w, target_h)))
    return np.hstack(resized)


def _group_indices_by_coverage(
    frames: List[np.ndarray],
    batch_size: int = BATCH_SIZE,
    overlap: int = BATCH_OVERLAP,
) -> List[List[int]]:
    """
    Return a list of index-lists grouping frames by scan coverage.

    Batch boundaries are placed at natural transitions — positions where
    histogram correlation between consecutive frames drops well below the
    run's average (the camera moved to a new part of the room) — rather
    than at fixed offsets.  A hard cap at ``batch_size`` prevents any single
    batch from overwhelming the bundle adjuster.

    Adjacent batches share ``overlap`` frame indices so the second-level
    stitch always has shared visual context between batches, preventing
    coverage gaps at batch boundaries.
    """
    n = len(frames)
    if n <= batch_size:
        return [list(range(n))]

    hists = [_hist(f) for f in frames]
    pair_scores = [
        cv2.compareHist(hists[i], hists[i + 1], cv2.HISTCMP_CORREL)
        for i in range(n - 1)
    ]
    mean_score = sum(pair_scores) / len(pair_scores) if pair_scores else 1.0
    # A score below 40 % of the run average signals a meaningful coverage jump
    break_threshold = mean_score * 0.40

    groups: List[List[int]] = []
    start = 0
    for i in range(1, n):
        batch_len = i - start
        at_capacity = batch_len >= batch_size
        natural_break = (
            batch_len >= batch_size // 2
            and pair_scores[i - 1] < break_threshold
        )
        if at_capacity or natural_break:
            groups.append(list(range(start, i)))
            # Next batch starts `overlap` frames back so both batches share
            # those frames as visual anchors during the second-level stitch.
            start = max(i - overlap, start + 1)

    if start < n:
        groups.append(list(range(start, n)))

    return groups


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
