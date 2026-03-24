"""
Frame stitching for panorama generation.
Primary: OpenCV Stitcher (RANSAC-based feature matching), batched
Fallback: Horizontal concat (resized to common height)

Low-light fix: frames are contrast-enhanced (CLAHE + gamma lift) on the
raw numpy arrays immediately after deduplication — before any JPEG
compression — so the stitcher receives the highest-quality enhanced
input for keypoint detection and alignment.

Batching fix: frames are split into small batches before stitching
to prevent OpenCV's bundle adjuster from consuming excessive memory
and crashing the machine.

Coverage fix: batch boundaries are chosen at natural transitions in the
scan (where histogram correlation between consecutive frames drops,
indicating the camera moved to a new area) rather than at fixed offsets.
Adjacent batches share BATCH_OVERLAP frames so the second-level stitch
always has visual anchors between batches, preventing coverage gaps.

Memory fix: waiting frames are JPEG-compressed in memory (~600 KB each
vs ~6 MB raw) so 76 frames occupy ~45 MB instead of ~456 MB while
batches are processed. Enhancement runs on the raw frames before
compression, so the JPEG only needs to store the already-enhanced image
and no re-enhancement is needed after decoding.
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
BATCH_OVERLAP = 2  # frames shared between adjacent batches for second-level stitch anchoring


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

    # Enhance all frames on the raw numpy arrays before any JPEG compression.
    # Running CLAHE+gamma on the original data gives the stitcher the best
    # possible keypoints; the JPEG only stores the already-enhanced result so
    # no lossy round-trip degrades the data before enhancement.
    with ThreadPoolExecutor() as pool:
        unique = list(pool.map(_enhance_for_stitching, unique))
    logger.info(f"Enhanced {len(unique)} frames on raw data")

    if len(unique) <= BATCH_SIZE:
        return _stitch_batch(unique)

    # Group frames by scan coverage before compressing: split at natural
    # transitions (low inter-frame similarity) so each batch contains frames
    # that actually overlap each other.  Adjacent batches share BATCH_OVERLAP
    # frames as visual anchors for the second-level stitch.
    frame_groups = _group_indices_by_coverage(unique, batch_size=BATCH_SIZE, overlap=BATCH_OVERLAP)
    n_batches = len(frame_groups)
    logger.info(
        f"Stitching in {n_batches} coverage-based batches "
        f"(sizes: {[len(g) for g in frame_groups]})"
    )

    # JPEG-compress all enhanced frames so 76×~6 MB → 76×~600 KB in memory.
    # Only the active batch is decompressed to full-res arrays; no
    # re-enhancement is needed after decoding.
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
            logger.warning(f"Batch {batch_num} stitch failed — using concat fallback for this batch")
            batch_panoramas.append(_concat_fallback(batch))
        del batch
        gc.collect()
    del compressed
    gc.collect()

    if len(batch_panoramas) == 1:
        return batch_panoramas[0], "ok"

    # Stitch the batch panoramas together, also respecting BATCH_SIZE
    logger.info(f"Stitching {len(batch_panoramas)} batch panoramas together…")
    if len(batch_panoramas) <= BATCH_SIZE:
        final, status = _stitch_batch(batch_panoramas)
        if final is not None:
            return final, "ok_batched"
    else:
        # Too many panoramas to stitch at once — batch this level too
        n_meta = math.ceil(len(batch_panoramas) / BATCH_SIZE)
        logger.info(f"Second-level stitch: {n_meta} meta-batches")
        meta_panoramas: List[np.ndarray] = []
        while batch_panoramas:
            meta_batch, batch_panoramas = batch_panoramas[:BATCH_SIZE], batch_panoramas[BATCH_SIZE:]
            pano, _status = _stitch_batch(meta_batch)
            if pano is not None:
                meta_panoramas.append(pano)
            else:
                meta_panoramas.append(_concat_fallback(meta_batch))
            del meta_batch
            gc.collect()
        if len(meta_panoramas) == 1:
            return meta_panoramas[0], "ok_batched"
        final, status = _stitch_batch(meta_panoramas)
        if final is not None:
            return final, "ok_batched"
        logger.warning("Meta-stitch failed — concatenating meta panoramas")
        return _concat_fallback(meta_panoramas), "fallback_concat"

    # Final fallback — concat all batch panoramas
    logger.warning("Final stitch failed — concatenating batch panoramas")
    return _concat_fallback(batch_panoramas), "fallback_concat"


def _stitch_batch(frames: List[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    """Stitch a small batch of pre-enhanced frames using OpenCV stitcher."""
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
