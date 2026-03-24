"""
Incremental panorama stitcher for sequential PTZ camera frames.

Each consecutive frame overlaps its neighbour.  Rather than using OpenCV's
high-level Stitcher (which estimates full homographies and breaks when the
accumulator outgrows a single frame), we:

  1. Enhance every raw frame (CLAHE + gamma) for better feature detection.
  2. Detect ORB features and match consecutive pairs to find the
     translation offset between them.
  3. Accumulate absolute (x, y) positions for every frame.
  4. Composite all frames onto one canvas with linear-blend seams.

The result is a single large image built from the *original* (un-enhanced)
frames so no data is lost.  The final output is returned as a numpy array
and can be compressed to JPEG at the caller's discretion.

For very large runs (>100 frames) the pipeline works identically — offsets
are O(n) pairwise computations and the final composite is a single pass.
"""

import base64
import gc
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enhancement (applied to raw frames *before* feature detection)
# ---------------------------------------------------------------------------

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """
    Boost contrast + brightness so ORB finds keypoints in dark sanctuary
    frames.  CLAHE on the L channel of LAB + a gamma 2.0 lift.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    lut = np.array(
        [((i / 255.0) ** (1.0 / 2.0)) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(enhanced, lut)


# ---------------------------------------------------------------------------
# Pairwise offset estimation
# ---------------------------------------------------------------------------

_ORB_FEATURES = 3000          # keypoints per frame
_MATCH_RATIO  = 0.75          # Lowe's ratio test threshold
_RANSAC_REPROJ = 5.0          # RANSAC reprojection threshold (px)
_MIN_INLIERS  = 8             # minimum matches to trust an offset


def _estimate_offset(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Estimate the (dx, dy) translation that maps *frame_b* onto the
    coordinate system of *frame_a*.  Returns None if matching fails.

    Positive dx → frame_b is to the right of frame_a.
    Positive dy → frame_b is below frame_a.
    """
    # Work on grayscale, down-scaled for speed
    h_a, w_a = frame_a.shape[:2]
    h_b, w_b = frame_b.shape[:2]
    scale = min(1.0, 800.0 / max(h_a, w_a, h_b, w_b))

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray_a = cv2.resize(gray_a, None, fx=scale, fy=scale)
        gray_b = cv2.resize(gray_b, None, fx=scale, fy=scale)

    orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
    kp_a, des_a = orb.detectAndCompute(gray_a, None)
    kp_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return None

    # Brute-force Hamming match + Lowe's ratio test
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)
    raw_matches = bf.knnMatch(des_a, des_b, k=2)

    good = []
    for m_pair in raw_matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < _MATCH_RATIO * n.distance:
                good.append(m)

    if len(good) < _MIN_INLIERS:
        return None

    pts_a = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    pts_b = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # Estimate affine (translation + rotation + scale) via RANSAC
    # We only really expect translation, but allowing affine handles minor
    # camera rotation between presets.
    M, inliers = cv2.estimateAffinePartial2D(
        pts_b, pts_a, method=cv2.RANSAC, ransacReprojThreshold=_RANSAC_REPROJ
    )

    if M is None:
        return None

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    if n_inliers < _MIN_INLIERS:
        return None

    # Translation component of the 2×3 affine matrix, scaled back up
    dx = M[0, 2] / scale
    dy = M[1, 2] / scale

    logger.debug(
        f"Offset: dx={dx:.1f}  dy={dy:.1f}  "
        f"({n_inliers}/{len(good)} inliers, scale={M[0,0]:.3f})"
    )
    return (dx, dy)


def _template_offset(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> Optional[Tuple[float, float]]:
    """
    Fallback offset estimation using template matching when ORB fails.
    Uses a center crop of frame_b as a template searched within frame_a's
    extended region.
    """
    h_a, w_a = frame_a.shape[:2]
    h_b, w_b = frame_b.shape[:2]

    # Downscale for speed
    scale = min(1.0, 600.0 / max(h_a, w_a, h_b, w_b))
    if scale < 1.0:
        a = cv2.resize(frame_a, None, fx=scale, fy=scale)
        b = cv2.resize(frame_b, None, fx=scale, fy=scale)
    else:
        a, b = frame_a, frame_b

    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    sh_a, sw_a = gray_a.shape
    sh_b, sw_b = gray_b.shape

    # Use a generous center crop of B as the template (40% of each dimension)
    crop_h = max(32, int(sh_b * 0.4))
    crop_w = max(32, int(sw_b * 0.4))
    y0 = (sh_b - crop_h) // 2
    x0 = (sw_b - crop_w) // 2
    template = gray_b[y0:y0 + crop_h, x0:x0 + crop_w]

    if template.shape[0] > gray_a.shape[0] or template.shape[1] > gray_a.shape[1]:
        return None

    res = cv2.matchTemplate(gray_a, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, max_loc = cv2.minMaxLoc(res)

    if max_val < 0.25:
        return None

    # max_loc is where the template's top-left was found in A.
    # The template was taken from (x0, y0) in B.
    # So B's origin in A's coordinate space is at (max_loc[0] - x0, max_loc[1] - y0).
    dx = (max_loc[0] - x0) / scale
    dy = (max_loc[1] - y0) / scale

    logger.debug(f"Template offset: dx={dx:.1f}  dy={dy:.1f}  confidence={max_val:.3f}")
    return (dx, dy)


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(frames: List[np.ndarray], threshold: float = 0.97) -> List[np.ndarray]:
    """Remove near-duplicate consecutive frames based on histogram correlation."""
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


# ---------------------------------------------------------------------------
# Canvas compositing with linear blending
# ---------------------------------------------------------------------------

def _composite(
    frames: List[np.ndarray],
    positions: List[Tuple[float, float]],
) -> np.ndarray:
    """
    Place every frame at its absolute (x, y) position on a single canvas.
    Where frames overlap, use distance-based alpha blending so seams are
    invisible.
    """
    # Compute canvas bounds
    xs, ys = [], []
    for (x, y), frame in zip(positions, frames):
        h, w = frame.shape[:2]
        xs.extend([x, x + w])
        ys.extend([y, y + h])

    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    logger.info(f"Canvas size: {canvas_w} × {canvas_h}")

    # Accumulate weighted colour and weight for linear blending
    # Use float32 for accumulation
    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float64)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float64)

    for (x, y), frame in zip(positions, frames):
        h, w = frame.shape[:2]
        ox = int(round(x - min_x))
        oy = int(round(y - min_y))

        # Clamp to canvas (shouldn't be needed, but safety)
        pw = min(w, canvas_w - ox)
        ph = min(h, canvas_h - oy)
        if pw <= 0 or ph <= 0:
            continue

        # Per-pixel weight: ramp from edges so overlapping regions blend smoothly
        # Weight = min(distance from each edge) — higher in the centre
        ramp_x = np.minimum(
            np.arange(pw, dtype=np.float64),
            np.arange(pw, 0, -1, dtype=np.float64),
        ) + 1.0
        ramp_y = np.minimum(
            np.arange(ph, dtype=np.float64),
            np.arange(ph, 0, -1, dtype=np.float64),
        ) + 1.0
        w_map = ramp_y[:, None] * ramp_x[None, :]  # (ph, pw)

        roi = frame[:ph, :pw].astype(np.float64)
        canvas[oy:oy + ph, ox:ox + pw] += roi * w_map[:, :, None]
        weight[oy:oy + ph, ox:ox + pw] += w_map

    # Normalise
    mask = weight > 0
    for c in range(3):
        canvas[:, :, c][mask] /= weight[mask]

    return canvas.astype(np.uint8)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def stitch_frames(
    frames: List[np.ndarray],
    grid_shape: Optional[Tuple[int, int]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Stitch a sequence of overlapping frames into a single panorama.

    Algorithm:
      1. Deduplicate near-identical consecutive frames.
      2. Enhance each frame (CLAHE + gamma) for feature detection.
      3. Compute pairwise translation offsets between consecutive enhanced
         frames using ORB feature matching (with template-match fallback).
      4. Accumulate absolute positions.
      5. Composite *original* (un-enhanced) frames onto one canvas with
         distance-weighted blending.

    grid_shape is accepted for API compatibility but ignored — sequential
    registration handles any scan pattern.

    Returns (panorama_image, status_string).
    """
    if not frames:
        return None, "no_frames"

    if len(frames) == 1:
        return frames[0], "single_frame"

    # Step 1: deduplicate
    unique = _deduplicate(frames)
    n = len(unique)
    logger.info(f"Stitching {n} unique frames (from {len(frames)} captured)")

    if n == 1:
        return unique[0], "single_frame"

    # Step 2: enhance for feature detection
    logger.info("Enhancing frames for feature detection…")
    enhanced = [enhance_frame(f) for f in unique]

    # Step 3: compute pairwise offsets
    logger.info("Computing pairwise offsets…")
    positions: List[Tuple[float, float]] = [(0.0, 0.0)]
    failed_pairs = 0

    for i in range(len(enhanced) - 1):
        offset = _estimate_offset(enhanced[i], enhanced[i + 1])

        if offset is None:
            # Fallback: template matching on enhanced frames
            offset = _template_offset(enhanced[i], enhanced[i + 1])

        if offset is None:
            # Last resort: try on originals
            offset = _estimate_offset(unique[i], unique[i + 1])

        if offset is None:
            offset = _template_offset(unique[i], unique[i + 1])

        if offset is None:
            # Cannot determine offset — place to the right with slight overlap
            h, w = unique[i + 1].shape[:2]
            prev_w = unique[i].shape[1]
            overlap_guess = int(prev_w * 0.15)
            offset = (prev_w - overlap_guess, 0)
            failed_pairs += 1
            logger.warning(
                f"  Pair {i}→{i+1}: all matching failed, guessing horizontal placement"
            )

        dx, dy = offset
        prev_x, prev_y = positions[-1]
        positions.append((prev_x + dx, prev_y + dy))

        if (i + 1) % 10 == 0 or i == len(enhanced) - 2:
            logger.info(f"  Offset {i+1}/{len(enhanced)-1} done")

        gc.collect()

    # Free enhanced frames — we only need originals for compositing
    del enhanced
    gc.collect()

    if failed_pairs > 0:
        logger.warning(f"{failed_pairs}/{n-1} pairs used fallback placement")

    # Step 4: composite originals onto canvas
    logger.info("Compositing panorama…")
    panorama = _composite(unique, positions)

    # Step 5: crop any black border
    panorama = _crop_black(panorama)

    logger.info(f"Stitch complete — output shape: {panorama.shape}")
    status = "ok" if failed_pairs == 0 else f"ok_with_{failed_pairs}_fallbacks"
    return panorama, status


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _crop_black(img: np.ndarray) -> np.ndarray:
    """Crop black borders that may appear from canvas compositing."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img
    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
    return img[y:y + h, x:x + w]


def to_base64(frame: np.ndarray, quality: int = 85) -> str:
    """Encode frame as base64 JPEG."""
    _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
    return base64.b64encode(buf).decode("utf-8")


def from_base64(b64: str) -> np.ndarray:
    """Decode base64 JPEG to frame."""
    arr = np.frombuffer(base64.b64decode(b64), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)
