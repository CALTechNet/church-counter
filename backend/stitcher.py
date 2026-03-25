"""
Panorama stitcher for sequential PTZ camera frames.

Supports two modes:
1. Grid-aware stitching (when grid_shape is provided): arranges frames into
   a 2D grid matching the camera's boustrophedon scan pattern, stitches each
   row horizontally, then stitches rows vertically.  This avoids the failure
   mode where column-boundary frames have no overlap but the sequential
   stitcher tries to chain them anyway.
2. Sequential stitching (fallback): chains all frames linearly as before.

Uses affine transforms (not full perspective homographies) to prevent
accumulated perspective distortion across many frames.
"""

import base64
import gc
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Boost contrast + brightness so ORB finds keypoints in dark frames."""
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


def _brighten_output(img: np.ndarray) -> np.ndarray:
    """Apply uniform brightness/contrast boost to the final panorama output.

    Uses a global gamma correction for even brightness across the whole
    stitched image, plus CLAHE with a large tile grid scaled to the panorama
    size so that brightness is consistent across seams.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Scale CLAHE tile grid to panorama size so tiles are ~256px each,
    # giving smooth, even local contrast without visible tile boundaries.
    h, w = l.shape[:2]
    tile_w = max(1, w // 256)
    tile_h = max(1, h // 256)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_w, tile_h))
    l = clahe.apply(l)

    # Global gamma correction on L channel for uniform brightening
    lut = np.array(
        [((i / 255.0) ** (1.0 / 1.8)) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    l = cv2.LUT(l, lut)
    return cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)


# ---------------------------------------------------------------------------
# Pairwise affine estimation
# ---------------------------------------------------------------------------

_ORB_FEATURES  = 3000
_MATCH_RATIO   = 0.75
_RANSAC_REPROJ = 5.0
_MIN_INLIERS   = 10
_PHASE_CORR_MIN = 0.05          # minimum phase-correlation response to accept


def _estimate_affine(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Estimate the 2×3 affine transform that maps *frame_b* into *frame_a*'s
    coordinate system.  Returns a 3×3 matrix (with [0,0,1] bottom row) or
    None if matching fails.
    """
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

    A, inliers = cv2.estimateAffine2D(
        pts_b, pts_a, method=cv2.RANSAC, ransacReprojThreshold=_RANSAC_REPROJ,
    )

    if A is None:
        return None

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    if n_inliers < _MIN_INLIERS:
        return None

    # Scale the affine transform back to original resolution
    A_3x3 = np.vstack([A, [0, 0, 1]])
    S_full = np.array(
        [[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1]], dtype=np.float64,
    )
    S_inv = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)
    H_full = S_full @ A_3x3 @ S_inv

    logger.debug(
        f"Affine: {n_inliers}/{len(good)} inliers, "
        f"dx≈{H_full[0, 2]:.1f}, dy≈{H_full[1, 2]:.1f}"
    )
    return H_full


def _template_affine(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> Optional[np.ndarray]:
    """Fallback: pure-translation affine via template matching."""
    h_a, w_a = frame_a.shape[:2]
    h_b, w_b = frame_b.shape[:2]

    scale = min(1.0, 600.0 / max(h_a, w_a, h_b, w_b))
    if scale < 1.0:
        a = cv2.resize(frame_a, None, fx=scale, fy=scale)
        b = cv2.resize(frame_b, None, fx=scale, fy=scale)
    else:
        a, b = frame_a, frame_b

    gray_a = cv2.cvtColor(a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(b, cv2.COLOR_BGR2GRAY)
    sh_b, sw_b = gray_b.shape

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

    dx = (max_loc[0] - x0) / scale
    dy = (max_loc[1] - y0) / scale

    H = np.eye(3, dtype=np.float64)
    H[0, 2] = dx
    H[1, 2] = dy

    logger.debug(
        f"Template affine: dx={dx:.1f} dy={dy:.1f} conf={max_val:.3f}"
    )
    return H


def _phase_correlation_affine(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    direction: str = "horizontal",
) -> Optional[np.ndarray]:
    """Translation-only matching via phase correlation.

    Phase correlation uses the global frequency content of the overlap region,
    making it far more robust than local feature matching on dark, repetitive
    textures (e.g. rows of identical chairs in a dim church).
    """
    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY).astype(np.float64)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY).astype(np.float64)

    # Match dimensions
    h = min(gray_a.shape[0], gray_b.shape[0])
    w = min(gray_a.shape[1], gray_b.shape[1])
    gray_a = gray_a[:h, :w]
    gray_b = gray_b[:h, :w]

    # Hann window to suppress edge artifacts in FFT
    window = cv2.createHanningWindow((w, h), cv2.CV_64F)
    gray_a = gray_a * window
    gray_b = gray_b * window

    shift, response = cv2.phaseCorrelate(gray_a, gray_b)

    if response < _PHASE_CORR_MIN:
        return None

    dx, dy = shift

    # Sanity: for horizontal stitching, dx should be significant and positive
    # For vertical stitching, dy should be significant and positive
    if direction == "horizontal":
        if abs(dx) < w * 0.05:      # too small — likely a phantom peak
            return None
    else:
        if abs(dy) < h * 0.05:
            return None

    H = np.eye(3, dtype=np.float64)
    H[0, 2] = dx
    H[1, 2] = dy

    logger.debug(
        f"Phase correlation: dx={dx:.1f} dy={dy:.1f} response={response:.4f}"
    )
    return H


# ---------------------------------------------------------------------------
# Affine validation
# ---------------------------------------------------------------------------

def _is_valid_affine(H: np.ndarray, frame_shape: Tuple[int, ...]) -> bool:
    """Reject affine transforms that produce extreme distortion."""
    h, w = frame_shape[:2]
    corners = np.float32(
        [[0, 0], [w, 0], [w, h], [0, h]]
    ).reshape(-1, 1, 2)

    A = H[:2, :]
    warped = cv2.transform(corners, A).reshape(-1, 2)

    n = len(warped)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += warped[i][0] * warped[j][1]
        area -= warped[j][0] * warped[i][1]
    area /= 2.0

    original_area = float(h * w)
    ratio = abs(area) / original_area

    if area < 0 or ratio < 0.5 or ratio > 2.0:
        return False

    orig_corners = corners.reshape(-1, 2)
    for i in range(4):
        j = (i + 1) % 4
        edge = float(np.linalg.norm(warped[i] - warped[j]))
        orig = float(np.linalg.norm(orig_corners[i] - orig_corners[j]))
        if orig > 0 and (edge / orig > 2.0 or edge / orig < 0.3):
            return False

    return True


# ---------------------------------------------------------------------------
# Deduplication
# ---------------------------------------------------------------------------

def _deduplicate(
    frames: List[np.ndarray], threshold: float = 0.97,
) -> List[np.ndarray]:
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
# Pairwise affine with fallbacks (used for both row and column stitching)
# ---------------------------------------------------------------------------

def _direction_ok(
    H: np.ndarray,
    direction: str,
    frame_shape: Tuple[int, ...],
) -> bool:
    """Check that the dominant translation axis matches the expected direction."""
    dx, dy = abs(H[0, 2]), abs(H[1, 2])
    h, w = frame_shape[:2]
    if direction == "horizontal":
        # dx should be the dominant component and at least 10% of frame width
        return dx > w * 0.10 and dx > dy * 0.5
    else:
        return dy > h * 0.10 and dy > dx * 0.5


def _match_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    enhanced_a: np.ndarray,
    enhanced_b: np.ndarray,
    fallback_direction: str = "horizontal",
    expected_offset: Optional[Tuple[float, float]] = None,
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Try to match a pair of frames.  Returns (H, used_fallback).
    fallback_direction controls the last-resort translation guess:
      "horizontal" → shift right by 85% of frame width
      "vertical"   → shift down by 85% of frame height

    When expected_offset=(dx, dy) is provided (from known camera positions),
    it is used to validate matches and as a better final fallback.
    """
    candidates: list = []

    # 1. ORB on enhanced frames
    H = _estimate_affine(enhanced_a, enhanced_b)
    if H is not None and not _is_valid_affine(H, frame_b.shape):
        H = None
    if H is not None and _direction_ok(H, fallback_direction, frame_b.shape):
        candidates.append(("orb_enhanced", H))

    # 2. Template matching on enhanced frames
    H = _template_affine(enhanced_a, enhanced_b)
    if H is not None and _direction_ok(H, fallback_direction, frame_b.shape):
        candidates.append(("template_enhanced", H))

    # 3. Phase correlation (very robust for repetitive textures)
    H = _phase_correlation_affine(frame_a, frame_b, direction=fallback_direction)
    if H is not None:
        candidates.append(("phase_corr", H))

    # 4. ORB on raw frames
    if not candidates:
        H = _estimate_affine(frame_a, frame_b)
        if H is not None and not _is_valid_affine(H, frame_b.shape):
            H = None
        if H is not None and _direction_ok(H, fallback_direction, frame_b.shape):
            candidates.append(("orb_raw", H))

    # 5. Template matching on raw frames
    if not candidates:
        H = _template_affine(frame_a, frame_b)
        if H is not None and _direction_ok(H, fallback_direction, frame_b.shape):
            candidates.append(("template_raw", H))

    # If we have an expected offset, pick the candidate closest to it
    if expected_offset is not None and candidates:
        ex_dx, ex_dy = expected_offset

        def offset_err(name_h):
            _, h_mat = name_h
            return abs(h_mat[0, 2] - ex_dx) + abs(h_mat[1, 2] - ex_dy)

        best_name, best_H = min(candidates, key=offset_err)
        logger.debug(
            f"  Best match: {best_name} "
            f"(dx={best_H[0,2]:.0f}, dy={best_H[1,2]:.0f}; "
            f"expected dx={ex_dx:.0f}, dy={ex_dy:.0f})"
        )
        return best_H, False
    elif candidates:
        # No expected offset — prefer the first (highest-priority) candidate
        best_name, best_H = candidates[0]
        logger.debug(f"  Best match: {best_name} (dx={best_H[0,2]:.0f}, dy={best_H[1,2]:.0f})")
        return best_H, False

    # Last resort: use expected offset if available, else generic guess
    h, w = frame_a.shape[:2]
    H = np.eye(3, dtype=np.float64)
    if expected_offset is not None:
        H[0, 2], H[1, 2] = expected_offset
        logger.warning(
            f"  All matching failed — using expected offset "
            f"(dx={expected_offset[0]:.0f}, dy={expected_offset[1]:.0f})"
        )
    elif fallback_direction == "vertical":
        overlap_guess = int(h * 0.15)
        H[1, 2] = h - overlap_guess
    else:
        overlap_guess = int(w * 0.15)
        H[0, 2] = w - overlap_guess
    return H, True


# ---------------------------------------------------------------------------
# Canvas compositing with affine warping and blending
# ---------------------------------------------------------------------------

def _compute_gain_compensation(
    frames: List[np.ndarray],
    transforms: List[np.ndarray],
    canvas_w: int,
    canvas_h: int,
    T: np.ndarray,
) -> np.ndarray:
    """Compute per-frame gain factors so overlapping regions have matched brightness.

    For each pair of overlapping frames we measure the mean intensity ratio in
    the overlap zone and solve a least-squares system to find a single scalar
    gain per frame that minimises brightness differences across all overlaps.
    Returns an array of shape (n_frames,) with gain multipliers.
    """
    n = len(frames)
    if n <= 1:
        return np.ones(n, dtype=np.float64)

    # Build small masks (quarter-res) for speed
    scale = 0.25
    sw = max(1, int(canvas_w * scale))
    sh = max(1, int(canvas_h * scale))
    S = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)

    gray_warped = []
    masks = []
    kernel = np.ones((3, 3), np.uint8)

    for frame, H in zip(frames, transforms):
        H_final = S @ T @ H
        A_final = H_final[:2, :]

        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        w_g = cv2.warpAffine(g, A_final, (sw, sh),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)

        m = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        w_m = cv2.warpAffine(m, A_final, (sw, sh),
                             flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        valid = (w_m > 128).astype(np.uint8)
        valid = cv2.erode(valid, kernel, iterations=1)

        gray_warped.append(w_g.astype(np.float64))
        masks.append(valid)

    # Build equations: for each overlapping pair, gain[i]*mean_i ≈ gain[j]*mean_j
    # We solve via least-squares with the constraint that mean gain = 1.
    A_rows = []
    b_rows = []

    for i in range(n):
        for j in range(i + 1, n):
            overlap = masks[i] & masks[j]
            count = int(overlap.sum())
            if count < 100:
                continue
            mean_i = gray_warped[i][overlap > 0].mean()
            mean_j = gray_warped[j][overlap > 0].mean()
            if mean_i < 1 or mean_j < 1:
                continue
            # We want gain[i]*mean_i = gain[j]*mean_j
            # => gain[i]*mean_i - gain[j]*mean_j = 0
            row = np.zeros(n, dtype=np.float64)
            row[i] = mean_i
            row[j] = -mean_j
            A_rows.append(row)
            b_rows.append(0.0)

    if not A_rows:
        return np.ones(n, dtype=np.float64)

    # Add soft constraint: sum(gain) = n (mean gain = 1)
    constraint = np.ones(n, dtype=np.float64)
    A_rows.append(constraint * 10.0)  # high weight
    b_rows.append(n * 10.0)

    A_mat = np.array(A_rows)
    b_vec = np.array(b_rows)

    gains, _, _, _ = np.linalg.lstsq(A_mat, b_vec, rcond=None)

    # Clamp to reasonable range
    gains = np.clip(gains, 0.5, 2.0)
    # Renormalize so mean is 1
    gains /= gains.mean()

    logger.info(f"Gain compensation: min={gains.min():.3f} max={gains.max():.3f}")
    return gains


def _composite_affine(
    frames: List[np.ndarray],
    transforms: List[np.ndarray],
) -> np.ndarray:
    """
    Warp every frame using its accumulated affine transform and blend onto a
    single canvas.  Uses gain compensation to match brightness across frames
    and power-weighted distance-transform blending for seamless seams.
    """
    all_corners = []
    for frame, H in zip(frames, transforms):
        h, w = frame.shape[:2]
        corners = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]]
        ).reshape(-1, 1, 2)
        A = H[:2, :]
        warped_corners = cv2.transform(corners, A)
        all_corners.append(warped_corners.reshape(-1, 2))

    all_pts = np.vstack(all_corners)
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)

    T = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64,
    )

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    max_dim = 16000
    if max(canvas_w, canvas_h) > max_dim:
        s = max_dim / max(canvas_w, canvas_h)
        S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
        T = S @ T
        canvas_w = int(canvas_w * s)
        canvas_h = int(canvas_h * s)

    logger.info(f"Canvas size: {canvas_w} x {canvas_h}")

    # Compute per-frame gain compensation to match brightness across frames
    gains = _compute_gain_compensation(frames, transforms, canvas_w, canvas_h, T)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for i, (frame, H) in enumerate(zip(frames, transforms)):
        H_final = T @ H
        A_final = H_final[:2, :]

        warped = cv2.warpAffine(
            frame, A_final, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Apply gain compensation
        warped = np.clip(warped.astype(np.float32) * gains[i], 0, 255).astype(np.uint8)

        frame_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        warped_mask = cv2.warpAffine(
            frame_mask, A_final, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = (warped_mask > 128).astype(np.uint8)

        # Larger erosion to remove edge artifacts and vignetting
        kernel = np.ones((5, 5), np.uint8)
        valid = cv2.erode(valid, kernel, iterations=4)

        dist = cv2.distanceTransform(valid, cv2.DIST_L2, 5)

        # Power-weight the distance so centre pixels dominate more strongly;
        # this makes the transition zone narrower and hides residual brightness
        # differences better than linear weighting.
        dist = np.power(dist, 2.0)

        canvas += warped.astype(np.float32) * dist[:, :, None]
        weight += dist

        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            logger.info(f"  Warped {i + 1}/{len(frames)} frames")

    mask = weight > 0
    for c in range(3):
        canvas[:, :, c][mask] /= weight[mask]

    return canvas.astype(np.uint8)


# ---------------------------------------------------------------------------
# Sequential stitching (no grid info)
# ---------------------------------------------------------------------------

def _stitch_sequential(frames: List[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    """Stitch frames as a linear chain (original algorithm)."""
    unique = _deduplicate(frames)
    n = len(unique)
    logger.info(f"Sequential stitch: {n} unique frames (from {len(frames)} captured)")

    if n == 0:
        return None, "no_frames"
    if n == 1:
        return unique[0], "single_frame"

    enhanced = [enhance_frame(f) for f in unique]

    logger.info("Computing pairwise affine transforms...")
    pairwise: List[np.ndarray] = []
    failed_pairs = 0

    for i in range(n - 1):
        H, failed = _match_pair(
            unique[i], unique[i + 1],
            enhanced[i], enhanced[i + 1],
            fallback_direction="horizontal",
        )
        if failed:
            failed_pairs += 1
            logger.warning(f"  Pair {i}->{i+1}: all matching failed, guessing placement")
        pairwise.append(H)

        if (i + 1) % 10 == 0 or i == n - 2:
            logger.info(f"  Affine {i+1}/{n-1} done")
        gc.collect()

    del enhanced
    gc.collect()

    if failed_pairs > 0:
        logger.warning(f"{failed_pairs}/{n-1} pairs used fallback placement")

    # Chain into absolute transforms
    H_abs: List[np.ndarray] = [np.eye(3, dtype=np.float64)]
    for i in range(n - 1):
        H_abs.append(H_abs[-1] @ pairwise[i])

    # Re-reference to middle frame
    ref = n // 2
    H_ref_inv = np.linalg.inv(H_abs[ref])
    H_abs = [H_ref_inv @ H for H in H_abs]

    logger.info("Compositing panorama...")
    panorama = _composite_affine(unique, H_abs)
    panorama = _crop_black(panorama)

    status = "ok" if failed_pairs == 0 else f"ok_with_{failed_pairs}_fallbacks"
    return panorama, status


# ---------------------------------------------------------------------------
# Strip stitching helper (stitch a 1D list of overlapping frames)
# ---------------------------------------------------------------------------

def _estimate_expected_offsets(
    positions: Optional[List[Tuple[float, float]]],
    frames: List[np.ndarray],
    direction: str,
) -> List[Optional[Tuple[float, float]]]:
    """Compute expected pixel offsets between consecutive frames from
    known pan/tilt positions.

    Returns a list of (dx, dy) tuples, one per consecutive pair.
    If positions are not available, returns a list of Nones.

    The conversion from pan/tilt units to pixels is estimated by assuming
    the total position range maps to roughly ``(1 - overlap) * n_frames``
    frame widths/heights, with ~35 % overlap.
    """
    n = len(frames)
    if positions is None or len(positions) != n:
        return [None] * max(0, n - 1)

    h, w = frames[0].shape[:2]
    overlap = 0.35

    if direction == "horizontal":
        # Compute pan range
        pans = [p[0] for p in positions]
        pan_range = max(pans) - min(pans)
        if pan_range < 1:
            return [None] * (n - 1)
        # Total panorama width in pixels ≈ n columns with overlap
        pixels_per_pan = (w * (1 - overlap)) * (n - 1) / pan_range if pan_range else 1
        offsets = []
        for i in range(n - 1):
            dp = positions[i + 1][0] - positions[i][0]
            dt = positions[i + 1][1] - positions[i][1]
            offsets.append((dp * pixels_per_pan, dt * pixels_per_pan))
        return offsets
    else:
        # Compute tilt range
        tilts = [p[1] for p in positions]
        tilt_range = max(tilts) - min(tilts)
        if tilt_range < 1:
            return [None] * (n - 1)
        pixels_per_tilt = (h * (1 - overlap)) * (n - 1) / tilt_range if tilt_range else 1
        offsets = []
        for i in range(n - 1):
            dp = positions[i + 1][0] - positions[i][0]
            dt = positions[i + 1][1] - positions[i][1]
            offsets.append((dp * pixels_per_tilt, dt * pixels_per_tilt))
        return offsets


def _stitch_strip(
    frames: List[np.ndarray],
    direction: str = "horizontal",
    positions: Optional[List[Tuple[float, float]]] = None,
) -> Optional[np.ndarray]:
    """
    Stitch a list of frames that overlap in the given direction.
    Returns a single composited image, or None if empty.

    When *positions* (list of (pan, tilt) per frame) is provided, expected
    pixel offsets are computed from the known camera geometry and used to
    validate matches and as a better fallback than the generic 85 % guess.
    """
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]

    n = len(frames)
    enhanced = [enhance_frame(f) for f in frames]

    expected_offsets = _estimate_expected_offsets(positions, frames, direction)

    pairwise: List[np.ndarray] = []
    for i in range(n - 1):
        H, failed = _match_pair(
            frames[i], frames[i + 1],
            enhanced[i], enhanced[i + 1],
            fallback_direction=direction,
            expected_offset=expected_offsets[i],
        )
        if failed:
            logger.warning(f"  Strip pair {i}->{i+1}: fallback ({direction})")
        pairwise.append(H)

    del enhanced
    gc.collect()

    # Chain into absolute transforms
    H_abs: List[np.ndarray] = [np.eye(3, dtype=np.float64)]
    for i in range(n - 1):
        H_abs.append(H_abs[-1] @ pairwise[i])

    # Re-reference to middle
    ref = n // 2
    H_ref_inv = np.linalg.inv(H_abs[ref])
    H_abs = [H_ref_inv @ H for H in H_abs]

    result = _composite_affine(frames, H_abs)
    return _crop_black(result)


# ---------------------------------------------------------------------------
# Grid-aware stitching
# ---------------------------------------------------------------------------

def _arrange_boustrophedon(
    frames: List[np.ndarray],
    rows: int,
    cols: int,
) -> List[List[Optional[np.ndarray]]]:
    """
    Arrange frames from boustrophedon (vertical-S) scan order into a 2D grid.

    Camera scan order: column-major, even columns top->bottom, odd columns
    bottom->top.  We rearrange into grid[row][col] in standard raster order.
    """
    grid: List[List[Optional[np.ndarray]]] = [
        [None] * cols for _ in range(rows)
    ]

    idx = 0
    for col in range(cols):
        for row_idx in range(rows):
            if idx >= len(frames):
                break
            # Even columns: top->bottom (row_idx = actual row)
            # Odd columns: bottom->top (row_idx maps to rows-1-row_idx)
            if col % 2 == 0:
                actual_row = row_idx
            else:
                actual_row = rows - 1 - row_idx
            grid[actual_row][col] = frames[idx]
            idx += 1

    return grid


def _arrange_boustrophedon_positions(
    positions: List[Tuple[float, float]],
    rows: int,
    cols: int,
) -> List[List[Optional[Tuple[float, float]]]]:
    """Arrange positions into the same 2D grid as frames (mirrors _arrange_boustrophedon)."""
    grid: List[List[Optional[Tuple[float, float]]]] = [
        [None] * cols for _ in range(rows)
    ]
    idx = 0
    for col in range(cols):
        for row_idx in range(rows):
            if idx >= len(positions):
                break
            if col % 2 == 0:
                actual_row = row_idx
            else:
                actual_row = rows - 1 - row_idx
            grid[actual_row][col] = positions[idx]
            idx += 1
    return grid


def _stitch_grid(
    frames: List[np.ndarray],
    rows: int,
    cols: int,
    positions: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Grid-aware stitching: arrange frames into a 2D grid, stitch each row
    horizontally, then stitch the resulting row-strips vertically.

    When *positions* (flat list of (pan, tilt) in scan order) is provided,
    the known camera geometry is used to compute expected pixel offsets,
    dramatically improving alignment on dark / repetitive scenes.
    """
    logger.info(f"Grid-aware stitch: {rows} rows x {cols} cols, {len(frames)} frames")

    # Arrange into grid
    grid = _arrange_boustrophedon(frames, rows, cols)

    pos_grid = None
    if positions is not None and len(positions) == len(frames):
        pos_grid = _arrange_boustrophedon_positions(positions, rows, cols)
        logger.info("Using known camera positions for alignment guidance")

    # Log grid occupancy
    for r in range(rows):
        occupied = sum(1 for c in range(cols) if grid[r][c] is not None)
        logger.info(f"  Row {r}: {occupied}/{cols} frames")

    # Stitch each row horizontally
    logger.info("Stitching rows horizontally...")
    row_strips: List[np.ndarray] = []
    row_strip_positions: List[Optional[Tuple[float, float]]] = []
    for r in range(rows):
        row_frames = [grid[r][c] for c in range(cols) if grid[r][c] is not None]
        row_pos = None
        if pos_grid is not None:
            row_pos = [pos_grid[r][c] for c in range(cols) if grid[r][c] is not None]
        if not row_frames:
            logger.warning(f"  Row {r}: no frames, skipping")
            continue
        logger.info(f"  Row {r}: stitching {len(row_frames)} frames horizontally")
        strip = _stitch_strip(row_frames, direction="horizontal", positions=row_pos)
        if strip is not None:
            row_strips.append(strip)
            # For vertical stitching, use the mean position of each row
            if row_pos:
                mean_pan = np.mean([p[0] for p in row_pos])
                mean_tilt = np.mean([p[1] for p in row_pos])
                row_strip_positions.append((mean_pan, mean_tilt))
        gc.collect()

    if not row_strips:
        return None, "no_frames"
    if len(row_strips) == 1:
        return row_strips[0], "single_row"

    # Stitch row strips vertically
    logger.info(f"Stitching {len(row_strips)} row strips vertically...")
    vert_pos = row_strip_positions if len(row_strip_positions) == len(row_strips) else None
    panorama = _stitch_strip(row_strips, direction="vertical", positions=vert_pos)

    if panorama is None:
        return None, "no_frames"

    panorama = _crop_black(panorama)
    logger.info(f"Grid stitch complete — output shape: {panorama.shape}")
    return panorama, "ok"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def stitch_frames(
    frames: List[np.ndarray],
    grid_shape: Optional[Tuple[int, int]] = None,
    positions: Optional[List[Tuple[float, float]]] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Stitch a sequence of overlapping frames into a single panorama.

    When grid_shape=(rows, cols) is provided, uses grid-aware stitching
    that properly handles the boustrophedon scan pattern by stitching
    rows first, then stacking vertically.

    When *positions* (list of (pan, tilt) per frame in scan order) is
    provided alongside grid_shape, the known camera geometry guides
    alignment — reducing ghosting on dark / repetitive scenes.

    When grid_shape is None, falls back to sequential chain stitching.

    Returns (panorama_image, status_string).
    """
    if not frames:
        return None, "no_frames"

    if len(frames) == 1:
        return frames[0], "single_frame"

    if grid_shape is not None:
        rows, cols = grid_shape
        if rows > 0 and cols > 0 and len(frames) >= rows * cols * 0.5:
            panorama, status = _stitch_grid(frames, rows, cols, positions=positions)
        else:
            logger.warning(
                f"Grid shape {grid_shape} doesn't match frame count "
                f"{len(frames)}, falling back to sequential"
            )
            panorama, status = _stitch_sequential(frames)
    else:
        panorama, status = _stitch_sequential(frames)

    # Apply brightness correction to final output
    if panorama is not None:
        panorama = _brighten_output(panorama)

    return panorama, status


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def _crop_black(img: np.ndarray) -> np.ndarray:
    """Crop black borders from panorama."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE,
    )
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
