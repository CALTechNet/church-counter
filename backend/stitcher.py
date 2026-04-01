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
import os
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# CUDA device detection — probe once at import time
# ---------------------------------------------------------------------------

_USE_CUDA = False
_cuda_stream: Optional[cv2.cuda.Stream] = None  # type: ignore[attr-defined]

try:
    if hasattr(cv2, "cuda") and cv2.cuda.getCudaEnabledDeviceCount() > 0:
        cv2.cuda.setDevice(0)
        _cuda_stream = cv2.cuda.Stream()
        # Smoke-test: upload a tiny mat and download it
        _test = cv2.cuda.GpuMat()
        _test.upload(np.zeros((2, 2), dtype=np.uint8))
        _ = _test.download()
        del _test
        _USE_CUDA = True
        logger.info(
            "CUDA OpenCV enabled — device 0: %s",
            cv2.cuda.getDevice(),
        )
    else:
        logger.info("No CUDA-enabled OpenCV detected; stitcher will use CPU")
except Exception as exc:
    logger.warning("CUDA probe failed (%s); stitcher falling back to CPU", exc)


def _to_gpu(mat: np.ndarray) -> cv2.cuda.GpuMat:  # type: ignore[attr-defined]
    """Upload a numpy array to GPU memory."""
    gpu = cv2.cuda.GpuMat()
    gpu.upload(mat)
    return gpu

# ---------------------------------------------------------------------------
# Lens distortion correction
# ---------------------------------------------------------------------------

def undistort_frame(frame: np.ndarray, k1: float = -0.32, k2: float = 0.12) -> np.ndarray:
    """Remove barrel/pincushion distortion from a PTZ camera frame.

    Uses the Brown–Conrady model with radial coefficients *k1* and *k2*.
    Negative k1 corrects barrel distortion (edges curve inward);
    positive k1 corrects pincushion distortion.

    The camera matrix is synthesised from the frame dimensions, assuming
    the optical centre is at the image centre and focal length is
    approximately equal to the frame width (typical for PTZ cameras at
    moderate-to-high zoom).

    When CUDA is available, uses GPU-accelerated remap for the undistortion.
    """
    if k1 == 0.0 and k2 == 0.0:
        return frame

    h, w = frame.shape[:2]
    fx = fy = float(w)
    cx, cy = w / 2.0, h / 2.0

    camera_matrix = np.array([
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1],
    ], dtype=np.float64)

    dist_coeffs = np.array([k1, k2, 0, 0, 0], dtype=np.float64)

    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (w, h), alpha=0.5,
    )

    if _USE_CUDA:
        try:
            # Compute remap tables on CPU (small), then remap on GPU (fast)
            map1, map2 = cv2.initUndistortRectifyMap(
                camera_matrix, dist_coeffs, None, new_camera_matrix,
                (w, h), cv2.CV_32FC1,
            )
            gpu_frame = _to_gpu(frame)
            gpu_map1 = _to_gpu(map1)
            gpu_map2 = _to_gpu(map2)
            gpu_result = cv2.cuda.remap(
                gpu_frame, gpu_map1, gpu_map2,
                interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            undistorted = gpu_result.download()
        except cv2.error:
            undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)
    else:
        undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # Crop to the valid ROI to remove black borders from remapping
    rx, ry, rw, rh = roi
    if rw > 0 and rh > 0:
        undistorted = undistorted[ry:ry + rh, rx:rx + rw]

    return undistorted


# ---------------------------------------------------------------------------
# Enhancement
# ---------------------------------------------------------------------------

def enhance_frame(frame: np.ndarray) -> np.ndarray:
    """Boost contrast + brightness so ORB finds keypoints in dark frames.

    Adapts enhancement strength to frame brightness: very dark frames get
    stronger CLAHE and gamma correction to pull out features that would
    otherwise be invisible to ORB / template matching.

    Uses CUDA-accelerated CLAHE when available.
    """
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Adapt CLAHE strength to frame brightness — dark frames need more help
    mean_l = float(l.mean())
    if mean_l < 50:
        clip_limit = 6.0   # very dark — aggressive contrast boost
        gamma = 2.5
    elif mean_l < 90:
        clip_limit = 5.0   # dim — moderate-strong boost
        gamma = 2.2
    else:
        clip_limit = 4.0   # normal / well-lit
        gamma = 2.0

    if _USE_CUDA:
        try:
            clahe_gpu = cv2.cuda.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            gpu_l = _to_gpu(l)
            gpu_l_out = clahe_gpu.apply(gpu_l, cv2.cuda.Stream_Null())
            l = gpu_l_out.download()
        except cv2.error:
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
            l = clahe.apply(l)
    else:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)

    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    lut = np.array(
        [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(enhanced, lut)


def _brighten_output(img: np.ndarray) -> np.ndarray:
    """Apply uniform brightness/contrast boost to the final panorama output.

    Uses a global gamma correction for even brightness across the whole
    stitched image, plus CLAHE with a large tile grid scaled to the panorama
    size so that brightness is consistent across seams.

    Uses CUDA CLAHE when available (the final panorama is large so this helps).
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # Scale CLAHE tile grid to panorama size so tiles are ~256px each,
    # giving smooth, even local contrast without visible tile boundaries.
    h, w = l.shape[:2]
    tile_w = max(1, w // 256)
    tile_h = max(1, h // 256)

    if _USE_CUDA:
        try:
            clahe_gpu = cv2.cuda.createCLAHE(clipLimit=2.0, tileGridSize=(tile_w, tile_h))
            gpu_l = _to_gpu(l)
            gpu_l_out = clahe_gpu.apply(gpu_l, cv2.cuda.Stream_Null())
            l = gpu_l_out.download()
        except cv2.error:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(tile_w, tile_h))
            l = clahe.apply(l)
    else:
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
_PHASE_CORR_MIN = 0.02          # minimum phase-correlation response to accept (low for dark scenes)


def _estimate_affine_cuda(
    gray_a: np.ndarray,
    gray_b: np.ndarray,
) -> Tuple[Optional[list], Optional[np.ndarray], Optional[list], Optional[np.ndarray]]:
    """Run ORB detect+compute on GPU.

    Returns (kp_a, des_a, kp_b, des_b) as CPU objects ready for RANSAC.
    The RANSAC step itself stays on CPU (tiny data, not worth GPU transfer).
    """
    orb_gpu = cv2.cuda.ORB_create(nfeatures=_ORB_FEATURES)
    gpu_a = _to_gpu(gray_a)
    gpu_b = _to_gpu(gray_b)

    # detectAndCompute returns (list[KeyPoint], GpuMat) — keypoints are
    # already on CPU, only descriptors live on GPU.
    kp_a, des_a_gpu = orb_gpu.detectAndCompute(gpu_a, None)
    kp_b, des_b_gpu = orb_gpu.detectAndCompute(gpu_b, None)

    des_a = des_a_gpu.download() if des_a_gpu is not None and not des_a_gpu.empty() else None
    des_b = des_b_gpu.download() if des_b_gpu is not None and not des_b_gpu.empty() else None

    return kp_a, des_a, kp_b, des_b


def _estimate_affine(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Estimate the 2×3 affine transform that maps *frame_b* into *frame_a*'s
    coordinate system.  Returns a 3×3 matrix (with [0,0,1] bottom row) or
    None if matching fails.

    Uses CUDA-accelerated ORB + BFMatcher when available.
    """
    h_a, w_a = frame_a.shape[:2]
    h_b, w_b = frame_b.shape[:2]
    scale = min(1.0, 800.0 / max(h_a, w_a, h_b, w_b))

    gray_a = cv2.cvtColor(frame_a, cv2.COLOR_BGR2GRAY)
    gray_b = cv2.cvtColor(frame_b, cv2.COLOR_BGR2GRAY)
    if scale < 1.0:
        gray_a = cv2.resize(gray_a, None, fx=scale, fy=scale)
        gray_b = cv2.resize(gray_b, None, fx=scale, fy=scale)

    if _USE_CUDA:
        try:
            kp_a, des_a, kp_b, des_b = _estimate_affine_cuda(gray_a, gray_b)
        except cv2.error:
            # Fallback to CPU on any CUDA error
            kp_a, des_a, kp_b, des_b = None, None, None, None

        if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
            # Retry on CPU if GPU gave insufficient results
            orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
            kp_a, des_a = orb.detectAndCompute(gray_a, None)
            kp_b, des_b = orb.detectAndCompute(gray_b, None)
    else:
        orb = cv2.ORB_create(nfeatures=_ORB_FEATURES)
        kp_a, des_a = orb.detectAndCompute(gray_a, None)
        kp_b, des_b = orb.detectAndCompute(gray_b, None)

    if des_a is None or des_b is None or len(kp_a) < 4 or len(kp_b) < 4:
        return None

    # BFMatcher — GPU version for large descriptor sets
    if _USE_CUDA and des_a.shape[0] >= 50 and des_b.shape[0] >= 50:
        try:
            bf_gpu = cv2.cuda.DescriptorMatcher_createBFMatcher(cv2.NORM_HAMMING)
            gpu_des_a = _to_gpu(des_a)
            gpu_des_b = _to_gpu(des_b)
            raw_matches = bf_gpu.knnMatch(gpu_des_a, gpu_des_b, k=2)
        except cv2.error:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING)
            raw_matches = bf.knnMatch(des_a, des_b, k=2)
    else:
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
    candidates are only accepted if they are within a tolerance of the
    expected offset.  This prevents bad feature matches from pulling frames
    out of alignment — the known camera geometry is trusted as ground truth
    and feature matching only *refines* within that constraint.
    """
    h, w = frame_a.shape[:2]
    candidates: list = []

    # Maximum distance (pixels) a feature match can deviate from the
    # expected offset before it's rejected.  70 % of the frame dimension
    # along the stitching axis accommodates a wide range of actual
    # overlaps (the expected offset is calibrated from phase correlation
    # but may still be imperfect).
    if expected_offset is not None:
        if fallback_direction == "horizontal":
            max_deviation = w * 0.70
        else:
            max_deviation = h * 0.70
    else:
        max_deviation = float("inf")

    def _within_tolerance(H_mat):
        """Check if a candidate transform is close enough to expected offset."""
        if expected_offset is None:
            return True
        ex_dx, ex_dy = expected_offset
        err = abs(H_mat[0, 2] - ex_dx) + abs(H_mat[1, 2] - ex_dy)
        return err <= max_deviation

    # 1. ORB on enhanced frames
    H = _estimate_affine(enhanced_a, enhanced_b)
    if H is not None and not _is_valid_affine(H, frame_b.shape):
        H = None
    if H is not None and _direction_ok(H, fallback_direction, frame_b.shape) and _within_tolerance(H):
        candidates.append(("orb_enhanced", H))

    # 2. Template matching on enhanced frames
    H = _template_affine(enhanced_a, enhanced_b)
    if H is not None and _direction_ok(H, fallback_direction, frame_b.shape) and _within_tolerance(H):
        candidates.append(("template_enhanced", H))

    # 3. Phase correlation (very robust for repetitive textures)
    H = _phase_correlation_affine(frame_a, frame_b, direction=fallback_direction)
    if H is not None and _within_tolerance(H):
        candidates.append(("phase_corr", H))

    # 4. ORB on raw frames
    if not candidates:
        H = _estimate_affine(frame_a, frame_b)
        if H is not None and not _is_valid_affine(H, frame_b.shape):
            H = None
        if H is not None and _direction_ok(H, fallback_direction, frame_b.shape) and _within_tolerance(H):
            candidates.append(("orb_raw", H))

    # 5. Template matching on raw frames
    if not candidates:
        H = _template_affine(frame_a, frame_b)
        if H is not None and _direction_ok(H, fallback_direction, frame_b.shape) and _within_tolerance(H):
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

    # All matching failed or all candidates were outside tolerance —
    # use expected offset directly if available (camera positions are
    # reliable), else generic guess.
    H = np.eye(3, dtype=np.float64)
    if expected_offset is not None:
        H[0, 2], H[1, 2] = expected_offset
        logger.info(
            f"  Using camera-position placement "
            f"(dx={expected_offset[0]:.0f}, dy={expected_offset[1]:.0f})"
        )
        # Position-based placement is not a "failure" — it's a reliable
        # fallback from known geometry.  Return False so it's not counted
        # as a fallback warning.
        return H, False
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
            if mean_i < 0.1 or mean_j < 0.1:
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

    # Clamp to reasonable range — wide enough to handle dramatic lighting
    # changes (e.g. house lights up in some areas, dim in others).
    gains = np.clip(gains, 0.2, 4.0)
    # Renormalize so mean is 1
    gains /= gains.mean()

    logger.info(f"Gain compensation: min={gains.min():.3f} max={gains.max():.3f}")
    return gains


def _normalize_exposures(frames: List[np.ndarray]) -> List[np.ndarray]:
    """Normalize per-frame brightness so all frames share a similar mean luminance.

    Converts each frame to LAB, measures the mean L value, and applies a
    per-frame affine mapping on L so that every frame's mean matches the
    global median.  This brings dark and bright frames much closer in
    exposure *before* the global gain compensation and blending steps,
    dramatically reducing visible seams when lighting varies.
    """
    if len(frames) <= 1:
        return frames

    # Measure mean luminance of each frame
    means = []
    for f in frames:
        l_ch = cv2.cvtColor(f, cv2.COLOR_BGR2LAB)[:, :, 0]
        means.append(float(l_ch.mean()))

    target = float(np.median(means))
    if target < 1:
        return frames  # all frames are nearly black, nothing to normalise

    out = []
    for f, m in zip(frames, means):
        if m < 1 or abs(m - target) < 3:
            # Already close enough, or too dark to normalise safely
            out.append(f)
            continue
        ratio = target / m
        # Clamp ratio to avoid extreme corrections on very dark frames
        ratio = max(0.3, min(ratio, 4.0))
        lab = cv2.cvtColor(f, cv2.COLOR_BGR2LAB).astype(np.float32)
        lab[:, :, 0] = np.clip(lab[:, :, 0] * ratio, 0, 255)
        out.append(cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR))

    logger.info(
        f"Exposure normalization: frame L means {min(means):.0f}–{max(means):.0f} "
        f"→ target {target:.0f}"
    )
    return out


def _composite_affine(
    frames: List[np.ndarray],
    transforms: List[np.ndarray],
) -> np.ndarray:
    """
    Warp every frame using its accumulated affine transform and blend onto a
    single canvas.  Uses per-frame exposure normalization and gain compensation
    to match brightness across frames, with power-weighted distance-transform
    blending for seamless seams.

    When CUDA is available, warpAffine runs on the GPU for a significant
    speed-up (the canvas is large and there are many frames to warp).
    """
    # Normalize per-frame brightness before compositing so dark and
    # bright frames are brought closer together in exposure.
    frames = _normalize_exposures(frames)

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

    logger.info(f"Canvas size: {canvas_w} x {canvas_h} (CUDA={'yes' if _USE_CUDA else 'no'})")

    # Compute per-frame gain compensation to match brightness across frames
    gains = _compute_gain_compensation(frames, transforms, canvas_w, canvas_h, T)

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)
    kernel = np.ones((7, 7), np.uint8)

    for i, (frame, H) in enumerate(zip(frames, transforms)):
        H_final = T @ H
        A_final = H_final[:2, :]

        if _USE_CUDA:
            try:
                gpu_frame = _to_gpu(frame)
                gpu_warped = cv2.cuda.warpAffine(
                    gpu_frame, A_final, (canvas_w, canvas_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                warped = gpu_warped.download()

                frame_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
                gpu_mask = _to_gpu(frame_mask)
                gpu_warped_mask = cv2.cuda.warpAffine(
                    gpu_mask, A_final, (canvas_w, canvas_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
                warped_mask = gpu_warped_mask.download()
            except cv2.error:
                # Fall through to CPU path
                warped = cv2.warpAffine(
                    frame, A_final, (canvas_w, canvas_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=(0, 0, 0),
                )
                frame_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
                warped_mask = cv2.warpAffine(
                    frame_mask, A_final, (canvas_w, canvas_h),
                    flags=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT,
                    borderValue=0,
                )
        else:
            warped = cv2.warpAffine(
                frame, A_final, (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=(0, 0, 0),
            )
            frame_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
            warped_mask = cv2.warpAffine(
                frame_mask, A_final, (canvas_w, canvas_h),
                flags=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )

        # Apply gain compensation
        warped = np.clip(warped.astype(np.float32) * gains[i], 0, 255).astype(np.uint8)

        valid = (warped_mask > 128).astype(np.uint8)

        # Erosion removes edge artefacts from lens distortion,
        # vignetting, and warp-interpolation fringes that cause ghosting.
        # 3 iterations with a 7×7 kernel removes ~21px from each edge —
        # enough to hide interpolation fringes without creating black gaps.
        valid = cv2.erode(valid, kernel, iterations=3)

        dist = cv2.distanceTransform(valid, cv2.DIST_L2, 5)

        # Power-weight the distance so centre pixels dominate strongly;
        # exponent 2.0 gives a moderate transition zone that hides
        # residual misalignment while blending exposure differences
        # smoothly across overlap regions.
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

    The overlap factor is calibrated empirically by running phase
    correlation on the first pair of frames, so it adapts to the actual
    camera zoom / FOV rather than relying on a hardcoded assumption.
    Cross-axis components (dy for horizontal, dx for vertical) are zeroed
    to prevent accumulated drift from numerical noise in positions.
    """
    n = len(frames)
    if positions is None or len(positions) != n:
        return [None] * max(0, n - 1)

    h, w = frames[0].shape[:2]

    # --- Calibrate overlap from the first pair via phase correlation ---
    overlap = 0.35  # fallback default
    if n >= 2:
        pc = _phase_correlation_affine(frames[0], frames[1], direction=direction)
        if pc is not None:
            if direction == "horizontal":
                measured = abs(pc[0, 2])
                if measured > 0.03 * w:
                    overlap = 1.0 - measured / w
                    overlap = max(0.05, min(0.95, overlap))
                    logger.info(
                        f"Calibrated horizontal overlap from phase correlation: "
                        f"{overlap:.0%} (offset {measured:.0f}px / {w}px)"
                    )
            else:
                measured = abs(pc[1, 2])
                if measured > 0.03 * h:
                    overlap = 1.0 - measured / h
                    overlap = max(0.05, min(0.95, overlap))
                    logger.info(
                        f"Calibrated vertical overlap from phase correlation: "
                        f"{overlap:.0%} (offset {measured:.0f}px / {h}px)"
                    )

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
            # Zero out cross-axis component to prevent vertical drift
            # within a horizontal row (all frames share the same tilt).
            offsets.append((dp * pixels_per_pan, 0.0))
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
            dt = positions[i + 1][1] - positions[i][1]
            # Rows are ordered top→bottom in the grid, so each subsequent
            # row strip should be BELOW the previous one (positive dy).
            # Some cameras have inverted tilt (higher value = looking up),
            # so use abs(dt) to guarantee downward progression.
            # Zero out cross-axis (pan) component to prevent horizontal
            # staircase drift between rows.
            offsets.append((0.0, abs(dt) * pixels_per_tilt))
        return offsets


def _stitch_strip(
    frames: List[np.ndarray],
    direction: str = "horizontal",
    positions: Optional[List[Tuple[float, float]]] = None,
    constrain_transform: str = "similarity",
) -> Optional[np.ndarray]:
    """
    Stitch a list of frames that overlap in the given direction.
    Returns a single composited image, or None if empty.

    When *positions* (list of (pan, tilt) per frame) is provided, expected
    pixel offsets are computed from the known camera geometry and used to
    validate matches and as a better fallback than the generic 85 % guess.

    *constrain_transform* controls how pairwise affine transforms are
    simplified before chaining:
      - "none"        — keep the full affine (translation + rotation + scale
                        + shear).  Risk of accumulated drift on long strips.
      - "similarity"  — keep translation + rotation but force uniform scale=1
                        and zero shear.  Lets frames rotate slightly to match
                        features while preventing scale/shear drift.
      - "translation" — keep only (dx, dy).  Most rigid, may misalign if the
                        camera isn't perfectly straight.
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

        if constrain_transform == "translation":
            # Keep only translation (dx, dy), discard rotation/scale/shear.
            H_trans = np.eye(3, dtype=np.float64)
            H_trans[0, 2] = H[0, 2]
            H_trans[1, 2] = H[1, 2]
            H = H_trans
        elif constrain_transform == "similarity":
            # Extract translation + rotation, force scale=1 and zero shear.
            # This lets frames rotate slightly to match features without
            # accumulated scale/shear drift across many frames.
            a, b = H[0, 0], H[0, 1]
            theta = np.arctan2(b, a)
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            H_sim = np.array([
                [cos_t, -sin_t, H[0, 2]],
                [sin_t,  cos_t, H[1, 2]],
                [0,      0,     1       ],
            ], dtype=np.float64)
            H = H_sim

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
    else:
        # Synthesize approximate positions from grid layout so the stitcher
        # has expected-offset guidance even for preset scans without learned
        # positions.  Use evenly-spaced synthetic pan/tilt values — the
        # exact scale doesn't matter because _estimate_expected_offsets
        # normalises by the total range.
        logger.info("Synthesizing grid positions for alignment guidance (no learned positions)")
        synthetic = []
        for col in range(cols):
            for row_idx in range(rows):
                if col % 2 == 0:
                    r = row_idx
                else:
                    r = rows - 1 - row_idx
                # pan increases with column, tilt increases with row
                synthetic.append((float(col * 100), float(r * 100)))
        if len(synthetic) >= len(frames):
            synthetic = synthetic[:len(frames)]
            pos_grid = _arrange_boustrophedon_positions(synthetic, rows, cols)

    # Log grid occupancy
    for r in range(rows):
        occupied = sum(1 for c in range(cols) if grid[r][c] is not None)
        logger.info(f"  Row {r}: {occupied}/{cols} frames")

    # Stitch each row horizontally (parallel across rows)
    logger.info("Stitching rows horizontally...")

    # Build row tasks
    row_tasks = []  # (row_index, row_frames, row_positions)
    for r in range(rows):
        row_frames = [grid[r][c] for c in range(cols) if grid[r][c] is not None]
        row_pos = None
        if pos_grid is not None:
            row_pos = [pos_grid[r][c] for c in range(cols) if grid[r][c] is not None]
        if not row_frames:
            logger.warning(f"  Row {r}: no frames, skipping")
            continue
        logger.info(f"  Row {r}: stitching {len(row_frames)} frames horizontally")
        row_tasks.append((r, row_frames, row_pos))

    def _stitch_row(task):
        r, row_frames, row_pos = task
        strip = _stitch_strip(row_frames, direction="horizontal", positions=row_pos, constrain_transform="similarity")
        mean_pos = None
        if strip is not None and row_pos:
            mean_pos = (np.mean([p[0] for p in row_pos]), np.mean([p[1] for p in row_pos]))
        return r, strip, mean_pos

    n_row_workers = min(len(row_tasks), os.cpu_count() or 4)
    row_strips: List[np.ndarray] = []
    row_strip_positions: List[Optional[Tuple[float, float]]] = []

    with ThreadPoolExecutor(max_workers=n_row_workers) as pool:
        results = list(pool.map(_stitch_row, row_tasks))

    # Collect results in row order
    for r, strip, mean_pos in sorted(results, key=lambda x: x[0]):
        if strip is not None:
            row_strips.append(strip)
            if mean_pos:
                row_strip_positions.append(mean_pos)
    gc.collect()

    if not row_strips:
        return None, "no_frames"
    if len(row_strips) == 1:
        return row_strips[0], "single_row"

    # Stitch row strips vertically
    logger.info(f"Stitching {len(row_strips)} row strips vertically...")
    vert_pos = row_strip_positions if len(row_strip_positions) == len(row_strips) else None
    panorama = _stitch_strip(row_strips, direction="vertical", positions=vert_pos, constrain_transform="similarity")

    if panorama is None:
        return None, "no_frames"

    panorama = _crop_black(panorama)
    logger.info(f"Grid stitch complete — output shape: {panorama.shape}")
    return panorama, "ok"


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _stitch_opencv(frames: List[np.ndarray]) -> Tuple[Optional[np.ndarray], str]:
    """
    OpenCV Stitcher-based panorama (RANSAC feature matching).
    Used for preset scans where frame ordering / grid geometry is unknown.
    Falls back to horizontal concat if stitching fails.
    """
    unique = _deduplicate(frames)
    logger.info(f"OpenCV stitch: {len(unique)} unique frames (from {len(frames)} captured)")

    # Try enhanced frames first (better keypoints in low light)
    enhanced = [enhance_frame(f) for f in unique]
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    # When CUDA OpenCV is available, the high-level Stitcher can use
    # GPU-accelerated feature detection and matching internally via
    # ORB_CUDA / BFMatcher_CUDA.  We just need to set the features finder.
    if _USE_CUDA:
        try:
            stitcher.setFeaturesFinder(cv2.cuda.ORB_create(_ORB_FEATURES))
        except (cv2.error, AttributeError):
            pass  # Fall back to default CPU features finder
    status, pano = stitcher.stitch(enhanced)

    if status == cv2.Stitcher_OK:
        pano = _crop_black(pano)
        logger.info(f"OpenCV stitch OK (enhanced) — output shape: {pano.shape}")
        return pano, "ok"

    logger.warning(f"Enhanced OpenCV stitch failed (status={status}), trying original…")

    status2, pano2 = stitcher.stitch(unique)
    if status2 == cv2.Stitcher_OK:
        pano2 = _crop_black(pano2)
        logger.info(f"OpenCV stitch OK (original) — output shape: {pano2.shape}")
        return pano2, "ok_original"

    logger.warning("Both OpenCV stitch attempts failed, using fallback concat")

    # Fallback: resize all to same height and concat horizontally
    target_h = min(f.shape[0] for f in unique)
    resized = []
    for f in unique:
        h, w = f.shape[:2]
        new_w = int(w * target_h / h)
        resized.append(cv2.resize(f, (new_w, target_h)))
    return np.hstack(resized), "fallback_concat"


def stitch_frames(
    frames: List[np.ndarray],
    grid_shape: Optional[Tuple[int, int]] = None,
    positions: Optional[List[Tuple[float, float]]] = None,
    scan_mode: Optional[str] = None,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Stitch a sequence of overlapping frames into a single panorama.

    *scan_mode* controls which algorithm is used:
      - "preset"    → OpenCV Stitcher (RANSAC feature matching + fallback concat)
      - "calibrated"→ Grid-aware affine stitching with position guidance
      - None        → auto-select based on whether grid_shape is provided

    Barrel / pincushion lens distortion is automatically corrected on
    each frame before stitching (via ``undistort_frame``).

    Returns (panorama_image, status_string).
    """
    if not frames:
        return None, "no_frames"

    if len(frames) == 1:
        return frames[0], "single_frame"

    # --- Automatic lens distortion correction ---
    logger.info(
        f"Applying lens undistortion to {len(frames)} frames "
        f"(CUDA={'yes' if _USE_CUDA else 'no'})"
    )
    if _USE_CUDA:
        # CUDA remap is not thread-safe — run sequentially on GPU
        frames = [undistort_frame(f) for f in frames]
    else:
        n_workers = min(len(frames), os.cpu_count() or 4)
        with ThreadPoolExecutor(max_workers=n_workers) as pool:
            frames = list(pool.map(undistort_frame, frames))

    # Route to the correct algorithm based on scan mode
    if scan_mode == "preset" and grid_shape is not None:
        rows, cols = grid_shape
        if rows > 0 and cols > 0 and len(frames) >= rows * cols * 0.5:
            logger.info(f"Preset scan with grid {rows}×{cols}: using grid-aware stitcher")
            panorama, status = _stitch_grid(frames, rows, cols, positions=positions)
        else:
            logger.warning(
                f"Preset scan but grid shape {grid_shape} doesn't match "
                f"frame count {len(frames)}, falling back to OpenCV stitcher"
            )
            panorama, status = _stitch_opencv(frames)
    elif scan_mode == "preset":
        panorama, status = _stitch_opencv(frames)
    elif scan_mode == "calibrated" and grid_shape is not None:
        rows, cols = grid_shape
        if rows > 0 and cols > 0 and len(frames) >= rows * cols * 0.5:
            panorama, status = _stitch_grid(frames, rows, cols, positions=positions)
        else:
            logger.warning(
                f"Calibrated mode but grid shape {grid_shape} doesn't match "
                f"frame count {len(frames)}, falling back to OpenCV stitcher"
            )
            panorama, status = _stitch_opencv(frames)
    elif scan_mode == "calibrated":
        # Calibrated mode without grid info — fall back to sequential
        logger.warning("Calibrated mode but no grid_shape provided, using sequential stitch")
        panorama, status = _stitch_sequential(frames)
    elif grid_shape is not None:
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
