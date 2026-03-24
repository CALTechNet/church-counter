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
    """Apply mild brightness/contrast boost to the final panorama output."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))
    l = clahe.apply(l)
    # Gentle gamma correction (gamma=0.8 → slight brighten)
    lut = np.array(
        [((i / 255.0) ** 0.8) * 255 for i in range(256)],
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

def _match_pair(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
    enhanced_a: np.ndarray,
    enhanced_b: np.ndarray,
    fallback_direction: str = "horizontal",
) -> Tuple[Optional[np.ndarray], bool]:
    """
    Try to match a pair of frames.  Returns (H, used_fallback).
    fallback_direction controls the last-resort translation guess:
      "horizontal" → shift right by 85% of frame width
      "vertical"   → shift down by 85% of frame height
    """
    H = _estimate_affine(enhanced_a, enhanced_b)
    if H is not None and not _is_valid_affine(H, frame_b.shape):
        H = None
    if H is None:
        H = _template_affine(enhanced_a, enhanced_b)
    if H is None:
        H = _estimate_affine(frame_a, frame_b)
        if H is not None and not _is_valid_affine(H, frame_b.shape):
            H = None
    if H is None:
        H = _template_affine(frame_a, frame_b)

    if H is not None:
        return H, False

    # Last resort: translation guess
    h, w = frame_a.shape[:2]
    H = np.eye(3, dtype=np.float64)
    if fallback_direction == "vertical":
        overlap_guess = int(h * 0.15)
        H[1, 2] = h - overlap_guess
    else:
        overlap_guess = int(w * 0.15)
        H[0, 2] = w - overlap_guess
    return H, True


# ---------------------------------------------------------------------------
# Canvas compositing with affine warping and blending
# ---------------------------------------------------------------------------

def _composite_affine(
    frames: List[np.ndarray],
    transforms: List[np.ndarray],
) -> np.ndarray:
    """
    Warp every frame using its accumulated affine transform and blend onto a
    single canvas.  Uses distance-transform weights for seamless seams.
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

        frame_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        warped_mask = cv2.warpAffine(
            frame_mask, A_final, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = (warped_mask > 128).astype(np.uint8)

        kernel = np.ones((3, 3), np.uint8)
        valid = cv2.erode(valid, kernel, iterations=2)

        dist = cv2.distanceTransform(valid, cv2.DIST_L2, 5)

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

def _stitch_strip(
    frames: List[np.ndarray],
    direction: str = "horizontal",
) -> Optional[np.ndarray]:
    """
    Stitch a list of frames that overlap in the given direction.
    Returns a single composited image, or None if empty.
    """
    if not frames:
        return None
    if len(frames) == 1:
        return frames[0]

    n = len(frames)
    enhanced = [enhance_frame(f) for f in frames]

    pairwise: List[np.ndarray] = []
    for i in range(n - 1):
        H, failed = _match_pair(
            frames[i], frames[i + 1],
            enhanced[i], enhanced[i + 1],
            fallback_direction=direction,
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


def _stitch_grid(
    frames: List[np.ndarray],
    rows: int,
    cols: int,
) -> Tuple[Optional[np.ndarray], str]:
    """
    Grid-aware stitching: arrange frames into a 2D grid, stitch each row
    horizontally, then stitch the resulting row-strips vertically.
    """
    logger.info(f"Grid-aware stitch: {rows} rows x {cols} cols, {len(frames)} frames")

    # Deduplicate before arranging — but we need to be careful not to
    # remove frames that look similar but are at different positions.
    # With grid-aware stitching, we skip deduplication since we know
    # exactly where each frame belongs.

    # Arrange into grid
    grid = _arrange_boustrophedon(frames, rows, cols)

    # Log grid occupancy
    for r in range(rows):
        occupied = sum(1 for c in range(cols) if grid[r][c] is not None)
        logger.info(f"  Row {r}: {occupied}/{cols} frames")

    # Stitch each row horizontally
    logger.info("Stitching rows horizontally...")
    row_strips: List[np.ndarray] = []
    for r in range(rows):
        row_frames = [grid[r][c] for c in range(cols) if grid[r][c] is not None]
        if not row_frames:
            logger.warning(f"  Row {r}: no frames, skipping")
            continue
        logger.info(f"  Row {r}: stitching {len(row_frames)} frames horizontally")
        strip = _stitch_strip(row_frames, direction="horizontal")
        if strip is not None:
            row_strips.append(strip)
        gc.collect()

    if not row_strips:
        return None, "no_frames"
    if len(row_strips) == 1:
        return row_strips[0], "single_row"

    # Stitch row strips vertically
    logger.info(f"Stitching {len(row_strips)} row strips vertically...")
    panorama = _stitch_strip(row_strips, direction="vertical")

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
) -> Tuple[Optional[np.ndarray], str]:
    """
    Stitch a sequence of overlapping frames into a single panorama.

    When grid_shape=(rows, cols) is provided, uses grid-aware stitching
    that properly handles the boustrophedon scan pattern by stitching
    rows first, then stacking vertically.

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
            panorama, status = _stitch_grid(frames, rows, cols)
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
