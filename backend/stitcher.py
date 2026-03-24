"""
Panorama stitcher for sequential PTZ camera frames.

Uses full perspective homographies (not just translations) to properly
handle the curvature and perspective distortion from a ceiling-mounted
PTZ camera.  Each consecutive frame pair is matched via ORB features
with RANSAC homography estimation, then all frames are warped into a
common coordinate system and blended with distance-weighted seams.

The middle frame is used as the reference to minimise accumulated
distortion at the panorama edges.
"""

import base64
import gc
import logging
from typing import List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Enhancement (applied to copies for feature detection only)
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


# ---------------------------------------------------------------------------
# Pairwise homography estimation
# ---------------------------------------------------------------------------

_ORB_FEATURES  = 3000
_MATCH_RATIO   = 0.75
_RANSAC_REPROJ = 5.0
_MIN_INLIERS   = 10


def _estimate_homography(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> Optional[np.ndarray]:
    """
    Estimate the 3×3 homography that maps *frame_b* into *frame_a*'s
    coordinate system.  Returns None if matching fails.
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

    H, inliers = cv2.findHomography(
        pts_b, pts_a, cv2.RANSAC, _RANSAC_REPROJ,
    )

    if H is None:
        return None

    n_inliers = int(inliers.sum()) if inliers is not None else 0
    if n_inliers < _MIN_INLIERS:
        return None

    # Scale homography back to original resolution
    S = np.array([[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1]])
    S_inv = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]])
    H_full = S @ H @ S_inv

    logger.debug(
        f"Homography: {n_inliers}/{len(good)} inliers, "
        f"dx≈{H_full[0, 2]:.1f}, dy≈{H_full[1, 2]:.1f}"
    )
    return H_full


def _template_homography(
    frame_a: np.ndarray,
    frame_b: np.ndarray,
) -> Optional[np.ndarray]:
    """Fallback: pure-translation homography via template matching."""
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
        f"Template homography: dx={dx:.1f} dy={dy:.1f} conf={max_val:.3f}"
    )
    return H


# ---------------------------------------------------------------------------
# Homography validation
# ---------------------------------------------------------------------------

def _is_valid_homography(H: np.ndarray, frame_shape: Tuple[int, ...]) -> bool:
    """Reject homographies that produce extreme distortion."""
    h, w = frame_shape[:2]
    corners = np.float32(
        [[0, 0], [w, 0], [w, h], [0, h]]
    ).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H).reshape(-1, 2)

    # Signed area via shoelace — must be positive (not flipped) and sane
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

    # No edge should stretch beyond 2× or shrink below 0.3×
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
# Canvas compositing with perspective warping and blending
# ---------------------------------------------------------------------------

def _composite_homography(
    frames: List[np.ndarray],
    homographies: List[np.ndarray],
) -> np.ndarray:
    """
    Warp every frame using its accumulated homography and blend onto a
    single canvas.  Uses distance-transform weights for seamless seams.
    """
    # Determine canvas bounds by transforming all frame corners
    all_corners = []
    for frame, H in zip(frames, homographies):
        h, w = frame.shape[:2]
        corners = np.float32(
            [[0, 0], [w, 0], [w, h], [0, h]]
        ).reshape(-1, 1, 2)
        warped_corners = cv2.perspectiveTransform(corners, H)
        all_corners.append(warped_corners.reshape(-1, 2))

    all_pts = np.vstack(all_corners)
    min_x, min_y = all_pts.min(axis=0)
    max_x, max_y = all_pts.max(axis=0)

    # Translation so everything has positive coordinates
    T = np.array(
        [[1, 0, -min_x], [0, 1, -min_y], [0, 0, 1]], dtype=np.float64,
    )

    canvas_w = int(np.ceil(max_x - min_x))
    canvas_h = int(np.ceil(max_y - min_y))

    # Cap canvas to prevent memory blow-up.
    # With float32 buffers, peak memory ≈ 32 bytes/pixel × frame count overhead.
    # 16000 max dim keeps worst-case (16000×16000) under ~10 GB.
    max_dim = 16000
    if max(canvas_w, canvas_h) > max_dim:
        s = max_dim / max(canvas_w, canvas_h)
        S = np.array([[s, 0, 0], [0, s, 0], [0, 0, 1]], dtype=np.float64)
        T = S @ T
        canvas_w = int(canvas_w * s)
        canvas_h = int(canvas_h * s)

    logger.info(f"Canvas size: {canvas_w} × {canvas_h}")

    canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.float32)
    weight = np.zeros((canvas_h, canvas_w), dtype=np.float32)

    for i, (frame, H) in enumerate(zip(frames, homographies)):
        H_final = T @ H

        # Warp the frame onto the canvas
        warped = cv2.warpPerspective(
            frame, H_final, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        # Warp an all-white mask to know which pixels are valid
        # (avoids confusing dark image content with black border)
        frame_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
        warped_mask = cv2.warpPerspective(
            frame_mask, H_final, (canvas_w, canvas_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
        valid = (warped_mask > 128).astype(np.uint8)

        # Erode mask slightly to clip interpolation artefacts at edges
        kernel = np.ones((3, 3), np.uint8)
        valid = cv2.erode(valid, kernel, iterations=2)

        # Distance transform gives smooth per-pixel blending weights
        # (high in the centre of the valid area, tapering to zero at edges)
        dist = cv2.distanceTransform(valid, cv2.DIST_L2, 5)

        canvas += warped.astype(np.float32) * dist[:, :, None]
        weight += dist

        if (i + 1) % 10 == 0 or i == len(frames) - 1:
            logger.info(f"  Warped {i + 1}/{len(frames)} frames")

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

    Uses full perspective homographies to handle curvature and distortion
    from a ceiling-mounted PTZ camera.  The middle frame is used as the
    reference to minimise accumulated warp distortion.

    grid_shape is accepted for API compatibility but ignored.
    Returns (panorama_image, status_string).
    """
    if not frames:
        return None, "no_frames"

    if len(frames) == 1:
        return frames[0], "single_frame"

    # 1 — deduplicate
    unique = _deduplicate(frames)
    n = len(unique)
    logger.info(f"Stitching {n} unique frames (from {len(frames)} captured)")

    if n == 1:
        return unique[0], "single_frame"

    # 2 — enhance copies for feature detection
    logger.info("Enhancing frames for feature detection…")
    enhanced = [enhance_frame(f) for f in unique]

    # 3 — compute pairwise homographies between consecutive frames
    logger.info("Computing pairwise homographies…")
    pairwise: List[np.ndarray] = []
    failed_pairs = 0

    for i in range(n - 1):
        H = _estimate_homography(enhanced[i], enhanced[i + 1])

        if H is not None and not _is_valid_homography(H, unique[i + 1].shape):
            logger.warning(
                f"  Pair {i}→{i+1}: homography rejected (extreme distortion)"
            )
            H = None

        if H is None:
            H = _template_homography(enhanced[i], enhanced[i + 1])

        if H is None:
            H = _estimate_homography(unique[i], unique[i + 1])
            if H is not None and not _is_valid_homography(H, unique[i + 1].shape):
                H = None

        if H is None:
            H = _template_homography(unique[i], unique[i + 1])

        if H is None:
            # Pure translation guess as last resort
            prev_w = unique[i].shape[1]
            overlap_guess = int(prev_w * 0.15)
            H = np.eye(3, dtype=np.float64)
            H[0, 2] = prev_w - overlap_guess
            failed_pairs += 1
            logger.warning(
                f"  Pair {i}→{i+1}: all matching failed, guessing placement"
            )

        pairwise.append(H)

        if (i + 1) % 10 == 0 or i == n - 2:
            logger.info(f"  Homography {i+1}/{n-1} done")

        gc.collect()

    del enhanced
    gc.collect()

    if failed_pairs > 0:
        logger.warning(f"{failed_pairs}/{n-1} pairs used fallback placement")

    # 4 — chain pairwise homographies into absolute transforms
    #     pairwise[i] maps frame (i+1) → frame i's coordinate space
    #     So H_abs[k] = pairwise[0] @ pairwise[1] @ … @ pairwise[k-1]
    #     maps frame k into frame 0's space.
    H_abs: List[np.ndarray] = [np.eye(3, dtype=np.float64)]
    for i in range(n - 1):
        H_abs.append(H_abs[-1] @ pairwise[i])

    # Re-reference to the middle frame to minimise distortion
    ref = n // 2
    H_ref_inv = np.linalg.inv(H_abs[ref])
    H_abs = [H_ref_inv @ H for H in H_abs]

    # 5 — composite originals with perspective warping
    logger.info("Compositing panorama with perspective warping…")
    panorama = _composite_homography(unique, H_abs)

    # 6 — crop black borders
    panorama = _crop_black(panorama)

    logger.info(f"Stitch complete — output shape: {panorama.shape}")
    status = "ok" if failed_pairs == 0 else f"ok_with_{failed_pairs}_fallbacks"
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
