"""
YOLO26x person detection with tiled inference for Lakeshore Church attendance scanner.
Tiling splits the panorama into overlapping sections so people in back rows
(who appear small in the full image) are detected at a larger effective size.

YOLO26x improvements over YOLO11x:
  - NMS-free end-to-end inference per tile (no duplicate suppression within a tile)
  - STAL: Small-Target-Aware Label Assignment (better recall for distant/small people)
  - ProgLoss: improved small-object recall without increasing model size
  - ~43% faster CPU inference per tile

Performance notes (ENCS5412 / Xeon D / CPU-only):
  - All tiles are batched into a single YOLO call to minimise Python/PyTorch overhead
  - For further speedup, export to OpenVINO with export_openvino.sh (2-4x on Intel CPUs)
    then set YOLO_MODEL=/models/yolo26x_openvino_model in docker-compose.yml
  - Cross-tile NMS deduplication is still applied after merging all tile results
"""
import logging
import math
import os
from typing import Dict, List, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

_model = None

# ── Tiling config ─────────────────────────────────────────────────────────────
TILE_COLS    = 10    # number of columns to split panorama into
TILE_ROWS    = 8     # number of rows
TILE_OVERLAP = 0.50  # 50% overlap between tiles to avoid missing edge detections
NMS_IOU      = 0.30  # IOU threshold for cross-tile deduplication

# Max tiles per YOLO batch call. Reduce if you hit memory pressure (min 1).
TILE_BATCH_SIZE = int(os.getenv("TILE_BATCH_SIZE", "24"))


def _get_model():
    global _model
    if _model is None:
        from ultralytics import YOLO
        model_name = os.getenv("YOLO_MODEL", "yolo26x.pt")
        _model = YOLO(model_name)
        logger.info(f"YOLO model loaded: {model_name}")
    return _model


# ── Image enhancement ─────────────────────────────────────────────────────────
def enhance_image(image: np.ndarray) -> np.ndarray:
    """Brighten and boost contrast to improve detection in dark church images."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    enhanced = cv2.cvtColor(cv2.merge((l, a, b)), cv2.COLOR_LAB2BGR)
    gamma = 2.2
    lut = np.array([((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)], dtype=np.uint8)
    return cv2.LUT(enhanced, lut)


# ── Cross-tile NMS ─────────────────────────────────────────────────────────────
def _nms_detections(detections: List[Dict], iou_threshold: float = NMS_IOU) -> List[Dict]:
    """
    Non-maximum suppression across all tile detections.
    Removes duplicate detections of the same person from overlapping tile borders.
    (YOLO26 is NMS-free within each tile; this handles cross-tile dedup only.)
    """
    if not detections:
        return []

    boxes  = np.array([[d["x1"], d["y1"], d["x2"], d["y2"]] for d in detections], dtype=np.float32)
    scores = np.array([d["confidence"] for d in detections], dtype=np.float32)

    x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = (x2 - x1) * (y2 - y1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        ix1 = np.maximum(x1[i], x1[order[1:]])
        iy1 = np.maximum(y1[i], y1[order[1:]])
        ix2 = np.minimum(x2[i], x2[order[1:]])
        iy2 = np.minimum(y2[i], y2[order[1:]])
        inter = np.maximum(0, ix2 - ix1) * np.maximum(0, iy2 - iy1)
        iou   = inter / (areas[i] + areas[order[1:]] - inter + 1e-6)
        order = order[1:][iou <= iou_threshold]

    return [detections[i] for i in keep]


# ── Tiled detection ────────────────────────────────────────────────────────────
def detect_people(image: np.ndarray, confidence: float = 0.10) -> List[Dict]:
    """
    Run YOLO26x on image using batched tiled inference.
    Splits the panorama into a TILE_COLS x TILE_ROWS grid with TILE_OVERLAP overlap.
    All tiles are submitted to YOLO in a single batched call (TILE_BATCH_SIZE at a time)
    to minimise overhead on CPU, then coordinates are mapped back to full image space
    and cross-tile NMS removes border duplicates.
    Returns list of unique person detections in full image coordinates.
    """
    model    = _get_model()
    enhanced = enhance_image(image)
    h, w     = enhanced.shape[:2]

    tile_w = int(w / (TILE_COLS - TILE_OVERLAP * (TILE_COLS - 1)))
    tile_h = int(h / (TILE_ROWS - TILE_OVERLAP * (TILE_ROWS - 1)))
    step_x = int(tile_w * (1 - TILE_OVERLAP))
    step_y = int(tile_h * (1 - TILE_OVERLAP))

    # ── Build tile list ────────────────────────────────────────────────────────
    tiles: List[np.ndarray] = []
    offsets: List[Tuple[int, int]] = []  # (x_start, y_start) for each tile

    for row in range(TILE_ROWS):
        for col in range(TILE_COLS):
            x_start = col * step_x
            y_start = row * step_y
            x_end   = min(x_start + tile_w, w)
            y_end   = min(y_start + tile_h, h)
            tile    = enhanced[y_start:y_end, x_start:x_end]
            if tile.size == 0:
                continue
            tiles.append(tile)
            offsets.append((x_start, y_start))

    logger.info(f"Running batched inference on {len(tiles)} tiles "
                f"(batch size {TILE_BATCH_SIZE})")

    # ── Batched YOLO inference ─────────────────────────────────────────────────
    all_detections: List[Dict] = []

    for batch_start in range(0, len(tiles), TILE_BATCH_SIZE):
        batch_tiles   = tiles  [batch_start : batch_start + TILE_BATCH_SIZE]
        batch_offsets = offsets[batch_start : batch_start + TILE_BATCH_SIZE]

        results = model(
            batch_tiles,
            classes=[0],
            conf=confidence,
            imgsz=1280,
            verbose=False,
        )

        for result, (x_start, y_start) in zip(results, batch_offsets):
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                conf = float(box.conf[0])
                all_detections.append({
                    "x1":         x1 + x_start,
                    "y1":         y1 + y_start,
                    "x2":         x2 + x_start,
                    "y2":         y2 + y_start,
                    "cx":         (x1 + x2) // 2 + x_start,
                    "cy":         (y1 + y2) // 2 + y_start,
                    "confidence": round(conf, 3),
                })

    logger.info(f"Batched detection: {len(all_detections)} raw detections "
                f"across {len(tiles)} tiles")

    # ── Cross-tile deduplication ───────────────────────────────────────────────
    detections = _nms_detections(all_detections)
    logger.info(f"After cross-tile NMS: {len(detections)} unique people detected")
    return detections


def draw_detections(image: np.ndarray, detections: List[Dict]) -> np.ndarray:
    """Draw green bounding boxes + confidence scores."""
    out = image.copy()
    for d in detections:
        cv2.rectangle(out, (d["x1"], d["y1"]), (d["x2"], d["y2"]), (0, 220, 80), 2)
        label = f"{d['confidence']:.2f}"
        cv2.putText(out, label, (d["x1"], max(0, d["y1"] - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 80), 1, cv2.LINE_AA)
    return out


# ── Seat mapping via homography ───────────────────────────────────────────────
def map_detections_to_seats(
    detections: List[Dict],
    calibration: Dict,
    svg_viewbox: Tuple[float, float, float, float],
    seat_radius_fraction: float = 0.012,
) -> Dict[str, str]:
    """
    Map photo detections → SVG seat states using homography.
    calibration: {seat_id: {photo_x, photo_y, svg_x, svg_y}}
    svg_viewbox: (min_x, min_y, width, height)
    Returns: {seat_id: "occupied" | "empty"}
    """
    if len(calibration) < 4:
        logger.warning("Need ≥4 calibration points for homography — skipping seat mapping")
        return {}

    src = np.float32([[c["photo_x"], c["photo_y"]] for c in calibration.values()])
    dst = np.float32([[c["svg_x"],   c["svg_y"]  ] for c in calibration.values()])

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        logger.error("Homography failed")
        return {}

    det_pts = (np.float32([[d["cx"], d["cy"]] for d in detections]).reshape(-1, 1, 2)
               if detections else np.zeros((0, 1, 2), dtype=np.float32))
    svg_pts: List[Tuple[float, float]] = []
    if len(det_pts) > 0:
        proj    = cv2.perspectiveTransform(det_pts, H)
        svg_pts = [(float(p[0][0]), float(p[0][1])) for p in proj]

    radius = min(svg_viewbox[2], svg_viewbox[3]) * seat_radius_fraction

    seat_states: Dict[str, str] = {}
    for seat_id, cal in calibration.items():
        sx, sy   = cal["svg_x"], cal["svg_y"]
        occupied = any(math.hypot(sx - px, sy - py) <= radius for px, py in svg_pts)
        seat_states[seat_id] = "occupied" if occupied else "empty"

    n_occ = sum(1 for v in seat_states.values() if v == "occupied")
    logger.info(f"Seat map: {n_occ}/{len(seat_states)} occupied")
    return seat_states
