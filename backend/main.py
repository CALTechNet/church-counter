"""
Lakeshore Church — Attendance Counter API
FastAPI + APScheduler, single container.
"""
import asyncio
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import camera as cam
import database as db
import detector as det
import stitcher as stitch
from scheduler import build_scheduler, register_scan_callback

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
)
logger = logging.getLogger(__name__)

SVG_PATH = Path("/config/seats.svg")
FRONTEND_DIR = Path("/frontend/build")

app = FastAPI(title="Lakeshore Church Attendance Counter", version="1.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

# ── Shared state ──────────────────────────────────────────────────────────────
state = {
    "running": False,
    "progress": 0,
    "message": "Idle",
    "latest_count": None,
    "latest_timestamp": None,
    "latest_service": None,
    "latest_image_b64": None,
    "latest_raw_image_b64": None,
    "latest_seat_states": {},
    "latest_detections": [],
}
_ws_clients: list[WebSocket] = []


async def _broadcast(data: dict):
    dead = []
    for ws in _ws_clients:
        try:
            await ws.send_json(data)
        except Exception:
            dead.append(ws)
    for ws in dead:
        if ws in _ws_clients:
            _ws_clients.remove(ws)


# ── Scan pipeline ─────────────────────────────────────────────────────────────
async def _progress(msg: str, pct: int):
    state["progress"] = pct
    state["message"] = msg
    await _broadcast({"type": "progress", "message": msg, "progress": pct})


async def run_scan(service_type: str = "Manual"):
    if state["running"]:
        logger.warning("Scan already in progress — ignoring request")
        return

    state["running"] = True
    await _broadcast({"type": "scan_started", "service_type": service_type})

    try:
        import cv2
        scan_dir = Path("/data/scans")
        scan_dir.mkdir(parents=True, exist_ok=True)

        # 1. Capture
        await _progress("Scanning…", 0)
        frames = await cam.auto_scan(progress_callback=_progress)
        if not frames:
            raise RuntimeError("Camera returned 0 frames")

        # 2. Stitch
        await _progress("Stitching panorama…", 93)
        panorama, stitch_status = stitch.stitch_frames(frames)
        if panorama is None:
            raise RuntimeError("Panorama stitching failed")
        logger.info(f"Stitch: {stitch_status} → shape {panorama.shape}")

        # 3. Detect
        await _progress("Running AI detection…", 96)
        raw_image_b64 = stitch.to_base64(panorama)
        detections = det.detect_people(panorama)
        annotated  = det.draw_detections(panorama, detections)
        image_b64  = stitch.to_base64(annotated)
        total      = len(detections)
        logger.info(f"Detected {total} people")

        # 4. Save annotated image to disk
        try:
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".jpg"
            cv2.imwrite(str(scan_dir / filename), annotated)
            logger.info(f"Saved: {scan_dir / filename}")
        except Exception as save_exc:
            logger.warning(f"Could not save scan image: {save_exc}")

        # 5. Seat map (requires calibration)
        calibration = db.get_calibration()
        seat_states: dict = {}
        if len(calibration) >= 4:
            await _progress("Mapping seats…", 98)
            vb = _svg_viewbox()
            seat_states = det.map_detections_to_seats(detections, calibration, vb)

        ts = datetime.now().isoformat()

        # 6. Persist
        occupied = [k for k, v in seat_states.items() if v == "occupied"]
        db.save_scan(ts, service_type, total, occupied, image_b64, raw_image_b64=raw_image_b64)

        # 7. Update shared state
        state.update(
            latest_count=total,
            latest_timestamp=ts,
            latest_service=service_type,
            latest_image_b64=image_b64,
            latest_raw_image_b64=raw_image_b64,
            latest_seat_states=seat_states,
            latest_detections=detections,
        )

        await _progress("Complete!", 100)
        await _broadcast({
            "type":           "scan_complete",
            "count":          total,
            "timestamp":      ts,
            "service_type":   service_type,
            "seat_states":    seat_states,
            "image_b64":      image_b64,
            "raw_image_b64":  raw_image_b64,
        })
        logger.info(f"Scan done: {total} people  service={service_type}  ts={ts}")

    except Exception as exc:
        logger.exception(f"Scan error: {exc}")
        state["message"] = f"Error: {exc}"
        await _broadcast({"type": "scan_error", "error": str(exc)})
    finally:
        state["running"] = False


def _svg_viewbox():
    try:
        text = SVG_PATH.read_text()
        m = re.search(r'viewBox=["\']([^"\']+)["\']', text)
        if m:
            return tuple(map(float, m.group(1).split()))
    except Exception:
        pass
    return (0.0, 0.0, 1000.0, 800.0)


# ── Startup / shutdown ────────────────────────────────────────────────────────
@app.on_event("startup")
async def _startup():
    db.init_db()
    register_scan_callback(run_scan)
    sched = build_scheduler()
    sched.start()
    logger.info("Attendance counter ready ✓")


@app.on_event("shutdown")
async def _shutdown():
    from scheduler import _scheduler
    if _scheduler and _scheduler.running:
        _scheduler.shutdown(wait=False)


# ── REST endpoints ────────────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    return {
        "running":          state["running"],
        "progress":         state["progress"],
        "message":          state["message"],
        "latest_count":     state["latest_count"],
        "latest_timestamp": state["latest_timestamp"],
        "latest_service":   state["latest_service"],
        "seat_states":      state["latest_seat_states"],
        "has_calibration":  len(db.get_calibration()) >= 4,
        "db_scan_count":    len(db.get_all_scans()),
    }


@app.post("/api/scan/trigger")
async def api_trigger(service_type: str = "Manual"):
    if state["running"]:
        raise HTTPException(409, "Scan already in progress")
    asyncio.create_task(run_scan(service_type=service_type))
    return {"status": "started", "service_type": service_type}


@app.get("/api/scan/image")
async def api_latest_image():
    b64 = state.get("latest_image_b64")
    ts  = state.get("latest_timestamp")
    if not b64:
        row = db.get_latest_scan()
        if row and row.get("stitched_image"):
            return {"image_b64": row["stitched_image"], "raw_image_b64": row.get("raw_image"), "timestamp": row["timestamp"]}
        raise HTTPException(404, "No scan image available yet")
    return {"image_b64": b64, "raw_image_b64": state.get("latest_raw_image_b64"), "timestamp": ts}


@app.get("/api/attendance")
async def api_attendance(include_archived: bool = False):
    return db.get_all_scans(include_archived=include_archived)


class ManualEntry(BaseModel):
    timestamp:    str
    service_type: str = "Manual"
    count:        int
    notes:        Optional[str] = None


@app.post("/api/attendance")
async def api_create_scan(entry: ManualEntry):
    """Manually create a historical attendance record (no image)."""
    scan_id = db.save_scan(entry.timestamp, entry.service_type, entry.count, [], None, entry.notes)
    scan = next((s for s in db.get_all_scans() if s.get("id") == scan_id), None)
    if not scan:
        raise HTTPException(500, "Failed to retrieve created scan")
    return scan


@app.get("/api/attendance/{scan_id}/image")
async def api_scan_image(scan_id: int):
    """Return the stitched image for a specific scan by ID."""
    imgs = db.get_scan_image(scan_id)
    if not imgs or not imgs.get("annotated"):
        raise HTTPException(404, f"No image saved for scan {scan_id}")
    scan = next((s for s in db.get_all_scans() if s.get("id") == scan_id), {})
    return {
        "image_b64":     imgs["annotated"],
        "raw_image_b64": imgs.get("raw"),
        "timestamp":     scan.get("timestamp"),
        "count":         scan.get("count"),
    }


class ScanUpdate(BaseModel):
    notes:        Optional[str]  = None
    manual_add:   Optional[int]  = None
    service_type: Optional[str]  = None
    archived:     Optional[bool] = None


@app.patch("/api/attendance/{scan_id}")
async def api_update_scan(scan_id: int, update: ScanUpdate):
    """Update notes, manual_add, service_type, and/or archived for a scan."""
    db.update_scan(scan_id, notes=update.notes, manual_add=update.manual_add,
                   service_type=update.service_type, archived=update.archived)
    # Always fetch with include_archived=True so we can return the record even if just archived
    scan = next((s for s in db.get_all_scans(include_archived=True) if s.get("id") == scan_id), None)
    if not scan:
        raise HTTPException(404, f"Scan {scan_id} not found")
    return scan


@app.get("/api/svg")
async def api_svg():
    if SVG_PATH.exists():
        return FileResponse(SVG_PATH, media_type="image/svg+xml")
    return HTMLResponse(_placeholder_svg(), media_type="image/svg+xml")


# ── Calibration ───────────────────────────────────────────────────────────────
@app.get("/api/calibration")
async def api_get_cal():
    return db.get_calibration()


class CalPoint(BaseModel):
    seat_id: str
    svg_x: float
    svg_y: float
    photo_x: float
    photo_y: float


@app.post("/api/calibration")
async def api_save_cal(pt: CalPoint):
    db.save_calibration_point(pt.seat_id, pt.svg_x, pt.svg_y, pt.photo_x, pt.photo_y)
    return {"status": "saved", "seat_id": pt.seat_id}


@app.delete("/api/calibration/{seat_id}")
async def api_del_cal(seat_id: str):
    db.delete_calibration_point(seat_id)
    return {"status": "deleted"}


@app.delete("/api/calibration")
async def api_clear_cal():
    db.clear_calibration()
    return {"status": "cleared"}


# ── Camera snapshot (for calibration wizard) ──────────────────────────────────
@app.get("/api/capture")
async def api_capture():
    frame = cam.capture_frame()
    if frame is None:
        raise HTTPException(503, "Could not capture frame from camera")
    return {"image_b64": stitch.to_base64(frame, quality=90)}


# ── Manual PTZ control ────────────────────────────────────────────────────────
PTZ_ACTIONS = {
    "left":     lambda s: cam.pan_left(s),
    "right":    lambda s: cam.pan_right(s),
    "up":       lambda s: cam.tilt_up(s),
    "down":     lambda s: cam.tilt_down(s),
    "stop":     lambda _: cam.stop(),
    "zoomin":   lambda s: cam.zoom_in(s),
    "zoomout":  lambda s: cam.zoom_out(s),
    "zoomstop": lambda _: cam.zoom_stop(),
    "home":     lambda _: cam.go_home(),
}


@app.get("/api/ptz/position")
async def api_ptz_position():
    return await cam.get_position()


@app.post("/api/ptz/{action}")
async def api_ptz(action: str, speed: int = 10):
    fn = PTZ_ACTIONS.get(action.lower())
    if fn is None:
        raise HTTPException(400, f"Unknown action: {action}")
    await fn(speed)
    return {"status": "ok", "action": action}


# ── WebSocket ─────────────────────────────────────────────────────────────────
@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    _ws_clients.append(websocket)
    await websocket.send_json({"type": "state", **{k: v for k, v in state.items() if k != "latest_detections"}})
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        if websocket in _ws_clients:
            _ws_clients.remove(websocket)


# ── Placeholder SVG ───────────────────────────────────────────────────────────
def _placeholder_svg() -> str:
    import math
    seats: list[str] = []
    sid = 1
    cx, cy = 500, 900

    for row in range(28):
        r = 130 + row * 24
        a0, a1 = -72, 72
        n = max(6, int((a1 - a0) * r * math.pi / 180 / 21))
        for i in range(n):
            angle = math.radians(a0 + (a1 - a0) * i / max(n - 1, 1))
            x = cx + r * math.sin(angle)
            y = cy - r * math.cos(angle)
            seats.append(
                f'<circle id="seat_{sid}" class="seat" cx="{x:.1f}" cy="{y:.1f}" r="7" '
                f'data-section="main" data-row="{row+1}" data-num="{i+1}"/>'
            )
            sid += 1

    for row in range(8):
        r = 820 + row * 26
        a0, a1 = -82, 82
        n = max(12, int((a1 - a0) * r * math.pi / 180 / 21))
        for i in range(n):
            angle = math.radians(a0 + (a1 - a0) * i / max(n - 1, 1))
            x = cx + r * math.sin(angle)
            y = cy - r * math.cos(angle)
            seats.append(
                f'<circle id="seat_{sid}" class="seat" cx="{x:.1f}" cy="{y:.1f}" r="7" '
                f'data-section="balcony" data-row="B{row+1}" data-num="{i+1}"/>'
            )
            sid += 1

    body = "\n  ".join(seats)
    return f"""<?xml version="1.0" encoding="UTF-8"?>
<svg xmlns="http://www.w3.org/2000/svg" viewBox="-700 -80 2400 1100" id="seat-map">
  <style>
    .seat {{ fill: #4b5563; stroke: #374151; stroke-width:1; cursor:pointer; transition: fill 0.3s; }}
    .seat.occupied {{ fill: #22c55e; }}
    .seat.empty    {{ fill: #ef4444; }}
    .seat.pending  {{ fill: #6b7280; }}
    .seat:hover    {{ stroke: #f9fafb; stroke-width:2; }}
  </style>
  <rect x="350" y="870" width="300" height="55" rx="8" fill="#1e293b"/>
  <text x="500" y="905" text-anchor="middle" fill="#64748b" font-size="22" font-family="sans-serif">STAGE</text>
  {body}
</svg>"""


# ── Serve React SPA (must be last) ────────────────────────────────────────────
if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="spa")
