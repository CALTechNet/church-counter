"""
Lakeshore Church — Attendance Counter API
FastAPI + APScheduler, single container.
"""
import asyncio
import logging
import math
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

# ── Log GPU status at startup ────────────────────────────────────────────────
def _log_hardware_info():
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024 / 1024
            logger.info(f"Hardware: GPU acceleration enabled — {gpu_name} ({vram:.0f} MB VRAM)")
            logger.info(f"Hardware: CUDA {torch.version.cuda}, PyTorch {torch.__version__}")
        else:
            logger.info("Hardware: No GPU detected — running in CPU-only mode")
    except ImportError:
        logger.info("Hardware: PyTorch not available — running in CPU-only mode")

_log_hardware_info()

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
_scan_cancel = asyncio.Event()


class _ScanCancelledError(Exception):
    pass


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


def _estimate_scan_positions(room: dict) -> int:
    """Estimate the total number of capture positions for a room's scan config."""
    camera_type = room.get("camera_type", "ptz_optics")
    if camera_type == "rtsp":
        return 1
    scan_mode = room.get("scan_mode", "preset")
    if scan_mode == "calibrated":
        bounds = db.get_config("camera_bounds", {})
        left = bounds.get("left") or (bounds.get("top_left", {}) or {}).get("pan")
        right = bounds.get("right") or (bounds.get("bottom_right", {}) or {}).get("pan")
        top = bounds.get("top") or (bounds.get("top_left", {}) or {}).get("tilt")
        bottom = bounds.get("bottom") or (bounds.get("bottom_right", {}) or {}).get("tilt")
        if None in (left, right, top, bottom):
            return 24  # fallback
        zoom = int(bounds.get("zoom") or 10000)
        pan_step = max(25, int(1_000_000 / zoom))
        tilt_step = max(25, int(pan_step * 0.65))
        cols = max(1, math.ceil(abs(int(right) - int(left)) / pan_step) + 1)
        rows = max(1, math.ceil(abs(int(bottom) - int(top)) / tilt_step) + 1)
        return rows * cols
    else:
        preset_start = int(room.get("preset_start", 100))
        preset_end = int(room.get("preset_end", 131))
        return preset_end - preset_start + 1


async def run_scan(service_type: str = "Manual", room_id: str = None):
    if state["running"]:
        logger.warning("Scan already in progress — ignoring request")
        return

    _scan_cancel.clear()
    state["running"] = True
    live_was_running = False

    room = _get_room(room_id)
    room_name = room.get("name", "Unknown")
    camera_type = room.get("camera_type", "ptz_optics")
    total_positions = _estimate_scan_positions(room)

    await _broadcast({"type": "scan_started", "service_type": service_type, "room": room_name, "total_positions": total_positions})

    try:
        import cv2
        scan_dir = Path("/data/scans")
        scan_dir.mkdir(parents=True, exist_ok=True)

        # 1. Capture
        await _progress(f"Scanning {room_name}…", 0)
        frames, grid_shape, positions = await cam.auto_scan(
            progress_callback=_progress,
            cancel_event=_scan_cancel,
            room_config=room,
        )
        if _scan_cancel.is_set():
            raise _ScanCancelledError()
        if not frames:
            raise RuntimeError(f"Camera returned 0 frames for {room_name}")

        # Pause live view during processing to free up resources
        live_was_running = cam._live_thread is not None and cam._live_thread.is_alive()
        if live_was_running:
            cam.stop_live_capture()
            logger.info("Live view paused for YOLO processing")

        # 2. Stitch (only for PTZ cameras with multiple frames)
        if camera_type == "rtsp" or len(frames) == 1:
            # Single frame — no stitching needed
            panorama = frames[0]
            logger.info(f"Single frame capture for {room_name} → shape {panorama.shape}")
        else:
            await _progress("Stitching panorama…", 93)
            scan_mode = room.get("scan_mode", "preset")
            panorama, stitch_status = stitch.stitch_frames(
                frames, grid_shape=grid_shape, positions=positions, scan_mode=scan_mode,
            )
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
        logger.info(f"Detected {total} people in {room_name}")

        # 4. Save annotated image to disk
        try:
            filename = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + f"_{room.get('id', 'default')}.jpg"
            cv2.imwrite(str(scan_dir / filename), annotated)
            logger.info(f"Saved: {scan_dir / filename}")
        except Exception as save_exc:
            logger.warning(f"Could not save scan image: {save_exc}")

        # 5. Seat map (requires calibration — only for PTZ cameras)
        calibration = db.get_calibration()
        seat_states: dict = {}
        if camera_type != "rtsp" and len(calibration) >= 4:
            await _progress("Mapping seats…", 98)
            vb = _svg_viewbox()
            seat_states = det.map_detections_to_seats(detections, calibration, vb)

        ts = datetime.now().isoformat()

        # 6. Persist
        occupied = [k for k, v in seat_states.items() if v == "occupied"]
        db.save_scan(ts, service_type, total, occupied, image_b64,
                     raw_image_b64=raw_image_b64, room=room_name)

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
            "room":           room_name,
            "seat_states":    seat_states,
            "image_b64":      image_b64,
            "raw_image_b64":  raw_image_b64,
        })
        logger.info(f"Scan done: {total} people  service={service_type}  room={room_name}  ts={ts}")

    except _ScanCancelledError:
        logger.info("Scan cancelled by user")
        state["message"] = "Cancelled"
        await _broadcast({"type": "scan_cancelled"})
    except Exception as exc:
        logger.exception(f"Scan error: {exc}")
        state["message"] = f"Error: {exc}"
        await _broadcast({"type": "scan_error", "error": str(exc)})
    finally:
        state["running"] = False
        if live_was_running:
            cam.start_live_capture()
            logger.info("Live view resumed")


async def run_all_rooms_scan(service_type: str = "Manual"):
    """Run scans across all configured rooms sequentially. Used by scheduler."""
    settings = db.get_config("app_settings", {})
    merged = {**_SETTINGS_DEFAULTS, **settings}
    merged = _ensure_rooms(merged)
    rooms = merged.get("rooms", [])

    if not rooms:
        logger.warning("No rooms configured — skipping scheduled scan")
        return

    for room in rooms:
        if _scan_cancel.is_set():
            break
        room_id = room.get("id", "default")
        logger.info(f"Scheduled scan: {service_type} — room {room.get('name', room_id)}")
        await run_scan(service_type=service_type, room_id=room_id)
        # Brief pause between rooms to let state settle
        if len(rooms) > 1:
            await asyncio.sleep(2)


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
    register_scan_callback(run_all_rooms_scan)
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
async def api_trigger(service_type: str = "Manual", room_id: str = None):
    if state["running"]:
        raise HTTPException(409, "Scan already in progress")
    asyncio.create_task(run_scan(service_type=service_type, room_id=room_id))
    return {"status": "started", "service_type": service_type, "room_id": room_id}


@app.post("/api/scan/cancel")
async def api_cancel_scan():
    if not state["running"]:
        raise HTTPException(409, "No scan in progress")
    _scan_cancel.set()
    return {"status": "cancelling"}


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


# ── App settings ──────────────────────────────────────────────────────────────
class RoomConfig(BaseModel):
    id:             Optional[str] = None
    name:           Optional[str] = None
    camera_type:    Optional[str] = None   # "ptz_optics" | "rtsp"
    camera_ip:      Optional[str] = None
    camera_user:    Optional[str] = None
    camera_pass:    Optional[str] = None
    scan_mode:      Optional[str] = None   # "preset" | "calibrated"
    preset_start:   Optional[int] = None
    preset_end:     Optional[int] = None
    preset_cols:    Optional[int] = None
    rtsp_url:       Optional[str] = None


class AppSettings(BaseModel):
    church_name:    Optional[str]  = None
    camera_ip:      Optional[str]  = None
    camera_user:    Optional[str]  = None
    camera_pass:    Optional[str]  = None
    scan_mode:      Optional[str]  = None   # "preset" | "calibrated"
    preset_start:   Optional[int]  = None
    preset_end:     Optional[int]  = None
    preset_cols:    Optional[int]  = None
    scan_positions: Optional[int]  = None
    rooms:          Optional[list] = None


_DEFAULT_ROOM = {
    "id":           "default",
    "name":         "Sanctuary",
    "camera_type":  "ptz_optics",
    "camera_ip":    "10.10.140.140",
    "camera_user":  "admin",
    "camera_pass":  "admin",
    "scan_mode":    "preset",
    "preset_start": 100,
    "preset_end":   131,
    "rtsp_url":     "",
}

_SETTINGS_DEFAULTS = {
    "church_name":    "Lakeshore Church",
    "camera_ip":      "10.10.140.140",
    "camera_user":    "admin",
    "camera_pass":    "admin",
    "scan_mode":      "preset",
    "preset_start":   100,
    "preset_end":     131,
    "scan_positions": 24,
}


def _ensure_rooms(settings: dict) -> dict:
    """Ensure settings has a rooms array, migrating from flat camera fields if needed."""
    if settings.get("rooms"):
        return settings
    # Migrate flat camera fields into a default room
    room = {
        "id":           "default",
        "name":         "Sanctuary",
        "camera_type":  "ptz_optics",
        "camera_ip":    settings.get("camera_ip")    or _SETTINGS_DEFAULTS["camera_ip"],
        "camera_user":  settings.get("camera_user")  or _SETTINGS_DEFAULTS["camera_user"],
        "camera_pass":  settings.get("camera_pass")  or _SETTINGS_DEFAULTS["camera_pass"],
        "scan_mode":    settings.get("scan_mode")     or _SETTINGS_DEFAULTS["scan_mode"],
        "preset_start": settings.get("preset_start")  or _SETTINGS_DEFAULTS["preset_start"],
        "preset_end":   settings.get("preset_end")    or _SETTINGS_DEFAULTS["preset_end"],
        "rtsp_url":     "",
    }
    settings["rooms"] = [room]
    return settings


def _get_room(room_id: str = None) -> dict:
    """Look up a room config by ID. Returns the first room if room_id is None."""
    settings = db.get_config("app_settings", {})
    merged = {**_SETTINGS_DEFAULTS, **settings}
    merged = _ensure_rooms(merged)
    rooms = merged.get("rooms", [])
    if not rooms:
        return {**_DEFAULT_ROOM}
    if room_id is None:
        return rooms[0]
    for r in rooms:
        if r.get("id") == room_id:
            return r
    return rooms[0]


@app.get("/api/settings")
async def api_get_settings():
    saved = db.get_config("app_settings", {})
    merged = {**_SETTINGS_DEFAULTS, **saved}
    merged = _ensure_rooms(merged)
    return merged


@app.post("/api/settings")
async def api_save_settings(body: AppSettings):
    existing = db.get_config("app_settings", {})
    patch = body.model_dump(exclude_none=True)
    existing.update(patch)
    db.set_config("app_settings", existing)
    merged = {**_SETTINGS_DEFAULTS, **existing}
    merged = _ensure_rooms(merged)
    return merged


# ── Camera bounds (individual pan/tilt edges + scan zoom) ─────────────────────
class CameraBounds(BaseModel):
    left:   Optional[int] = None   # pan value for left edge
    right:  Optional[int] = None   # pan value for right edge
    top:    Optional[int] = None   # tilt value for top edge
    bottom: Optional[int] = None   # tilt value for bottom edge
    zoom:   Optional[int] = None   # scan zoom level


def _compute_corners(bounds: dict) -> dict:
    """Compute all 4 corners from left/right/top/bottom bounds.
    Migrates old top_left/bottom_right format transparently."""
    left   = bounds.get("left")
    right  = bounds.get("right")
    top    = bounds.get("top")
    bottom = bounds.get("bottom")

    # Migrate from old {top_left: {pan, tilt}, bottom_right: {pan, tilt}} format
    old_tl = bounds.get("top_left")
    old_br = bounds.get("bottom_right")
    if old_tl and left is None:
        left = old_tl.get("pan")
    if old_br and right is None:
        right = old_br.get("pan")
    if old_tl and top is None:
        top = old_tl.get("tilt")
    if old_br and bottom is None:
        bottom = old_br.get("tilt")

    zoom = bounds.get("zoom", 10000)
    result = {**bounds, "left": left, "right": right, "top": top, "bottom": bottom}

    if left is not None and top is not None:
        result["top_left"]    = {"pan": left,  "tilt": top,    "zoom": zoom}
    if right is not None and top is not None:
        result["top_right"]   = {"pan": right, "tilt": top,    "zoom": zoom}
    if left is not None and bottom is not None:
        result["bottom_left"] = {"pan": left,  "tilt": bottom, "zoom": zoom}
    if right is not None and bottom is not None:
        result["bottom_right"]= {"pan": right, "tilt": bottom, "zoom": zoom}
    return result


@app.get("/api/camera-bounds")
async def api_get_bounds():
    stored = db.get_config("camera_bounds", {})
    return _compute_corners(stored)


@app.post("/api/camera-bounds")
async def api_save_bounds(bounds: CameraBounds):
    existing = db.get_config("camera_bounds", {})
    patch = bounds.model_dump(exclude_none=True)
    existing.update(patch)
    db.set_config("camera_bounds", existing)
    return _compute_corners(existing)


# ── Camera snapshot (for calibration wizard) ──────────────────────────────────
@app.get("/api/capture")
async def api_capture():
    frame = cam.capture_frame()
    if frame is None:
        raise HTTPException(503, "Could not capture frame from camera")
    return {"image_b64": stitch.to_base64(frame, quality=90)}


# ── Live frame — persistent stream for ~10 fps polling ───────────────────────
@app.post("/api/live-frame/start")
async def api_live_frame_start():
    cam.start_live_capture()
    return {"ok": True}


@app.post("/api/live-frame/stop")
async def api_live_frame_stop():
    cam.stop_live_capture()
    return {"ok": True}


@app.get("/api/live-frame")
async def api_live_frame():
    # Lazily start the persistent capture thread if not running
    if cam._live_thread is None or not cam._live_thread.is_alive():
        cam.start_live_capture()
    frame = cam.get_live_frame()
    if frame is None:
        raise HTTPException(503, "Could not capture frame from camera")
    return {"image_b64": stitch.to_base64(frame, quality=65)}


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


@app.post("/api/ptz/goto-bound")
async def api_ptz_goto_bound(corner: str = "top_left"):
    """Move camera to a computed corner (top_left, top_right, bottom_left, bottom_right)."""
    bounds = db.get_config("camera_bounds", {})
    corners = _compute_corners(bounds)
    pos = corners.get(corner)
    if not pos:
        raise HTTPException(404, f"No bound saved for corner: {corner}")
    pan  = int(pos["pan"])
    tilt = int(pos["tilt"])
    zoom = int(pos.get("zoom") or bounds.get("zoom") or 10000)
    await cam.move_abs(pan, tilt)
    await cam.zoom_abs(zoom)
    return {"status": "ok", "corner": corner, "pan": pan, "tilt": tilt, "zoom": zoom}


@app.post("/api/ptz/learn-presets")
async def api_ptz_learn_presets():
    """Visit every scan preset, record its pan/tilt position, and save to DB.
    After this runs, preset scans will use move_abs at maximum speed instead
    of relying on the camera's internal preset-recall speed."""
    positions = await cam.learn_preset_positions()
    return {"status": "ok", "learned": len(positions), "positions": positions}


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
