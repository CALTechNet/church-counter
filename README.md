# Lakeshore Church — Attendance Counter

Auto-counts congregation via PTZ camera + YOLO26x AI, with a mobile-responsive web UI showing a seat map, stitched panorama photo, attendance graphs, and full scan history.

## Quick Start

```bash
# 1. Clone / copy this folder to your server
# 2. (Optional) Drop your seats.svg into ./config/seats.svg
# 3. Build and run
docker compose up -d --build

# 4. Open the UI
http://<server-ip>:80
```

## First Run Checklist

1. **Test camera connection** — ensure the Docker host can reach `10.10.140.140`
2. **Open Settings** — configure your church name, camera IP, and scan mode (preset or calibrated)
3. **Run a manual scan** — click "Scan Now" in the UI, select "Manual"
4. **Calibrate** — click "Calibrate", capture a photo, mark ≥4 matching seat positions in both the photo and SVG map for per-seat coloring to work
5. **Check scheduled scans** — auto-runs at:
   - Sunday 9:45 AM CT → "Sunday Morning"
   - Sunday 11:30 AM CT → "Sunday Midday"
   - Wednesday 7:30 PM CT → "Wednesday Evening"

## UI Overview

The web interface has four tabs:

| Tab | Description |
|---|---|
| **Photo** | Stitched panorama from the latest scan, with detected people highlighted. Toggle between clean / overlay / count view modes. Browse image history via dropdown. |
| **Attendance** | Charts of attendance over time, broken down by service type, with date range filters and statistics |
| **Data** | Full scan history table with search, edit (notes / manual count / service type), archive, and CSV export |
| **Live View** | Real-time camera feed (~10 fps) with on-screen PTZ controls for pan, tilt, and zoom |

**Header controls:**
- Service type selector (Manual, Sunday Morning, Sunday Midday, Wednesday Evening)
- **Scan Now** — triggers an immediate scan with estimated time display
- **Reset** — appears after a scan has been running >5 minutes; force-clears stuck state
- **Calibrate** — opens the calibration wizard
- **Settings** — church name, camera config, scan mode, multi-room setup

A live progress bar and status message (with estimated time remaining) are shown while a scan is in progress. Toast notifications confirm scan completion or report errors.

The UI is fully **mobile-responsive** — header controls reflow to a second line on small screens, tab labels expand to fill the width, and font/padding scale down appropriately.

## Scan Modes

The system supports two scan modes, configurable per room in Settings:

| Mode | Description |
|---|---|
| **Preset** | Visits saved PTZ presets (default: 100–131) in sequence. Fast and reliable when presets are pre-configured on the camera. |
| **Calibrated** | Computes a grid of pan/tilt positions from user-defined bounds (left, right, top, bottom edges + zoom level). More flexible — no presets needed on the camera. |

Both modes capture frames continuously during camera movement for dense stitching coverage, then return to home (preset 0).

## Multi-Room Support

The system supports scanning multiple rooms sequentially. Each room is configured independently in Settings with its own:
- Camera type (`ptz_optics` or `rtsp`)
- Camera IP and credentials
- Scan mode (preset or calibrated)
- Preset range or calibrated bounds

Scheduled scans iterate through all configured rooms automatically.

## Swapping in Your SVG

Replace `./config/seats.svg` with your actual seat layout SVG.

Requirements:
- Each seat must be an element with a **unique `id`** attribute (e.g. `id="seat_42"`, `id="A-12"`)
- Include a `viewBox` attribute on the root `<svg>` element

No restart needed — the SVG is loaded fresh on each page load.

## Camera

- **Default IP:** 10.10.140.140
- **RTSP:** rtsp://<camera-ip>:554/1
- **PTZ protocol:** VISCA over TCP (port 5678)
- **Scan strategy:** visits presets in a vertical-S (boustrophedon) grid pattern, capturing frames continuously during movement, then returns to home (preset 0)

### Tuning the scan timing

Edit `backend/camera.py` constants if your room requires different timing:

| Constant | Default | Description |
|---|---|---|
| `SCAN_PRESETS` | range(100, 132) | Preset IDs to visit (32 positions) |
| `HOME_PRESET` | 0 | Preset to call after scan completes |
| `TRAVEL_TIME` | 1.5 | Seconds camera takes to travel between adjacent presets |
| `SETTLE_TIME` | 0 | Extra seconds to wait after travel before capturing |
| `CAPTURE_INTERVAL` | 0.25 | Seconds between frame captures during movement |

### Learn Presets

The "Learn Presets" API (`POST /api/ptz/learn-presets`) visits every scan preset and records its pan/tilt position. After learning, preset scans use `move_abs` at maximum speed instead of relying on the camera's slower internal preset-recall.

## Detection Pipeline

1. Camera moves through presets (or calibrated grid); frames captured continuously during movement
2. Automatic **lens distortion correction** (Brown–Conrady model, k1=-0.32, k2=0.12)
3. Frames stitched into a panorama:
   - **Preset scans** → OpenCV Stitcher (feature-based)
   - **Calibrated scans** → Grid-aware stitcher with position-guided alignment, phase correlation, and translation-only constraints to prevent ghosting
4. **CLAHE + gamma correction** (γ=1.8) applied for improved detection in dark church interiors
5. Panorama split into a **10 × 8 tiled grid with 50% overlap** — tiles run through YOLO26x in batches
6. Cross-tile NMS deduplication (IOU threshold 0.30) removes double-counts at tile edges
7. Detections mapped to seat IDs via calibration points (affine transform)
8. Results persisted to SQLite with both annotated and raw panorama images
9. Annotated images also saved to disk at `/data/scans/`

## Calibration

The calibration wizard lets you mark matching point pairs between the stitched photo and the SVG seat map. With ≥4 calibrated points, the UI colours each seat green (empty) or red (occupied) in the seat map.

The wizard also supports numeric VISCA value inputs for precise PTZ positioning during calibration.

Calibration data is stored in `./data/church.db` and survives container restarts.

## Architecture

```
Single Docker container (12 GB memory limit)
├── FastAPI (port 8000, exposed as 80)
│   ├── REST API  (/api/*)
│   ├── WebSocket (/ws)  — live scan progress broadcast
│   └── Static SPA       — React + Vite frontend
├── APScheduler          — 3 weekly cron jobs (America/Chicago)
├── YOLO26x              — tiled person detection (CPU)
│   ├── 10×8 tile grid, 50% overlap
│   ├── Batched inference (TILE_BATCH_SIZE tiles per call)
│   └── Cross-tile NMS deduplication
├── OpenCV Stitcher      — panorama from captured frames
│   ├── Lens distortion correction
│   ├── Grid-aware stitching (calibrated mode)
│   └── CLAHE + gamma brightness enhancement
├── VISCA TCP            — PTZ camera control (port 5678)
├── Live capture thread  — ~10 fps background frame polling
└── SQLite (/data/church.db)
    ├── scans        — images, counts, metadata
    ├── calibration  — photo↔SVG point pairs
    └── config       — app settings, camera bounds
```

### API Endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/status` | Current scan state, latest counts, seat states |
| POST | `/api/scan/trigger` | Start a scan (`?service_type=...&room_id=...`) |
| POST | `/api/scan/cancel` | Cancel an in-progress scan |
| GET | `/api/scan/image` | Latest stitched panorama (annotated + raw) |
| GET | `/api/attendance` | All scan records (`?include_archived=false`) |
| POST | `/api/attendance` | Create a manual historical attendance record |
| GET | `/api/attendance/{id}/image` | Panorama image for a specific scan |
| PATCH | `/api/attendance/{id}` | Update notes, manual count, service type, or archive status |
| GET/POST/DELETE | `/api/calibration` | Manage calibration points |
| GET | `/api/capture` | Single camera snapshot |
| GET | `/api/settings` | App settings (church name, rooms, scan config) |
| POST | `/api/settings` | Update app settings |
| GET | `/api/camera-bounds` | Calibrated scan bounds (left/right/top/bottom/zoom) |
| POST | `/api/camera-bounds` | Save camera bounds |
| GET | `/api/ptz/position` | Current camera pan/tilt/zoom values |
| POST | `/api/ptz/{action}` | Manual PTZ control (left/right/up/down/zoomin/zoomout/home/stop) |
| POST | `/api/ptz/goto-bound` | Move camera to a specific bound corner |
| POST | `/api/ptz/learn-presets` | Visit all presets and record their positions |
| POST | `/api/live-frame/start` | Start persistent live capture thread |
| POST | `/api/live-frame/stop` | Stop live capture |
| GET | `/api/live-frame` | Fetch latest live frame (JPEG base64) |
| GET | `/api/svg` | SVG seat layout |
| WS | `/ws` | WebSocket for live progress events |

## Data

All scan history is in `./data/church.db` (SQLite). Tables:

- **scans** — id, timestamp, service_type, total_count, occupied_seats, stitched_image (annotated), raw_image, notes, manual_add, archived, room
- **calibration** — seat_id, svg_x, svg_y, photo_x, photo_y, updated_at
- **config** — key/value store for app settings and camera bounds

Annotated scan images are also saved as JPEGs in `./data/scans/` for easy browsing outside the UI.

Export attendance CSV from the **Data** tab in the UI.

## CI/CD

A GitHub Actions workflow (`.github/workflows/docker-publish.yml`) builds and pushes a signed Docker image to `ghcr.io` on:
- Push or PR to `main`
- Semver tags (`v*.*.*`)
- Daily schedule (18:32 UTC)

Images are signed with [cosign](https://github.com/sigstore/cosign) via the Sigstore Fulcio CA.

## Frontend Build Versioning

The frontend version is tracked in `frontend/src/version.js` as `v1.2.{BUILD}`. The `BUILD` number is **automatically incremented** by a `prebuild` script every time `npm run build` is executed (including during `docker compose up --build`).

To check the current version, look at the bottom-right footer of the UI.

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `PYTHONUNBUFFERED` | `1` | Unbuffered Python logging |
| `TZ` | `America/Chicago` | Timezone for scheduler cron jobs |
| `TILE_BATCH_SIZE` | `4` | Tiles per YOLO batch call (raise to 8–12 on 16 GB+ hosts) |
| `YOLO_IMGSZ` | `640` | Inference resolution per tile (1280 for higher accuracy) |
| `YOLO_MODEL` | `yolo26x.pt` | Model path (or OpenVINO directory) |

## Optional: OpenVINO Acceleration

For 2–4× faster inference on Intel CPUs, export the YOLO26x model to OpenVINO INT8 format:

```bash
bash export_openvino.sh
```

This produces `./models/yolo26x_openvino_model/`. Then set `YOLO_MODEL=/models/yolo26x_openvino_model` in `docker-compose.yml`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Camera unreachable | Confirm Docker host network can ping the camera IP; check `network_mode: bridge` in docker-compose.yml |
| RTSP fails | Run `ffplay rtsp://<camera-ip>:554/1` from the host to verify stream |
| Stitching fails | Check `docker logs church-attendance` — falls back to horizontal concat automatically |
| Panorama ghosting/distortion | Lens correction is automatic; for calibrated scans, ensure camera bounds are set accurately |
| YOLO26x model not found | Container downloads `yolo26x.pt` on first run — needs internet access on first start |
| Seats not coloring | Open Calibration wizard and mark ≥4 point pairs |
| Scan stuck / spinner won't stop | Wait >5 min for the "Reset" button to appear, or restart the container |
| Per-seat colours wrong after SVG change | Re-run calibration — point mappings are tied to SVG coordinates |
| Docker OOM crash | Container is limited to 12 GB; reduce `TILE_BATCH_SIZE` or `YOLO_IMGSZ` if memory is tight |
