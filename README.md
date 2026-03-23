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
2. **Run a manual scan** — click "▶ Scan Now" in the UI, select "Manual"
3. **Calibrate** — click "⚙ Calibrate", capture a photo, mark ≥4 matching seat positions in both the photo and SVG map for per-seat coloring to work
4. **Check scheduled scans** — auto-runs at:
   - Sunday 9:45 AM CT  → "Sunday Morning"
   - Sunday 11:30 AM CT → "Sunday Midday"
   - Wednesday 7:30 PM CT → "Wednesday Evening"

## UI Overview

The web interface has three tabs:

| Tab | Description |
|---|---|
| **Photo** | Stitched panorama from the latest scan, with detected people highlighted |
| **Attendance** | Charts of attendance over time, broken down by service type |
| **Data** | Full scan history table with search, edit (notes / manual count), archive, and CSV export |

**Header controls:**
- Service type selector (Manual, Sunday Morning, Sunday Midday, Wednesday Evening)
- **▶ Scan Now** — triggers an immediate scan
- **✕ Reset** — appears after a scan has been running >5 minutes; force-clears stuck state
- **⚙ Calibrate** — opens the calibration wizard

A live progress bar and status message are shown while a scan is in progress. Toast notifications confirm scan completion or report errors.

The UI is fully **mobile-responsive** — header controls reflow to a second line on small screens, tab labels expand to fill the width, and font/padding scale down appropriately.

## Swapping in Your SVG

Replace `./config/seats.svg` with your actual seat layout SVG.

Requirements:
- Each seat must be an element with a **unique `id`** attribute (e.g. `id="seat_42"`, `id="A-12"`)
- Include a `viewBox` attribute on the root `<svg>` element

No restart needed — the SVG is loaded fresh on each page load.

## Camera

- **IP:** 10.10.140.140
- **RTSP:** rtsp://10.10.140.140:554/1
- **PTZ API:** http://10.10.140.140/cgi-bin/ptzctrl.cgi
- **Scan strategy:** visits 32 saved presets (100–131) in sequence, capturing frames continuously during each camera movement for dense stitching coverage, then returns to home (preset 0)

### Tuning the scan timing

Edit `backend/camera.py` constants if your room requires different timing:

| Constant | Default | Description |
|---|---|---|
| `SCAN_PRESETS` | range(100, 132) | Preset IDs to visit (32 positions) |
| `HOME_PRESET` | 0 | Preset to call after scan completes |
| `TRAVEL_TIME` | 1.5 | Seconds camera takes to travel between adjacent presets |
| `SETTLE_TIME` | 0 | Extra seconds to wait after travel before capturing |
| `CAPTURE_INTERVAL` | 0.25 | Seconds between frame captures during movement |

## Detection Pipeline

1. Camera moves through 32 presets; frames captured continuously during movement
2. Frames deduplicated and stitched into a panorama (OpenCV Stitcher, falls back to horizontal concat)
3. Panorama split into a **6 × 4 tiled grid with 30% overlap** — tiles run through YOLO26x in batches of up to 24
4. Cross-tile NMS deduplication (IOU threshold 0.30) removes double-counts at tile edges
5. CLAHE + gamma correction applied to improve detection in low-light conditions
6. Detections mapped to seat IDs via calibration points
7. Results persisted to SQLite with a base64-encoded panorama image

## Calibration

The calibration wizard lets you mark matching point pairs between the stitched photo and the SVG seat map. With ≥4 calibrated points, the UI colours each seat green (empty) or red (occupied) in the seat map.

Calibration data is stored in `./data/church.db` and survives container restarts.

## Architecture

```
Single Docker container
├── FastAPI (port 8000)
│   ├── REST API  (/api/*)
│   ├── WebSocket (/ws)  — live scan progress broadcast
│   └── Static SPA       — React + Vite frontend
├── APScheduler          — 3 weekly cron jobs (America/Chicago)
├── YOLO26x              — tiled person detection (CPU)
├── OpenCV Stitcher      — panorama from captured frames
└── SQLite (/data/church.db)
```

### API endpoints

| Method | Path | Description |
|---|---|---|
| GET | `/api/status` | Current scan state, latest counts, seat states |
| POST | `/api/scan/trigger` | Start a scan (body: `{"service_type": "..."}`) |
| GET | `/api/scan/image` | Latest stitched panorama (base64) |
| GET | `/api/attendance` | All scan records |
| PATCH | `/api/attendance/{id}` | Update notes or manual count for a scan |
| GET/POST/DELETE | `/api/calibration` | Manage calibration points |
| GET | `/api/capture` | Single camera snapshot |
| POST | `/api/ptz/{action}` | Manual PTZ control |
| GET | `/api/svg` | SVG seat layout |
| WS | `/ws` | WebSocket for live progress events |

## Data

All scan history is in `./data/church.db` (SQLite). Tables:

- **scans** — id, timestamp, service_type, total_count, occupied_seats, stitched_image_b64, notes, manual_add, archived
- **calibration** — seat_id, svg_x, svg_y, photo_x, photo_y

Export attendance CSV from the **Data** tab in the UI.

## CI/CD

A GitHub Actions workflow (`.github/workflows/docker-publish.yml`) builds and pushes a signed Docker image to `ghcr.io` on:
- Push or PR to `main`
- Semver tags (`v*.*.*`)
- Daily schedule (18:32 UTC)

Images are signed with [cosign](https://github.com/sigstore/cosign) via the Sigstore Fulcio CA.

## Frontend Build Versioning

The frontend version is tracked in `frontend/src/version.js` as `v1.1.{BUILD}`. The `BUILD` number is **automatically incremented** by a `prebuild` script every time `npm run build` is executed (including during `docker compose up --build`).

To check the current version, look at the bottom-right footer of the UI.

## Optional: OpenVINO Acceleration

For 2–4× faster inference on Intel CPUs, export the YOLO26x model to OpenVINO INT8 format:

```bash
bash export_openvino.sh
```

This produces `./models/yolo26x_openvino_model/`. Then set `YOLO_MODEL=/models/yolo26x_openvino_model` in `docker-compose.yml`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Camera unreachable | Confirm Docker host network can ping 10.10.140.140; check `network_mode: bridge` in docker-compose.yml |
| RTSP fails | Run `ffplay rtsp://10.10.140.140:554/1` from the host to verify stream |
| Stitching fails | Check `docker logs church-attendance` — falls back to horizontal concat automatically |
| YOLO26x model not found | Container downloads `yolo26x.pt` on first run — needs internet access on first start |
| Seats not coloring | Open Calibration wizard and mark ≥4 point pairs |
| Scan stuck / spinner won't stop | Wait >5 min for the "✕ Reset" button to appear, or restart the container |
| Per-seat colours wrong after SVG change | Re-run calibration — point mappings are tied to SVG coordinates |
