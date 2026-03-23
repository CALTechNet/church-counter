# Lakeshore Church — Attendance Counter

Auto-counts congregation via PTZ camera + YOLOv8n AI, with a web UI showing seat map, stitched photo, and attendance history.

## Quick Start

```bash
# 1. Clone / copy this folder to your server
# 2. (Optional) Drop your seats.svg into ./config/seats.svg
# 3. Build and run
docker compose up -d --build

# 4. Open the UI
http://<server-ip>:8000
```

## First Run Checklist

1. **Test camera connection** — ensure the Docker host can reach `10.10.140.140`
2. **Run a manual scan** — click "Scan Now" in the UI, select "Manual"
3. **Calibrate** — click "⚙ Calibrate", capture a photo, mark at least 4 matching seat positions in both photo and SVG map for per-seat coloring to work
4. **Check scheduled scans** — auto-runs at:
   - Sunday 9:45 AM CT  → "Sunday Morning"
   - Sunday 11:30 AM CT → "Sunday Midday"
   - Wednesday 7:30 PM CT → "Wednesday Evening"

## Swapping in Your SVG

Replace `./config/seats.svg` with your actual seat layout SVG.  
Requirements:
- Each seat must be an element with a **unique `id`** attribute (e.g. `id="seat_42"`, `id="A-12"`, etc.)
- Include a `viewBox` attribute on the root `<svg>` element

No restart needed — the SVG is loaded fresh on each page load.

## Camera

- **IP:** 10.10.140.140
- **RTSP:** rtsp://10.10.140.140:554/1
- **PTZ API:** http://10.10.140.140/cgi-bin/ptzctrl.cgi
- Scan strategy: auto-pan across main floor, tilt up for balcony, return home

### Tuning the scan sweep

Edit `backend/camera.py` constants if your room is larger/smaller:

| Constant | Default | Description |
|---|---|---|
| `PAN_SPEED_SCAN` | 4 | Pan speed during capture (1=slow, 24=fast) |
| `PAN_FULL_TIME` | 14 | Seconds for a full left→right sweep |
| `FRAME_INTERVAL` | 1.8 | Seconds between frame captures |
| `TILT_UP_TIME` | 2.2 | Seconds to tilt up to balcony |
| `TILT_SPEED` | 5 | Tilt speed (1=slow, 20=fast) |

## Architecture

```
Single Docker container
├── FastAPI (port 8000)
│   ├── REST API  (/api/*)
│   ├── WebSocket (/ws)  — live scan progress
│   └── Static SPA       — React frontend
├── APScheduler          — 3 weekly jobs
├── YOLOv8n              — person detection (CPU)
├── OpenCV Stitcher      — panorama from frames
└── SQLite (/data/church.db)
```

## Data

All scan history is in `./data/church.db` (SQLite).  
Export attendance CSV from the **Data** tab in the UI.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Camera unreachable | Check that Docker host network can ping 10.10.140.140 |
| RTSP fails | Try `ffplay rtsp://10.10.140.140:554/1` from host |
| Stitching fails | Check `docker logs church-attendance` — falls back to horizontal concat |
| YOLOv8 model not found | Container downloads `yolov8n.pt` on first run — needs internet on first start |
| Seats not coloring | Run Calibration wizard (≥4 points required) |
