# lcc-headcount

Backend ML starter for controlling a PTZ camera, taking section snapshots, and estimating occupied seats.

## What this provides

- PTZ control using ONVIF presets.
- Snapshot capture from the camera.
- Person detection using YOLOv8.
- Seat occupancy count based on seat polygons.

## Architecture

1. **PTZ move**: move camera to a predefined preset (e.g., left, center, right).
2. **Capture**: take one still image for that preset.
3. **Detect people**: run YOLO person detection on the image.
4. **Map to seats**: treat each seat as a polygon; if a detected person centroid falls inside, mark seat occupied.
5. **Aggregate**: combine all section counts into one total headcount.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create a `seat_map.json` file:

```json
{
  "presets": [
    {
      "token": "1",
      "name": "left_section",
      "snapshot_url": "http://CAMERA_IP/snapshot.jpg",
      "seats": [
        {"id": "L1-01", "polygon": [[120, 320], [160, 320], [165, 360], [115, 360]]}
      ]
    }
  ]
}
```

Run:

```bash
python backend_ml.py
```

## Notes for church deployment

- Calibrate seat polygons from real camera views (one map per preset).
- Keep presets fixed and avoid changing zoom/focus once calibrated.
- Run 2-3 snapshots per preset and use median count for stability.
- Add a simple confidence threshold alert if occupancy suddenly spikes/drop.
- Prefer on-prem processing to avoid uploading sanctuary images externally.

## Next improvements

- Expose `run_headcount` through a FastAPI endpoint.
- Save snapshots + count JSON for auditing.
- Add tracking or pose estimation to reduce double-counting edge cases.
- Add a nightly calibration check against known empty-room baseline.
