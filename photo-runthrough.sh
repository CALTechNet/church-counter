#!/bin/bash

SCAN_DIR="${DATA_DIR:-/opt/church-counter/data}/scans"

echo ""
echo "========================================="
echo "  Church Attendance — Detection Tester"
echo "========================================="
echo ""

# List jpg files
files=()
i=1
for f in "$SCAN_DIR"/*.jpg; do
    [ -f "$f" ] || continue
    [[ "$(basename "$f")" == test_* ]] && continue
    files+=("$f")
    echo "  [$i] $(basename "$f")"
    ((i++))
done

if [ ${#files[@]} -eq 0 ]; then
    echo "No scan images found in $SCAN_DIR"
    exit 1
fi

echo ""
read -p "Select a file [1-${#files[@]}]: " choice

if ! [[ "$choice" =~ ^[0-9]+$ ]] || [ "$choice" -lt 1 ] || [ "$choice" -gt "${#files[@]}" ]; then
    echo "Invalid selection."
    exit 1
fi

selected="${files[$((choice-1))]}"
filename=$(basename "$selected")
run_ts=$(date +"%Y-%m-%d_%H-%M-%S")
run_ts_iso=$(date +"%Y-%m-%dT%H:%M:%S")
test_filename="test_${run_ts}_${filename}"

echo ""
echo "Running detection on: $filename"
echo "Output will be saved as: $test_filename"
echo "This will take several minutes with YOLO26x + tiling on CPU."
echo ""

docker exec -i church-attendance python3 -u << PYEOF
import sys
import logging
import time
import os
import base64

# Enable verbose logging so we can see what's happening
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    stream=sys.stdout,
)

sys.path.insert(0, '/app/backend')
import cv2
import detector as det
import database as db

import os as _os
data_dir    = _os.environ.get('DATA_DIR', '/opt/church-counter/data')
input_path  = f'{data_dir}/scans/$filename'
output_path = f'{data_dir}/scans/$test_filename'

print("Loading image...")
sys.stdout.flush()
image = cv2.imread(input_path)
if image is None:
    print("ERROR: Could not load image")
    sys.exit(1)

h, w = image.shape[:2]
print(f"Image size: {w}x{h}")
print(f"Tile grid: {det.TILE_COLS} cols x {det.TILE_ROWS} rows with {int(det.TILE_OVERLAP*100)}% overlap")
print(f"Total tiles: {det.TILE_COLS * det.TILE_ROWS}")
print("")
sys.stdout.flush()

print("Loading YOLO model...")
sys.stdout.flush()
t0 = time.time()
model = det._get_model()
print(f"Model loaded in {time.time()-t0:.1f}s")
print("")
sys.stdout.flush()

print("Running tiled detection...")
sys.stdout.flush()
try:
    t1 = time.time()
    detections = det.detect_people(image)
    elapsed = time.time() - t1
except Exception as e:
    print(f"ERROR during detection: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("")
print("=========================================")
print(f"  RESULT: {len(detections)} people detected")
print(f"  Detection time: {elapsed:.1f}s")
print("=========================================")
sys.stdout.flush()

if detections:
    print("")
    print(f"  {'#':<4} {'Confidence':<12} {'Center':<20} {'Box'}")
    print(f"  {'-'*60}")
    for i, d in enumerate(detections, 1):
        print(f"  {i:<4} {d['confidence']:<12} ({d['cx']:>5},{d['cy']:>5})       ({d['x1']},{d['y1']}) -> ({d['x2']},{d['y2']})")

print("")
print("Saving annotated image...")
sys.stdout.flush()
annotated = det.draw_detections(image, detections)
ok = cv2.imwrite(output_path, annotated)
if ok:
    size_kb = os.path.getsize(output_path) // 1024
    print(f"Saved: {output_path} ({size_kb} KB)")
else:
    print(f"ERROR: cv2.imwrite failed for {output_path}")
    print(f"  Directory exists: {os.path.isdir(os.path.dirname(output_path))}")
    print(f"  Directory writable: {os.access(os.path.dirname(output_path), os.W_OK)}")
sys.stdout.flush()

print("")
print("Saving to database...")
sys.stdout.flush()
try:
    _, buf = cv2.imencode('.jpg', annotated, [cv2.IMWRITE_JPEG_QUALITY, 85])
    image_b64 = base64.b64encode(buf).decode('utf-8')
    scan_id = db.save_scan(
        '$run_ts_iso',
        'Test',
        len(detections),
        [],
        image_b64,
        'Test rerun of $filename',
    )
    print(f"Database entry created: scan #{scan_id}")
except Exception as e:
    print(f"WARNING: Could not save to database: {e}")
    import traceback
    traceback.print_exc()
sys.stdout.flush()
PYEOF

EXIT_CODE=$?
echo ""
if [ $EXIT_CODE -eq 0 ]; then
    echo "Done! Annotated image saved to:"
    echo "  $SCAN_DIR/${test_filename}"
    echo "Entry also added to the Data and Attendance tabs (service type: Test)."
else
    echo "Detection failed (exit code $EXIT_CODE). Check output above for errors."
fi
echo ""
