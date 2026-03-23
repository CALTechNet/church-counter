#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Export YOLO26x → OpenVINO INT8 for faster inference on Intel Xeon D (ENCS5412)
#
#  Run this ONCE after the container is up:
#    chmod +x export_openvino.sh
#    ./export_openvino.sh
#
#  Then activate it by adding this line to the `environment:` section of
#  docker-compose.yml and running `docker compose up -d` (no rebuild needed):
#    - YOLO_MODEL=/models/yolo26x_openvino_model
# ─────────────────────────────────────────────────────────────────────────────

set -e

MODELS_DIR="/etc/church-counter/models"
mkdir -p "$MODELS_DIR"

echo ""
echo "============================================="
echo "  YOLO26x → OpenVINO INT8 Export"
echo "============================================="
echo ""
echo "This will:"
echo "  1. Load the YOLO26x model inside the container"
echo "  2. Export it to OpenVINO INT8 format (~5-10 min)"
echo "  3. Save to: $MODELS_DIR/yolo26x_openvino_model/"
echo ""
echo "Expected speedup on Xeon D: 2-4x faster per scan"
echo ""
read -p "Continue? [y/N] " confirm
[[ "$confirm" =~ ^[Yy]$ ]] || { echo "Cancelled."; exit 0; }

echo ""
echo "Starting export inside container..."
echo ""

docker exec -i church-attendance python3 -u << 'PYEOF'
import sys
import os

sys.path.insert(0, '/app/backend')

from ultralytics import YOLO

model_name = os.getenv("YOLO_MODEL", "yolo26x.pt")
output_dir = "/models/yolo26x_openvino_model"

# Don't re-export if already done
if os.path.isdir(output_dir):
    print(f"OpenVINO model already exists at {output_dir}")
    print("Delete the directory and re-run to force re-export.")
    sys.exit(0)

print(f"Loading {model_name}...")
model = YOLO(model_name)

print("Exporting to OpenVINO INT8 format...")
print("(This may take several minutes on first run)")
export_path = model.export(
    format="openvino",
    int8=True,
    dynamic=False,
    simplify=True,
)
print(f"Export complete: {export_path}")

# Move/copy to /models so it persists across rebuilds
import shutil
if str(export_path) != output_dir:
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    shutil.copytree(str(export_path), output_dir)
    print(f"Copied to: {output_dir}")

print("")
print("=============================================")
print("  Done! To activate OpenVINO inference:")
print("  Add this to docker-compose.yml environment:")
print("    - YOLO_MODEL=/models/yolo26x_openvino_model")
print("  Then run: docker compose up -d")
print("  (no rebuild needed)")
print("=============================================")
PYEOF

echo ""
echo "Export finished."
echo ""
echo "Next step — add this to docker-compose.yml under 'environment:':"
echo "  - YOLO_MODEL=/models/yolo26x_openvino_model"
echo ""
echo "Then apply with:  docker compose up -d"
echo "(no rebuild needed — just an env var change)"
echo ""
