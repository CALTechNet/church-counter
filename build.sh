#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Smart build script — auto-detects GPU and builds the right Docker image.
#
#  Usage:
#    ./build.sh          # auto-detect GPU
#    ./build.sh --gpu    # force GPU build
#    ./build.sh --cpu    # force CPU build
# ─────────────────────────────────────────────────────────────────────────────
set -e

USE_GPU=""

# Parse arguments
case "${1:-}" in
    --gpu)  USE_GPU="yes" ;;
    --cpu)  USE_GPU="no"  ;;
    *)
        # Auto-detect: check for NVIDIA GPU + nvidia-container-toolkit
        if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
            echo "Detected GPU: $GPU_NAME"

            if command -v nvidia-ctk &>/dev/null || docker info 2>/dev/null | grep -qi nvidia; then
                USE_GPU="yes"
            else
                echo ""
                echo "WARNING: GPU detected but nvidia-container-toolkit not found."
                echo "Install it to enable GPU acceleration:"
                echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
                echo ""
                echo "Falling back to CPU build."
                USE_GPU="no"
            fi
        else
            echo "No NVIDIA GPU detected — using CPU build."
            USE_GPU="no"
        fi
        ;;
esac

echo ""

if [ "$USE_GPU" = "yes" ]; then
    echo "============================================="
    echo "  Building with GPU (CUDA) support"
    echo "============================================="
    echo ""
    docker compose -f docker-compose.gpu.yml build
    echo ""
    echo "Done! Start with:  docker compose -f docker-compose.gpu.yml up -d"
else
    echo "============================================="
    echo "  Building for CPU only"
    echo "============================================="
    echo ""
    docker compose build
    echo ""
    echo "Done! Start with:  docker compose up -d"
fi
