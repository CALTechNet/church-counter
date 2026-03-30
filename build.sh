#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Smart build script — auto-detects GPU and builds the right Docker image.
#
#  Usage:
#    ./build.sh          # auto-detect GPU
#    ./build.sh --gpu    # force NVIDIA CUDA build
#    ./build.sh --rocm   # force AMD ROCm build
#    ./build.sh --cpu    # force CPU build
# ─────────────────────────────────────────────────────────────────────────────
set -e

GPU_TYPE=""

# ── Parse arguments ───────────────────────────────────────────────────────────
case "${1:-}" in
    --gpu)   GPU_TYPE="nvidia" ;;
    --rocm)  GPU_TYPE="amd"    ;;
    --cpu)   GPU_TYPE="cpu"    ;;
    *)
        # ── Auto-detect NVIDIA ───────────────────────────────────────────────
        if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
            echo "Detected NVIDIA GPU: $GPU_NAME"

            if command -v nvidia-ctk &>/dev/null || docker info 2>/dev/null | grep -qi nvidia; then
                GPU_TYPE="nvidia"
            else
                echo ""
                echo "WARNING: NVIDIA GPU detected but nvidia-container-toolkit not found."
                echo "Install it to enable GPU acceleration:"
                echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
                echo ""
            fi

        # ── Auto-detect AMD ROCm ─────────────────────────────────────────────
        elif command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
            GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -i "card\|gpu" | head -1 || echo "AMD GPU")
            echo "Detected AMD GPU (ROCm): $GPU_NAME"

            if [ -e /dev/kfd ] && [ -e /dev/dri ]; then
                GPU_TYPE="amd"
            else
                echo ""
                echo "WARNING: AMD GPU detected but /dev/kfd or /dev/dri not accessible."
                echo "Ensure the amdgpu kernel driver is loaded and your user is in the"
                echo "'render' and 'video' groups."
                echo ""
            fi

        # ── AMD GPU present but ROCm tools not installed ─────────────────────
        elif command -v lspci &>/dev/null && \
             lspci 2>/dev/null | grep -qiE "(AMD|ATI).*(Radeon|RX |Vega|Navi|RDNA)|(Radeon).*(AMD|ATI)"; then
            GPU_NAME=$(lspci 2>/dev/null | grep -iE "(AMD|ATI).*(Radeon|RX|Vega|Navi)|(Radeon).*(AMD|ATI)" | head -1)
            echo "Detected AMD GPU: $GPU_NAME"
            echo ""
            echo "WARNING: AMD GPU found but ROCm tools (rocm-smi) not installed."
            echo "Install ROCm to enable GPU acceleration:"
            echo "  https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
            echo ""
        fi

        # ── Fall back to CPU if no usable GPU found ───────────────────────────
        if [ -z "$GPU_TYPE" ]; then
            echo "No GPU acceleration available — using CPU build."
            GPU_TYPE="cpu"
        fi
        ;;
esac

echo ""

# ── Launch the appropriate build ──────────────────────────────────────────────
case "$GPU_TYPE" in
    nvidia)
        echo "============================================="
        echo "  Building with NVIDIA CUDA support"
        echo "  (OpenCV built from source with CUDA)"
        echo "  First build: ~20-40 min  |  Cached: fast"
        echo "============================================="
        echo ""
        docker compose -f docker-compose.gpu.yml build
        echo ""
        echo "Done! Start with:  docker compose -f docker-compose.gpu.yml up -d"
        ;;
    amd)
        echo "============================================="
        echo "  Building with AMD ROCm support"
        echo "  (OpenCV built from source with OpenCL)"
        echo "  First build: ~15-25 min  |  Cached: fast"
        echo "============================================="
        echo ""
        docker compose -f docker-compose.rocm.yml build
        echo ""
        echo "Done! Start with:  docker compose -f docker-compose.rocm.yml up -d"
        ;;
    cpu)
        echo "============================================="
        echo "  Building for CPU only"
        echo "============================================="
        echo ""
        docker compose build
        echo ""
        echo "Done! Start with:  docker compose up -d"
        ;;
esac
