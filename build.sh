#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
#  Smart build & compose script — detects CPU and GPU, builds the right
#  Docker image, and brings the stack up automatically.
#
#  Usage:
#    ./build.sh              # auto-detect hardware, build & start
#    ./build.sh --gpu        # force NVIDIA CUDA build
#    ./build.sh --rocm       # force AMD ROCm build
#    ./build.sh --cpu        # force CPU-only build
#    ./build.sh --build-only # auto-detect but don't start (just build)
#    ./build.sh --info       # show detected hardware and exit
# ─────────────────────────────────────────────────────────────────────────────
set -e

GPU_TYPE=""
BUILD_ONLY=false
INFO_ONLY=false
CPU_VENDOR=""
CPU_MODEL=""
CPU_CORES=0

# ── Parse arguments ──────────────────────────────────────────────────────────
for arg in "$@"; do
    case "$arg" in
        --gpu)        GPU_TYPE="nvidia" ;;
        --rocm)       GPU_TYPE="amd"    ;;
        --cpu)        GPU_TYPE="cpu"    ;;
        --build-only) BUILD_ONLY=true   ;;
        --info)       INFO_ONLY=true    ;;
        --help|-h)
            echo "Usage: ./build.sh [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --gpu         Force NVIDIA CUDA build"
            echo "  --rocm        Force AMD ROCm build"
            echo "  --cpu         Force CPU-only build"
            echo "  --build-only  Build image without starting the container"
            echo "  --info        Show detected hardware and exit"
            echo "  -h, --help    Show this help message"
            echo ""
            echo "With no options, auto-detects CPU/GPU and builds + starts the stack."
            exit 0
            ;;
        *)
            echo "Unknown option: $arg (try --help)"
            exit 1
            ;;
    esac
done

# ── Detect CPU ───────────────────────────────────────────────────────────────
detect_cpu() {
    if [ -f /proc/cpuinfo ]; then
        CPU_MODEL=$(grep -m1 "model name" /proc/cpuinfo 2>/dev/null | cut -d: -f2 | xargs)
        CPU_CORES=$(nproc 2>/dev/null || grep -c "^processor" /proc/cpuinfo 2>/dev/null || echo 1)

        if echo "$CPU_MODEL" | grep -qi "intel"; then
            CPU_VENDOR="intel"
        elif echo "$CPU_MODEL" | grep -qi "amd"; then
            CPU_VENDOR="amd"
        else
            CPU_VENDOR="unknown"
        fi
    elif command -v sysctl &>/dev/null; then
        # macOS / BSD
        CPU_MODEL=$(sysctl -n machdep.cpu.brand_string 2>/dev/null || echo "Unknown")
        CPU_CORES=$(sysctl -n hw.ncpu 2>/dev/null || echo 1)

        if echo "$CPU_MODEL" | grep -qi "intel"; then
            CPU_VENDOR="intel"
        elif echo "$CPU_MODEL" | grep -qi "apple"; then
            CPU_VENDOR="apple"
        else
            CPU_VENDOR="unknown"
        fi
    elif [ "$(uname -m)" = "aarch64" ] || [ "$(uname -m)" = "arm64" ]; then
        CPU_VENDOR="arm"
        CPU_MODEL="ARM $(uname -m)"
        CPU_CORES=$(nproc 2>/dev/null || echo 1)
    else
        CPU_VENDOR="unknown"
        CPU_MODEL="Unknown"
        CPU_CORES=$(nproc 2>/dev/null || echo 1)
    fi
}

# ── Detect GPU ───────────────────────────────────────────────────────────────
detect_gpu() {
    # Skip if GPU type was forced via CLI flag
    [ -n "$GPU_TYPE" ] && return

    # ── NVIDIA ───────────────────────────────────────────────────────────
    if command -v nvidia-smi &>/dev/null && nvidia-smi &>/dev/null 2>&1; then
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits 2>/dev/null | head -1)
        echo "  GPU:    $GPU_NAME (NVIDIA)"

        if command -v nvidia-ctk &>/dev/null || docker info 2>/dev/null | grep -qi nvidia; then
            GPU_TYPE="nvidia"
        else
            echo ""
            echo "  WARNING: NVIDIA GPU detected but nvidia-container-toolkit not found."
            echo "  Install it to enable GPU acceleration:"
            echo "    https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            echo ""
        fi

    # ── AMD ROCm ─────────────────────────────────────────────────────────
    elif command -v rocm-smi &>/dev/null && rocm-smi &>/dev/null 2>&1; then
        GPU_NAME=$(rocm-smi --showproductname 2>/dev/null | grep -i "card\|gpu" | head -1 || echo "AMD GPU")
        echo "  GPU:    $GPU_NAME (AMD ROCm)"

        if [ -e /dev/kfd ] && [ -e /dev/dri ]; then
            GPU_TYPE="amd"
        else
            echo ""
            echo "  WARNING: AMD GPU detected but /dev/kfd or /dev/dri not accessible."
            echo "  Ensure the amdgpu kernel driver is loaded and your user is in the"
            echo "  'render' and 'video' groups."
            echo ""
        fi

    # ── AMD GPU without ROCm tools ───────────────────────────────────────
    elif command -v lspci &>/dev/null && \
         lspci 2>/dev/null | grep -qiE "(AMD|ATI).*(Radeon|RX |Vega|Navi|RDNA)|(Radeon).*(AMD|ATI)"; then
        GPU_NAME=$(lspci 2>/dev/null | grep -iE "(AMD|ATI).*(Radeon|RX|Vega|Navi)|(Radeon).*(AMD|ATI)" | head -1)
        echo "  GPU:    $GPU_NAME"
        echo ""
        echo "  WARNING: AMD GPU found but ROCm tools (rocm-smi) not installed."
        echo "  Install ROCm to enable GPU acceleration:"
        echo "    https://rocm.docs.amd.com/en/latest/deploy/linux/quick_start.html"
        echo ""
    else
        echo "  GPU:    None detected"
    fi

    # ── Fall back to CPU ─────────────────────────────────────────────────
    if [ -z "$GPU_TYPE" ]; then
        GPU_TYPE="cpu"
    fi
}

# ── Compute recommended TILE_BATCH_SIZE based on hardware ────────────────────
recommend_batch_size() {
    local batch=4
    if [ "$GPU_TYPE" != "cpu" ]; then
        batch=16
    elif [ "$CPU_CORES" -ge 16 ]; then
        batch=12
    elif [ "$CPU_CORES" -ge 8 ]; then
        batch=8
    fi
    echo "$batch"
}

# ── Show hardware summary ───────────────────────────────────────────────────
show_hardware_info() {
    echo "============================================="
    echo "  Hardware Detection"
    echo "============================================="
    echo "  CPU:    $CPU_MODEL"
    echo "  Cores:  $CPU_CORES"
    echo "  Arch:   $(uname -m)"
}

# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

detect_cpu
show_hardware_info
detect_gpu

BATCH_SIZE=$(recommend_batch_size)

echo "---------------------------------------------"
echo "  Build:  $GPU_TYPE  |  Batch size: $BATCH_SIZE"
echo "============================================="
echo ""

# ── Intel CPU hint ───────────────────────────────────────────────────────────
if [ "$CPU_VENDOR" = "intel" ] && [ "$GPU_TYPE" = "cpu" ]; then
    echo "TIP: Intel CPU detected. After building, run ./export_openvino.sh"
    echo "     for 2-4x faster inference via OpenVINO INT8."
    echo ""
fi

# ── ARM / Apple Silicon warning ──────────────────────────────────────────────
if [ "$CPU_VENDOR" = "arm" ] || [ "$CPU_VENDOR" = "apple" ]; then
    echo "NOTE: ARM architecture detected. GPU passthrough is not supported."
    echo "      Using CPU-only build."
    GPU_TYPE="cpu"
    echo ""
fi

# ── Info-only mode: just show detection and exit ─────────────────────────────
if $INFO_ONLY; then
    exit 0
fi

# ── Select compose file ─────────────────────────────────────────────────────
COMPOSE_FILE="docker-compose.yml"
DOCKERFILE_LABEL="CPU-only"

case "$GPU_TYPE" in
    nvidia)
        COMPOSE_FILE="docker-compose.gpu.yml"
        DOCKERFILE_LABEL="NVIDIA CUDA (OpenCV built from source)"
        ;;
    amd)
        COMPOSE_FILE="docker-compose.rocm.yml"
        DOCKERFILE_LABEL="AMD ROCm (OpenCV built from source)"
        ;;
esac

echo "Building:  $DOCKERFILE_LABEL"
echo "Compose:   $COMPOSE_FILE"
echo ""

# ── Export batch size so compose can pick it up ──────────────────────────────
export TILE_BATCH_SIZE=$BATCH_SIZE

# ── Build ────────────────────────────────────────────────────────────────────
docker compose -f "$COMPOSE_FILE" build

echo ""

if $BUILD_ONLY; then
    echo "Build complete. Start with:"
    echo "  docker compose -f $COMPOSE_FILE up -d"
    exit 0
fi

# ── Bring up the stack ───────────────────────────────────────────────────────
echo "Starting stack..."
docker compose -f "$COMPOSE_FILE" up -d

echo ""
echo "============================================="
echo "  Church Counter is running!"
echo "  Open:  http://$(hostname -I 2>/dev/null | awk '{print $1}' || echo 'localhost'):80"
echo "============================================="
