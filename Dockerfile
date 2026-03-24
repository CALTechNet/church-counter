# ── Stage 1: Build React frontend ────────────────────────────────────────────
FROM node:20-alpine AS frontend
WORKDIR /app
COPY frontend/package*.json ./
RUN npm ci --silent
COPY frontend/ ./
RUN npm run build

# ── Stage 2: Python backend + serve ──────────────────────────────────────────
FROM python:3.11-slim
WORKDIR /app

# System libs for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libsm6 libxext6 libxrender1 libgl1 libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Upgrade ultralytics to fix PyTorch 2.6 weights_only compatibility
RUN pip install --upgrade ultralytics

# Backend source
COPY backend/ ./backend/

# Built frontend from stage 1
COPY --from=frontend /app/build /frontend/build

# Runtime directories — actual data lives in Docker volumes mounted here
RUN mkdir -p /opt/church-counter/data \
             /opt/church-counter/config \
             /opt/church-counter/models

# Default env paths (override via docker-compose environment or -e flags)
ENV DATA_DIR=/opt/church-counter/data \
    CONFIG_DIR=/opt/church-counter/config \
    MODELS_DIR=/opt/church-counter/models

EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
