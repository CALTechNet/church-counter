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

# Python deps — single layer, no cache, strip __pycache__/.pyc to save ~200 MB
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir --upgrade ultralytics && \
    find /usr/local/lib/python* -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null; true

# Backend source
COPY backend/ ./backend/

# Built frontend from stage 1
COPY --from=frontend /app/build /frontend/build

# Runtime directories
RUN mkdir -p /data /config

# Ultralytics settings directory — writable location inside the container
ENV YOLO_CONFIG_DIR=/tmp/Ultralytics

EXPOSE 8000

WORKDIR /app/backend
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
