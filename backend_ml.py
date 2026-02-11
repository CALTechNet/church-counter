"""Backend ML pipeline for PTZ seat occupancy counting.

This module provides an end-to-end workflow:
1. Move a PTZ camera across configured presets.
2. Capture snapshots from each preset.
3. Run person detection on each snapshot.
4. Count occupied seats by checking whether detected people fall inside seat polygons.

Expected seat map format (JSON):
{
  "presets": [
    {
      "token": "1",
      "name": "left_section",
      "snapshot_url": "http://CAMERA_IP/snapshot.jpg",
      "seats": [
        {"id": "L1-01", "polygon": [[x1, y1], [x2, y2], [x3, y3], [x4, y4]]}
      ]
    }
  ]
}
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import requests
from onvif import ONVIFCamera
from ultralytics import YOLO


@dataclass
class CameraConfig:
    host: str
    port: int
    username: str
    password: str
    move_wait_seconds: float = 2.0
    request_timeout_seconds: float = 8.0


class PTZCameraController:
    def __init__(self, config: CameraConfig):
        self.config = config
        self.camera = ONVIFCamera(
            config.host,
            config.port,
            config.username,
            config.password,
        )
        self.media = self.camera.create_media_service()
        self.ptz = self.camera.create_ptz_service()
        self.profile = self.media.GetProfiles()[0]

    def goto_preset(self, preset_token: str) -> None:
        request = self.ptz.create_type("GotoPreset")
        request.ProfileToken = self.profile.token
        request.PresetToken = preset_token
        self.ptz.GotoPreset(request)
        time.sleep(self.config.move_wait_seconds)

    def capture_snapshot(self, snapshot_url: str) -> np.ndarray:
        response = requests.get(
            snapshot_url,
            auth=(self.config.username, self.config.password),
            timeout=self.config.request_timeout_seconds,
        )
        response.raise_for_status()

        image_array = np.frombuffer(response.content, dtype=np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
        if frame is None:
            raise ValueError("Unable to decode snapshot image from camera response.")
        return frame


class SeatOccupancyCounter:
    def __init__(self, model_name: str = "yolov8n.pt", conf_threshold: float = 0.35):
        self.model = YOLO(model_name)
        self.conf_threshold = conf_threshold

    def detect_people(self, frame: np.ndarray) -> list[tuple[int, int, int, int]]:
        results = self.model.predict(frame, conf=self.conf_threshold, verbose=False)
        person_boxes: list[tuple[int, int, int, int]] = []

        for result in results:
            for box in result.boxes:
                class_id = int(box.cls.item())
                if class_id != 0:  # YOLO class 0 = person
                    continue
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                person_boxes.append((int(x1), int(y1), int(x2), int(y2)))

        return person_boxes

    @staticmethod
    def _centroid(box: tuple[int, int, int, int]) -> tuple[int, int]:
        x1, y1, x2, y2 = box
        return ((x1 + x2) // 2, (y1 + y2) // 2)

    def count_occupied(self, seats: list[dict[str, Any]], person_boxes: list[tuple[int, int, int, int]]) -> dict[str, Any]:
        occupied_ids: list[str] = []
        centroids = [self._centroid(box) for box in person_boxes]

        for seat in seats:
            seat_id = seat["id"]
            polygon = np.array(seat["polygon"], dtype=np.int32)
            is_occupied = any(cv2.pointPolygonTest(polygon, point, False) >= 0 for point in centroids)
            if is_occupied:
                occupied_ids.append(seat_id)

        return {
            "occupied": len(occupied_ids),
            "total": len(seats),
            "occupied_ids": occupied_ids,
        }


def run_headcount(camera_cfg: CameraConfig, seat_map_path: str) -> dict[str, Any]:
    controller = PTZCameraController(camera_cfg)
    counter = SeatOccupancyCounter()

    seat_map = json.loads(Path(seat_map_path).read_text())
    summaries: list[dict[str, Any]] = []

    for preset in seat_map["presets"]:
        controller.goto_preset(preset["token"])
        frame = controller.capture_snapshot(preset["snapshot_url"])
        people = counter.detect_people(frame)
        summary = counter.count_occupied(preset["seats"], people)
        summary["preset"] = preset.get("name", preset["token"])
        summaries.append(summary)

    total_occupied = sum(item["occupied"] for item in summaries)
    total_seats = sum(item["total"] for item in summaries)

    return {
        "total_occupied": total_occupied,
        "total_seats": total_seats,
        "occupancy_rate": round(total_occupied / total_seats, 4) if total_seats else 0.0,
        "sections": summaries,
    }


if __name__ == "__main__":
    # Example runtime configuration. Move these values to environment variables in production.
    config = CameraConfig(
        host="192.168.1.10",
        port=80,
        username="admin",
        password="password",
    )

    result = run_headcount(config, "seat_map.json")
    print(json.dumps(result, indent=2))
