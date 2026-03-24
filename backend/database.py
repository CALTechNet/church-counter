import os
import sqlite3
import json
from datetime import datetime
from pathlib import Path

_DATA_DIR = Path(os.environ.get("DATA_DIR", "/opt/church-counter/data"))
DB_PATH = Path(os.environ.get("DB_PATH", str(_DATA_DIR / "church.db")))


def init_db():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS scans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            service_type TEXT,
            total_count INTEGER,
            occupied_seats TEXT,
            stitched_image TEXT,
            notes TEXT,
            manual_add INTEGER DEFAULT 0
        )
    """)
    # Migrations: add columns if they don't exist yet
    for migration in [
        "ALTER TABLE scans ADD COLUMN manual_add INTEGER DEFAULT 0",
        "ALTER TABLE scans ADD COLUMN archived  INTEGER DEFAULT 0",
        "ALTER TABLE scans ADD COLUMN raw_image  TEXT",
    ]:
        try:
            c.execute(migration)
        except Exception:
            pass  # column already exists
    c.execute("""
        CREATE TABLE IF NOT EXISTS calibration (
            seat_id TEXT PRIMARY KEY,
            svg_x REAL,
            svg_y REAL,
            photo_x REAL,
            photo_y REAL,
            updated_at TEXT
        )
    """)
    c.execute("""
        CREATE TABLE IF NOT EXISTS config (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()


def get_config(key: str, default=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT value FROM config WHERE key = ?", (key,))
    row = c.fetchone()
    conn.close()
    if not row:
        return default
    return json.loads(row[0])


def set_config(key: str, value):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO config (key, value, updated_at) VALUES (?, ?, ?)",
        (key, json.dumps(value), datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def save_scan(timestamp, service_type, total_count, occupied_seats, stitched_image_b64, notes=None, raw_image_b64=None):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT INTO scans (timestamp, service_type, total_count, occupied_seats, stitched_image, notes, raw_image) VALUES (?,?,?,?,?,?,?)",
        (timestamp, service_type, total_count, json.dumps(occupied_seats), stitched_image_b64, notes, raw_image_b64),
    )
    scan_id = c.lastrowid
    conn.commit()
    conn.close()
    return scan_id


def get_all_scans(include_archived: bool = False):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    where = "" if include_archived else "WHERE COALESCE(archived, 0) = 0"
    c.execute(
        f"SELECT id, timestamp, service_type, total_count, occupied_seats, notes, "
        f"COALESCE(manual_add, 0), COALESCE(archived, 0) FROM scans {where} ORDER BY timestamp ASC"
    )
    rows = c.fetchall()
    conn.close()
    return [
        {
            "id":            r[0],
            "timestamp":     r[1],
            "service_type":  r[2],
            "count":         r[3],
            "occupied_seats": json.loads(r[4]) if r[4] else [],
            "notes":         r[5],
            "manual_add":    r[6],
            "total":         (r[3] or 0) + (r[6] or 0),
            "archived":      bool(r[7]),
        }
        for r in rows
    ]


def get_latest_scan():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT id, timestamp, service_type, total_count, occupied_seats, stitched_image, notes FROM scans ORDER BY timestamp DESC LIMIT 1"
    )
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {
        "id": row[0],
        "timestamp": row[1],
        "service_type": row[2],
        "count": row[3],
        "occupied_seats": json.loads(row[4]) if row[4] else [],
        "stitched_image": row[5],
        "notes": row[6],
    }


def get_calibration():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT seat_id, svg_x, svg_y, photo_x, photo_y FROM calibration")
    rows = c.fetchall()
    conn.close()
    return {r[0]: {"svg_x": r[1], "svg_y": r[2], "photo_x": r[3], "photo_y": r[4]} for r in rows}


def save_calibration_point(seat_id, svg_x, svg_y, photo_x, photo_y):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "INSERT OR REPLACE INTO calibration (seat_id, svg_x, svg_y, photo_x, photo_y, updated_at) VALUES (?,?,?,?,?,?)",
        (seat_id, svg_x, svg_y, photo_x, photo_y, datetime.now().isoformat()),
    )
    conn.commit()
    conn.close()


def delete_calibration_point(seat_id):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM calibration WHERE seat_id = ?", (seat_id,))
    conn.commit()
    conn.close()


def clear_calibration():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM calibration")
    conn.commit()
    conn.close()

def get_scan_image(scan_id: int):
    """Fetch the annotated and raw images for a single scan by ID."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT stitched_image, raw_image FROM scans WHERE id = ?", (scan_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        return None
    return {"annotated": row[0], "raw": row[1]}


def update_scan(scan_id: int, notes: str = None, manual_add: int = None,
                service_type: str = None, archived: bool = None):
    """Update notes, manual_add, service_type, and/or archived for a scan."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    if notes is not None:
        c.execute("UPDATE scans SET notes = ? WHERE id = ?", (notes, scan_id))
    if manual_add is not None:
        c.execute("UPDATE scans SET manual_add = ? WHERE id = ?", (manual_add, scan_id))
    if service_type is not None:
        c.execute("UPDATE scans SET service_type = ? WHERE id = ?", (service_type, scan_id))
    if archived is not None:
        c.execute("UPDATE scans SET archived = ? WHERE id = ?", (1 if archived else 0, scan_id))
    conn.commit()
    conn.close()
