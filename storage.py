"""
storage.py - saves every completed test in a small SQLite file (tremor.db).

No names or personal data are stored: only the measurement, a few settings
and the recording CSV (wrist coordinates).
"""
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

DB_PATH = Path(__file__).resolve().parent / "tremor.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS records (
    id           INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at   TEXT NOT NULL,
    condition    TEXT,
    hand         TEXT,
    status       TEXT,
    frequency_hz REAL,
    rms_cm       REAL,
    duration_s   REAL,
    result_title TEXT,
    result_json  TEXT NOT NULL,
    meta_json    TEXT,
    csv_text     TEXT
)
"""


def _conn():
    c = sqlite3.connect(DB_PATH)
    c.row_factory = sqlite3.Row
    c.execute(SCHEMA)
    return c


def save(result, meta, csv_text):
    created = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with _conn() as c:
        cur = c.execute(
            "INSERT INTO records (created_at, condition, hand, status, frequency_hz, rms_cm, duration_s,"
            " result_title, result_json, meta_json, csv_text) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (created, result.get("condition"), (meta or {}).get("hand"), result.get("status"),
             result.get("frequency_hz"), result.get("rms_cm"), result.get("duration_s"),
             result.get("result_title"), json.dumps(result), json.dumps(meta or {}), csv_text))
        return cur.lastrowid, created


def _summary(r):
    return {k: r[k] for k in ("id", "created_at", "condition", "hand", "status",
                              "frequency_hz", "rms_cm", "duration_s", "result_title")}


def list_records(limit=200):
    with _conn() as c:
        rows = c.execute("SELECT * FROM records ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        total = c.execute("SELECT COUNT(*) FROM records").fetchone()[0]
    return {"total": total, "records": [_summary(r) for r in rows]}


def get(record_id):
    with _conn() as c:
        r = c.execute("SELECT * FROM records WHERE id=?", (record_id,)).fetchone()
    if r is None:
        return None
    out = _summary(r)
    out["result"] = json.loads(r["result_json"])
    out["meta"] = json.loads(r["meta_json"] or "{}")
    return out


def get_csv(record_id):
    with _conn() as c:
        r = c.execute("SELECT csv_text, condition, created_at FROM records WHERE id=?", (record_id,)).fetchone()
    return None if r is None else (r["csv_text"] or "", r["condition"], r["created_at"])


def delete(record_id):
    with _conn() as c:
        return c.execute("DELETE FROM records WHERE id=?", (record_id,)).rowcount > 0