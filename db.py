"""
db.py

SQLite storage for detected events (spec section 10 - "Video Event Search
Engine" needs something to search over). Stage 1 only writes to it; the
rule-based natural-language search on top of this table comes in Stage 4.
"""

import sqlite3
from pathlib import Path

from .events import Event

SCHEMA = """
CREATE TABLE IF NOT EXISTS events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    video_filename TEXT NOT NULL,
    label TEXT NOT NULL,
    start_frame INTEGER NOT NULL,
    end_frame INTEGER NOT NULL,
    start_time REAL NOT NULL,
    end_time REAL NOT NULL,
    duration REAL NOT NULL,
    start_pos_x REAL,
    start_pos_y REAL,
    end_pos_x REAL,
    end_pos_y REAL,
    confidence REAL,
    evidence_based INTEGER NOT NULL,
    notes TEXT
);
"""


class EventStore:
    def __init__(self, db_path: str):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(db_path)
        self.conn.execute(SCHEMA)
        self.conn.commit()

    def save_events(self, video_filename: str, events: list[Event]) -> None:
        rows = [
            (
                video_filename,
                e.label,
                e.start_frame,
                e.end_frame,
                e.start_time,
                e.end_time,
                e.duration,
                e.start_pos[0], e.start_pos[1],
                e.end_pos[0], e.end_pos[1],
                e.confidence,
                int(e.evidence_based),
                e.notes,
            )
            for e in events
        ]
        # Reprocessing a video replaces its old events instead of duplicating them.
        with self.conn:
            self.conn.execute("DELETE FROM events WHERE video_filename = ?", (video_filename,))
            self.conn.executemany(
            """INSERT INTO events (
                video_filename, label, start_frame, end_frame,
                start_time, end_time, duration,
                start_pos_x, start_pos_y, end_pos_x, end_pos_y,
                confidence, evidence_based, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            rows,
        )
        # The transaction commits atomically on success.

    def all_events(self, video_filename: str | None = None) -> list[sqlite3.Row]:
        self.conn.row_factory = sqlite3.Row
        cur = self.conn.cursor()
        if video_filename:
            cur.execute(
                "SELECT * FROM events WHERE video_filename = ? ORDER BY start_time",
                (video_filename,),
            )
        else:
            cur.execute("SELECT * FROM events ORDER BY video_filename, start_time")
        return cur.fetchall()

    def close(self):
        self.conn.close()
