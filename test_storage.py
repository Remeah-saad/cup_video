"""Database idempotency regression test. Run: python -m unittest tests.test_storage"""
import tempfile
import unittest
from pathlib import Path
from src.db import EventStore
from src.events import Event

class StorageTests(unittest.TestCase):
    def test_reprocessing_does_not_duplicate(self):
        e = Event("cup_moved", 1, 10, 0.1, 1.0, (1, 1), (2, 2), 0.9, True)
        with tempfile.TemporaryDirectory() as d:
            store = EventStore(str(Path(d) / "events.db"))
            store.save_events("clip.mp4", [e])
            store.save_events("clip.mp4", [e])
            self.assertEqual(len(store.all_events("clip.mp4")), 1)
            store.close()

if __name__ == "__main__":
    unittest.main()
