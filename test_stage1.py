"""
test_stage1.py

Automated check: run Stage 1 on the synthetic video and assert the
detected events match the known ground truth (within tolerance).

Run:
    python tests/make_synthetic_video.py   # once, to (re)generate the video
    python tests/test_stage1.py
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.pipeline import run_pipeline  # noqa: E402

VIDEO = "data/synthetic_cup_test.mp4"
BBOX = (135, 215, 50, 50)

# Ground truth from make_synthetic_video.py's PHASES
EXPECTED = [
    ("cup_moved", 2.0, 3.0),
    ("cup_returned", 5.0, 6.0),
]
TOLERANCE_SEC = 0.5


def main():
    assert os.path.exists(VIDEO), f"Run tests/make_synthetic_video.py first to create {VIDEO}"

    events = run_pipeline(
        video_path=VIDEO,
        bbox=BBOX,
        db_path="output/test_events.db",
        csv_path="output/test_events.csv",
        debug_video_path=None,
    )

    print(f"Detected {len(events)} events, expected {len(EXPECTED)}")
    assert len(events) == len(EXPECTED), (
        f"Expected {len(EXPECTED)} events, got {len(events)}: {[e.label for e in events]}"
    )

    for (exp_label, exp_start, exp_end), e in zip(EXPECTED, events):
        print(f"  checking {e.label} ~ expected {exp_label}: "
              f"{e.start_time:.2f}-{e.end_time:.2f}s vs {exp_start}-{exp_end}s")
        assert e.label == exp_label, f"Expected label {exp_label}, got {e.label}"
        assert abs(e.start_time - exp_start) <= TOLERANCE_SEC, (
            f"{exp_label} start off by too much: {e.start_time} vs {exp_start}"
        )
        assert abs(e.end_time - exp_end) <= TOLERANCE_SEC, (
            f"{exp_label} end off by too much: {e.end_time} vs {exp_end}"
        )

    print("\nALL STAGE 1 TESTS PASSED")


if __name__ == "__main__":
    main()
