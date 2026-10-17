"""
make_synthetic_video.py

Generates a short synthetic MP4 with a fixed camera, a static "table"
background, and a colored circle (standing in for the cup) that:
  1. sits still at its ORIGINAL position,
  2. moves to a new position and sits still there,
  3. moves back to the ORIGINAL position and sits still again.

This gives us known, ground-truth event timestamps to sanity-check the
Stage 1 pipeline end-to-end before you point it at real footage, per the
spec's requirement for a simple test at each stage.

Run:
    python tests/make_synthetic_video.py
Produces:
    data/synthetic_cup_test.mp4
    (prints the ground-truth bbox + expected event windows)
"""

import cv2
import numpy as np

OUT_PATH = "data/synthetic_cup_test.mp4"
FPS = 30
W, H = 640, 480
CUP_RADIUS = 25
CUP_COLOR = (40, 90, 200)  # BGR - a reddish-orange "cup"

ORIGINAL_POS = (160, 240)
MOVED_POS = (460, 240)

# Timeline (seconds): still -> move -> still -> move back -> still
PHASES = [
    ("still_original", 2.0),
    ("moving_out", 1.0),
    ("still_moved", 2.0),
    ("moving_back", 1.0),
    ("still_original_2", 2.0),
]


def lerp(a, b, t):
    return a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t


def main():
    import os
    os.makedirs("data", exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(OUT_PATH, fourcc, FPS, (W, H))

    t = 0.0
    phase_log = []
    for phase_name, duration in PHASES:
        n_frames = int(duration * FPS)
        for i in range(n_frames):
            frac = i / max(n_frames - 1, 1)
            if phase_name == "still_original" or phase_name == "still_original_2":
                pos = ORIGINAL_POS
            elif phase_name == "still_moved":
                pos = MOVED_POS
            elif phase_name == "moving_out":
                pos = lerp(ORIGINAL_POS, MOVED_POS, frac)
            elif phase_name == "moving_back":
                pos = lerp(MOVED_POS, ORIGINAL_POS, frac)

            frame = np.full((H, W, 3), (90, 140, 150), dtype=np.uint8)  # flat "table" bg
            cv2.circle(frame, (int(pos[0]), int(pos[1])), CUP_RADIUS, CUP_COLOR, -1)
            # mild per-pixel noise so it's not a trivially perfect signal
            noise = np.random.randint(-4, 5, frame.shape, dtype=np.int16)
            frame = np.clip(frame.astype(np.int16) + noise, 0, 255).astype(np.uint8)

            writer.write(frame)
            t += 1.0 / FPS
        phase_log.append((phase_name, t - duration, t))

    writer.release()

    bbox = (
        ORIGINAL_POS[0] - CUP_RADIUS, ORIGINAL_POS[1] - CUP_RADIUS,
        CUP_RADIUS * 2, CUP_RADIUS * 2,
    )
    print(f"Wrote {OUT_PATH}")
    print(f"Initial bbox for --bbox argument: {bbox[0]},{bbox[1]},{bbox[2]},{bbox[3]}")
    print("\nGround-truth phase windows (seconds):")
    for name, start, end in phase_log:
        print(f"  {start:5.2f} - {end:5.2f} | {name}")
    print("\nExpected detected events (approximately):")
    print("  one 'cup_moved'   during the moving_out window")
    print("  one 'cup_returned' during the moving_back window")


if __name__ == "__main__":
    main()
