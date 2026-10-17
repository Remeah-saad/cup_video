"""
config.py

All the "magic numbers" for Stage 1 live here, in one place, so you can
tune them for your own camera/lighting without hunting through the code.

These are starting points, not laws of physics - expect to adjust them
after looking at a debug run on your own footage (see README Stage 1,
step "Tuning").
"""

from dataclasses import dataclass


@dataclass
class TrackerConfig:
    # --- Motion thresholds (in pixels, measured on the ORIGINAL frame
    #     resolution - rescale these if you resize frames before tracking) ---

    # How far the cup's centroid must move from its position in the
    # previous "settled" frame before we call it MOVING rather than
    # jitter/noise.
    movement_threshold_px: float = 8.0

    # How many consecutive frames the cup must be below the movement
    # threshold before we consider it to have come to rest again
    # (debounces single-frame tracker jitter).
    settle_frames: int = 6

    # How close (in pixels) the cup's centroid must be to its ORIGINAL
    # (initial) position to count as "returned".
    return_radius_px: float = 20.0

    # How close the cup's centroid must be to the position it was
    # picked up from (not the ORIGINAL position) to still count as
    # "hasn't really moved yet" - helps ignore tiny tracker drift.
    noise_floor_px: float = 3.0

    # --- Tracker recovery ---
    # Consecutive tracker-confidence failures allowed before we declare
    # tracking lost and attempt to re-detect the cup via color matching.
    max_track_failures: int = 10

    # --- Event bookkeeping ---
    # Minimum duration (seconds) for an event to be recorded - filters
    # out single-frame glitches from being logged as events.
    min_event_duration_sec: float = 0.15


@dataclass
class VideoConfig:
    # If the video file's metadata FPS is wrong/missing, you can override
    # it here. None = trust cv2's reported FPS.
    fps_override: float | None = None

    # Downscale factor applied before tracking, purely for speed
    # (tracking math below is done in ORIGINAL-resolution pixels, so this
    # only affects runtime, not the reported coordinates/timestamps).
    processing_scale: float = 1.0
