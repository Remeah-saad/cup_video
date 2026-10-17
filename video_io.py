"""
video_io.py

Stage 1 responsibility: open an MP4, and yield (frame_index, timestamp_sec,
frame) tuples with timestamps that are actually derived from the video's
own frame position, not assumed from a constant FPS multiplied blindly
(cv2.CAP_PROP_POS_MSEC is used when available, falling back to
frame_index / fps if a container doesn't expose per-frame timestamps).
"""

from dataclasses import dataclass
from typing import Iterator

import cv2


@dataclass
class Frame:
    index: int
    timestamp_sec: float
    image: "cv2.typing.MatLike"


class VideoReader:
    """Thin wrapper around cv2.VideoCapture with honest timestamps."""

    def __init__(self, video_path: str, fps_override: float | None = None):
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {video_path}")

        reported_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps_override or (reported_fps if reported_fps > 1e-3 else 30.0)
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def frames(self) -> Iterator[Frame]:
        idx = 0
        while True:
            ok, img = self.cap.read()
            if not ok:
                break

            # Prefer the container's own per-frame timestamp when it's
            # present and sane; otherwise derive it from frame index / fps.
            ts_msec = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            if ts_msec and ts_msec > 0:
                timestamp = ts_msec / 1000.0
            else:
                timestamp = idx / self.fps

            yield Frame(index=idx, timestamp_sec=timestamp, image=img)
            idx += 1

    def release(self):
        self.cap.release()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.release()


def format_timestamp(seconds: float) -> str:
    """HH:MM:SS.ss for display, matching the examples in the spec."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:05.2f}"
