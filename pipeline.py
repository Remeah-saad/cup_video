"""
pipeline.py

Stage 1 end-to-end pipeline:

    video file --> VideoReader --> CupTracker --> EventDetector --> EventStore
                                                               \\--> CSV export
                                                               \\--> annotated debug video (optional)

Run as a script:
    python -m src.pipeline --video data/my_video.mp4 --bbox 310,220,60,80

If you omit --bbox and run in an environment with a display, it will pop
up the first frame for you to drag a box around the cup (cv2.selectROI).
"""

import argparse
import csv
from pathlib import Path

import cv2

from .config import TrackerConfig
from .db import EventStore
from .events import EventDetector
from .tracker import CupTracker
from .video_io import VideoReader, format_timestamp
from .report import create_event_report
from .annotate import create_annotated_video
from .presentation import display_label, describe_event


def run_pipeline(
    video_path: str,
    bbox: tuple[int, int, int, int] | None = None,
    db_path: str = "output/events.db",
    csv_path: str | None = None,
    debug_video_path: str | None = None,
    config: TrackerConfig | None = None,
    pdf_path: str | None = "output/event_report.pdf",
    annotated_video_path: str | None = "output/annotated_video.mp4",
):
    config = config or TrackerConfig()
    video_filename = Path(video_path).name

    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    if debug_video_path:
        Path(debug_video_path).parent.mkdir(parents=True, exist_ok=True)
    reader = VideoReader(video_path)
    tracker = CupTracker()
    detector = EventDetector(config, fps=reader.fps)

    writer = None
    if debug_video_path:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(debug_video_path, fourcc, reader.fps, (reader.width, reader.height))

    frame_iter = reader.frames()
    first = next(frame_iter)

    if bbox is None:
        h, w = first.image.shape[:2]
        scale = min(900 / w, 600 / h, 1.0)
        preview = cv2.resize(first.image, None, fx=scale, fy=scale,
                             interpolation=cv2.INTER_AREA)
        try:
            selected = cv2.selectROI("Select cup", preview, False, False)
        finally:
            cv2.destroyAllWindows()
        if selected[2] <= 0 or selected[3] <= 0:
            reader.release()
            if writer:
                writer.release()
            raise ValueError("No cup selected; run again and draw a rectangle.")
        bbox = tuple(round(v / scale) for v in selected)

    tracker.init(first.image, bbox)
    origin_centroid = tracker._bbox_centroid(bbox)  # noqa: SLF001 - internal helper reuse
    detector.set_origin(origin_centroid)
    tracks = [(bbox, "initial")]

    if writer:
        _draw_debug(first.image, bbox, "TRACKING", writer)

    for frame in frame_iter:
        result = tracker.update(frame.image, config)
        tracks.append((result.bbox if result.success else None, result.method))
        detector.update(frame.index, frame.timestamp_sec, result.centroid, result.success, result.method)

        if writer:
            _draw_debug(frame.image, result.bbox, "TRACKING", writer, method=result.method)

    detector.finalize()
    reader.release()
    if writer:
        writer.release()

    if annotated_video_path:
        count = create_annotated_video(video_path, annotated_video_path, reader.fps,
                                       len(tracks), tracks, detector.events, origin_centroid, detector.states)
        print(f"Annotated video saved: {annotated_video_path} ({count} frames)")

    # --- Persist results ---
    store = EventStore(db_path)
    store.save_events(video_filename, detector.events)
    store.close()

    if csv_path:
        _export_csv(csv_path, video_filename, detector.events)

    if pdf_path:
        duration_sec = max(0.0, (frame.index if "frame" in locals() else first.index) / reader.fps)
        create_event_report(pdf_path, video_filename, duration_sec, detector.events)
        print(f"PDF report saved: {pdf_path}")

    return detector.events


def _draw_debug(image, bbox, state_name, writer, method: str = ""):
    img = image.copy()
    if bbox is not None:
        x, y, w, h = [int(v) for v in bbox]
        color = (0, 255, 0) if method != "color_fallback" else (0, 165, 255)
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
    cv2.putText(img, f"state={state_name} method={method}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    writer.write(img)


def _export_csv(csv_path: str, video_filename: str, events):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "video_filename", "label", "start_timestamp", "end_timestamp",
            "duration_sec", "start_frame", "end_frame", "confidence",
            "evidence_based", "notes",
        ])
        for e in events:
            w.writerow([
                video_filename, e.label,
                format_timestamp(e.start_time), format_timestamp(e.end_time),
                round(e.duration, 2), e.start_frame, e.end_frame,
                e.confidence, e.evidence_based, e.notes,
            ])


def _parse_bbox(s: str) -> tuple[int, int, int, int]:
    parts = [int(p) for p in s.split(",")]
    if len(parts) != 4:
        raise argparse.ArgumentTypeError("bbox must be x,y,w,h")
    return tuple(parts)


def main():
    parser = argparse.ArgumentParser(description="Stage 1 cup event detection pipeline")
    parser.add_argument("--video", required=True, help="Path to input MP4")
    parser.add_argument("--bbox", type=_parse_bbox, default=None,
                         help="Initial cup bbox as x,y,w,h. Omit to select interactively.")
    parser.add_argument("--db", default="output/events.db")
    parser.add_argument("--csv", default="output/events.csv")
    parser.add_argument("--debug-video", default="output/debug_annotated.mp4")
    parser.add_argument("--annotated-video", default="output/annotated_video.mp4",
                        help="Annotated MP4 path; use an empty string to disable")
    parser.add_argument("--pdf", default="output/event_report.pdf",
                        help="PDF report output path; use an empty string to disable")
    args = parser.parse_args()

    events = run_pipeline(
        video_path=args.video,
        bbox=args.bbox,
        db_path=args.db,
        csv_path=args.csv,
        debug_video_path=args.debug_video,
        pdf_path=args.pdf or None,
        annotated_video_path=args.annotated_video or None,
    )

    print(f"\nDetected {len(events)} event(s):\n")
    for e in events:
        print(f"  {format_timestamp(e.start_time)} - {format_timestamp(e.end_time)} "
              f"| {display_label(e.label)} | {describe_event(e.label)}")


if __name__ == "__main__":
    main()
