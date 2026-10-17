"""Professional video with evidence-based motion status per frame."""
from pathlib import Path
import cv2
from .presentation import display_label, VIDEO_DETAILS

COLORS = {"cup_moved": (226, 178, 56), "cup_placed": (88, 188, 255),
          "cup_returned": (113, 219, 132)}


def _text(img, message, pos, scale=0.65, color=(255, 255, 255), thickness=2):
    cv2.putText(img, message, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (10, 14, 20), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, message, pos, cv2.FONT_HERSHEY_SIMPLEX, scale,
                color, thickness, cv2.LINE_AA)


def create_annotated_video(video_path, output_path, fps, frame_count, tracks, events, origin, states=None):
    """Render synchronized motion status and finalized movement intervals."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot reopen video for annotation: {video_path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"),
                             fps, (width, height))
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot write annotated video: {output_path}")
    # Scale graphics to the source video, without resizing the actual video.
    s = max(0.55, min(width / 1280, height / 720, 1.4))
    pad = max(12, int(20 * s))
    header_h = max(92, int(118 * s))
    footer_h = max(100, int(118 * s))
    duration = max(frame_count / fps, 0.001)
    try:
        idx = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            t = idx / fps
            bbox, method = tracks[idx] if idx < len(tracks) else (None, "lost")
            overlay = img.copy()
            cv2.rectangle(overlay, (0, 0), (width, min(header_h, height)), (15, 22, 32), -1)
            cv2.rectangle(overlay, (0, max(0, height-footer_h)), (width, height), (15, 22, 32), -1)
            cv2.addWeighted(overlay, 0.82, img, 0.18, 0, img)
            _text(img, "CUP EVENT DETECTION", (pad, max(23, int(37*s))),
                  scale=0.85*s, thickness=2)
            _text(img, f"TIME  {t:06.2f}s  /  {duration:06.2f}s", (pad, max(46, int(70*s))),
                  scale=0.62*s, color=(185, 206, 224), thickness=1)
            active = next((e for e in events if e.start_time <= t <= e.end_time), None)
            status = states[idx] if states is not None and idx < len(states) else ("MOVING" if active else "STATIONARY")
            label = "TRACKING LOST" if bbox is None else ("CUP MOVING" if status == "MOVING" else "CUP STATIONARY")
            label_color = (80,225,136) if status == "MOVING" else ((65,95,245) if bbox is None else (185,206,224))
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.75*s, 2)
            _text(img, label, (max(pad, width-tw-pad), max(24, int(39*s))),
                  scale=0.75*s, color=label_color)
            if active:
                _text(img, f"{active.start_time:.2f}s - {active.end_time:.2f}s",
                      (max(pad, width-int(210*s)), max(47, int(72*s))),
                      scale=0.55*s, color=label_color, thickness=1)
            if bbox is not None:
                _text(img, "Motion estimated from tracked cup position",
                      (pad, max(header_h + 24, int(145*s))), scale=0.54*s,
                      color=label_color, thickness=1)
            if origin is not None:
                ox, oy = [int(v) for v in origin]
                cv2.drawMarker(img, (ox, oy), (230, 185, 78), cv2.MARKER_CROSS, max(12, int(20*s)), 2)
                _text(img, "ORIGIN", (min(width-95, ox+12), max(header_h+18, oy-12)),
                      scale=0.48*s, color=(230, 185, 78), thickness=1)
            if bbox is not None:
                x,y,w,h = map(int, bbox)
                color = (80, 225, 136) if method != "color_fallback" else (40, 174, 255)
                cv2.rectangle(img, (x,y), (x+w,y+h), color, max(2, int(3*s)))
                cx,cy = x+w//2,y+h//2
                cv2.circle(img, (cx,cy), max(3, int(5*s)), color, -1)
                if origin is not None:
                    cv2.arrowedLine(img, (int(origin[0]), int(origin[1])), (cx,cy),
                                    (91, 207, 255), max(1,int(2*s)), tipLength=0.10)
                _text(img, "CUP TRACKED" if method != "color_fallback" else "CUP TRACKED (EST.)",
                      (max(pad,x), max(header_h+20,y-10)), scale=0.52*s, color=color, thickness=1)
            else:
                _text(img, "TRACKING LOST", (pad, min(height-footer_h-10, header_h+35)),
                      scale=0.65*s, color=(65, 95, 245))
            # Timeline: segments are drawn from the finalized events.
            bar_x, bar_right = pad, width-pad
            bar_y = height-max(54, int(62*s))
            bar_w = max(1, bar_right-bar_x)
            cv2.rectangle(img, (bar_x, bar_y), (bar_right, bar_y+max(9,int(13*s))), (85,92,105), -1)
            for e in events:
                left = bar_x + int(max(0,min(1,e.start_time/duration))*bar_w)
                right = bar_x + int(max(0,min(1,e.end_time/duration))*bar_w)
                cv2.rectangle(img, (left,bar_y), (max(left+2,right),bar_y+max(9,int(13*s))),
                              COLORS.get(e.label,(200,200,200)), -1)
            marker = bar_x + int(min(1,t/duration)*bar_w)
            cv2.line(img,(marker,bar_y-7),(marker,bar_y+max(9,int(13*s))+7),(255,255,255),2)
            _text(img, f"EVENTS  {len(events)}", (pad, height-max(18,int(19*s))),
                  scale=0.55*s, thickness=1)
            legend = "MOTION INTERVALS / NEAR START"  # label colors in the timeline
            _text(img, legend, (max(pad,width-int(260*s)),height-max(18,int(19*s))),
                  scale=0.48*s, color=(185,206,224), thickness=1)
            writer.write(img)
            idx += 1
    finally:
        writer.release()
        cap.release()
    return idx
