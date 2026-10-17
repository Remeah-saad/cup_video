"""Conservative, evidence-based cup motion detection (Stage 1).

Movement is based on smoothed centroid displacement, not inferred table contact.
Missing tracking never creates an event. Labels are not ML predictions.
"""
from dataclasses import dataclass
from collections import deque
from math import hypot
from statistics import median

@dataclass
class Event:
    label: str
    start_frame: int
    end_frame: int
    start_time: float
    end_time: float
    start_pos: tuple[float, float]
    end_pos: tuple[float, float]
    confidence: float
    evidence_based: bool
    notes: str = ""
    @property
    def duration(self):
        return self.end_time-self.start_time

class EventDetector:
    def __init__(self, config, fps):
        self.config, self.fps = config, fps
        self.original_pos = None
        self.events = []
        self.samples = []
        self.states = []
    def set_origin(self, pos):
        self.original_pos = pos
    def update(self, frame_index, timestamp, centroid, track_success, track_method):
        if self.original_pos is None:
            raise RuntimeError('Call set_origin first')
        self.samples.append((frame_index, timestamp, tuple(centroid) if track_success and centroid is not None else None))
    def finalize(self):
        """Build events after full video, so timestamps are consistent with overlays."""
        import numpy as np
        if not self.samples:
            return self.events
        fps = self.fps
        n = self.samples[-1][0]+1
        centers = [None]*n
        for idx, _, center in self.samples:
            if 0 <= idx < n: centers[idx] = center
        centers[0] = self.original_pos
        radius = max(2, round(.08*fps))
        smooth = [None]*n
        for i in range(n):
            if centers[i] is None: continue
            pts = [centers[j] for j in range(max(0,i-radius), min(n,i+radius+1)) if centers[j] is not None]
            if len(pts)>=2:
                smooth[i]=(median(p[0] for p in pts),median(p[1] for p in pts))
        lag=max(2,round(.18*fps))
        speeds=[0.0]*n
        for i in range(lag,n):
            if smooth[i] is None or smooth[i-lag] is None: continue
            if any(centers[j] is None for j in range(i-lag,i+1)): continue
            speeds[i]=hypot(smooth[i][0]-smooth[i-lag][0],smooth[i][1]-smooth[i-lag][1])/(lag/fps)
        # Moderate threshold rejects small stationary centroid jitter.
        threshold=max(30.0, self.config.movement_threshold_px*4.0)
        moving=[s>=threshold for s in speeds]
        # Bridge brief tracking jitter, but never across missing observations.
        max_gap=round(.24*fps)
        hits=[i for i,m in enumerate(moving) if m]
        for a,b in zip(hits,hits[1:]):
            if b-a<=max_gap+1 and all(centers[j] is not None for j in range(a,b+1)):
                moving[a:b+1]=[True]*(b-a+1)
        segments=[]; start=None
        for i,flag in enumerate(moving+[False]):
            if flag and start is None: start=i
            if not flag and start is not None:
                end=i-1
                if (end-start+1)/fps>=.35:
                    segments.append((start,end))
                start=None
        self.states=['MOVING' if x else ('STATIONARY' if centers[i] is not None else 'TRACKING LOST') for i,x in enumerate(moving)]
        self.events=[]
        away=False
        for start,end in segments:
            a=next((smooth[i] for i in range(start,min(end+1,n)) if smooth[i] is not None),None)
            b=next((smooth[i] for i in range(end,start-1,-1) if smooth[i] is not None),None)
            if a is None or b is None: continue
            displacement=hypot(a[0]-b[0],a[1]-b[1])
            if displacement<max(18,self.config.movement_threshold_px*2): continue
            near= hypot(b[0]-self.original_pos[0],b[1]-self.original_pos[1])<=self.config.return_radius_px
            label='cup_returned' if near and away else 'cup_moved'
            away=not near
            self.events.append(Event(label,start,end,start/fps,end/fps,a,b,1.0,True,
                'Observed centroid movement; placement and hand contact are not determined.' if label=='cup_moved' else
                'Tracked cup position moved near the initial reference point; surface contact not verified.'))
        return self.events
