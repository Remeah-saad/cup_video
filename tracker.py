"""Conservative cup tracker for the supplied dark-cup / light-background demo.

Unlike an unconstrained CSRT tracker, this tracker never draws a box on an
unverified target.  The dark-cup segmentation is scene-specific, not a trained
object detector; it may need tuning for a different cup or background.
"""
from dataclasses import dataclass
import cv2
import numpy as np

@dataclass
class TrackResult:
    success: bool
    bbox: tuple[int, int, int, int] | None
    centroid: tuple[float, float] | None
    method: str

class CupTracker:
    def __init__(self):
        self._last_bbox = None
        self._reference_area = None
        self._last_center = None
        self._lost_count = 0

    @staticmethod
    def _bbox_centroid(bbox):
        x,y,w,h=bbox
        return x+w/2, y+h/2

    def init(self, frame, bbox):
        x,y,w,h = map(int,bbox)
        if w <= 0 or h <= 0: raise ValueError('Invalid cup bounding box')
        self._last_bbox = (x,y,w,h)
        self._last_center = self._bbox_centroid(self._last_bbox)
        self._reference_area = w*h

    def update(self, frame, config):
        height,width=frame.shape[:2]
        hsv=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)
        # The demo's cup has a very dark body; skin, wall and tabletop do not.
        mask=cv2.inRange(hsv,(0,0,0),(179,255,94))
        # Ignore dark objects in the upper wall (e.g. monitor at top right).
        mask[:int(height*0.12),:]=0
        mask=cv2.morphologyEx(mask,cv2.MORPH_OPEN,np.ones((3,3),np.uint8))
        mask=cv2.morphologyEx(mask,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8))
        contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        candidates=[]
        for contour in contours:
            area=cv2.contourArea(contour)
            if area < max(350, self._reference_area*0.055): continue
            x,y,w,h=cv2.boundingRect(contour)
            if w<25 or h<20: continue
            if y+h/2 < height*0.36: continue
            if w/h < 0.55 or w/h>5: continue
            if w*h>width*height*0.45: continue
            cx,cy=x+w/2,y+h/2
            dist=np.hypot(cx-self._last_center[0],cy-self._last_center[1])
            # Allow reacquisition after the cup exits the view; prefer larger
            # dark cup-shaped blobs over small shadow or clothing fragments.
            score=area/max(1,self._reference_area) - (dist/max(width,height))*0.25
            candidates.append((score,(x,y,w,h)))
        if not candidates:
            self._lost_count+=1
            return TrackResult(False,None,None,'lost')
        candidates.sort(reverse=True)
        x,y,w,h=candidates[0][1]
        # Expand segmentation around the cup rim; clamp to actual frame.
        padx=int(w*.06)+3; pady=int(h*.08)+4
        x0=max(0,x-padx); y0=max(0,y-pady)
        x1=min(width,x+w+padx); y1=min(height,y+h+pady)
        bbox=(x0,y0,x1-x0,y1-y0)
        center=self._bbox_centroid(bbox)
        self._last_bbox=bbox
        self._last_center=center
        self._lost_count=0
        return TrackResult(True,bbox,center,'dark_cup_segmentation')
