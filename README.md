# Cup Video Event Understanding & Search — Project Plan + Stage 1

A personal, experimental system that watches fixed-camera footage, detects
when a cup is picked up / moved / placed / returned, timestamps each event,
and lets you search a video by natural-language query instead of scrubbing
through it. Built with your own tracking logic and (from Stage 3 on) your
own trained model — no pretrained action-recognition nets or cloud AI APIs.

## Full architecture (all 5 stages)

```
                        ┌─────────────────────────────────────────┐
                        │              Stage 5: UI                 │
                        │     Streamlit app (upload/search/play)   │
                        └───────────────┬───────────────────────────┘
                                         │
                        ┌────────────────▼──────────────────────────┐
                        │       Stage 4: Search Engine                │
                        │  rule-based NL query → SQL over `events`    │
                        └────────────────▲──────────────────────────┘
                                         │ reads
┌───────────────┐   ┌──────────────┐   ┌┴─────────────┐   ┌──────────────────┐
│ Stage 1:       │   │ Stage 1:     │   │ Stage 1/2:   │   │ Stage 3:          │
│ video_io.py    │──▶│ tracker.py   │──▶│ events.py    │──▶│ trained model      │
│ (frames + ts)  │   │ (CSRT+color  │   │ (state       │   │ (LSTM/TCN,        │
│                │   │  fallback)   │   │  machine)    │   │  replaces/refines │
└───────────────┘   └──────────────┘   └──────┬───────┘   │  the heuristic    │
                                               │            │  classifier)      │
                                               ▼            └──────────────────┘
                                        ┌──────────────┐
                                        │ db.py        │
                                        │ SQLite store │
                                        └──────────────┘
```

**Stage 1 (this delivery):** classical CV only — CSRT tracking + a
hand-written position-history state machine. No learning yet. This is the
foundation everything else builds on: Stage 3's model will be trained to
predict the same event labels this state machine currently assigns by
hand-tuned rules, using position-history features this code already
computes.

**Stage 2 (next):** annotation workflow — record your own clips, label
start/end timestamps per class (`No Action`, `Pick Up Cup`, `Move Cup`,
`Place Cup`, `Return Cup`), export to CSV/JSON, split by video/session
(not by frame) into train/val/test.

**Stage 3:** feature extraction (centroid position, velocity, displacement
from origin, rolling window stats) → PyTorch dataset/dataloader → a small
temporal model (start with an LSTM over the feature sequence, or a 1D
Temporal CNN) → training loop with checkpointing → evaluation
(precision/recall/F1/confusion matrix/temporal IoU) on held-out videos.

**Stage 4:** `events` table (already created in Stage 1) + rule-based query
parser (regex/keyword matching for verbs like "move", "return", "pick up",
and time ranges like "between minute 5 and 10") → SQL filters → results
with filename + timestamp + a way to seek the video to that point.

**Stage 5:** Streamlit app wrapping all of the above: upload → run Stage
1+3 pipeline → browse timeline → search box → video player seeked to the
matching segment → CSV/JSON export button.

## Why this design

- **CSRT + color fallback, not a detector network:** per your constraint
  of no pretrained recognition models, Stage 1 uses OpenCV's classical
  CSRT tracker (correlation-filter based, not a neural detector) seeded by
  a manual bounding box, with a local HSV-histogram back-projection search
  as a recovery path when CSRT's per-frame match fails (e.g. brief
  occlusion by a hand). This satisfies "handle temporary tracking
  failures" without reaching for a pretrained model.
- **Position-history state machine, not motion alone:** "returned" is only
  ever emitted if the cup previously left a radius around its recorded
  original position and has now come back inside that radius — not just
  "it stopped moving." See `src/events.py` docstring for the exact logic.
- **Honest confidence, honestly labeled:** Stage 1's `confidence` field is
  a heuristic (fraction of frames tracked by CSRT vs. the less-reliable
  color fallback), explicitly documented as such — not a calibrated
  probability. It gets replaced by the trained model's real confidence in
  Stage 3.
- **SQLite from day one:** even though search (Stage 4) isn't built yet,
  Stage 1 already writes every event into the same `events` table the
  search engine will query, so nothing needs to be re-architected later.

## Project structure

```
cup_video_ai/
├── requirements.txt
├── check_dependencies.py     # checks + optionally installs only what's missing
├── README.md
├── src/
│   ├── config.py              # all Stage 1 thresholds, in one place
│   ├── video_io.py            # frame extraction with real timestamps
│   ├── tracker.py             # CSRT tracking + color-based recovery
│   ├── events.py              # Event dataclass + state-machine detector
│   ├── db.py                  # SQLite storage
│   └── pipeline.py            # wires it together, CLI entry point
├── tests/
│   ├── make_synthetic_video.py  # generates a ground-truth test clip
│   └── test_stage1.py           # asserts detected events match ground truth
├── data/            # put your own MP4s here
├── models/          # (unused until Stage 3)
└── output/          # events.db, events.csv, debug_annotated.mp4 land here
```

## Running Stage 1

### 1. Check/install dependencies
```bash
python check_dependencies.py
```
This only prints what's missing and asks before installing anything — it
never reinstalls what you already have. (Stage 1 itself only needs
`opencv-contrib-python`, `numpy`, and `pandas`; `torch`/`scikit-learn`/
`streamlit` aren't needed until later stages, so it's fine if those show
as missing for now.)

### 2. Verify the pipeline with a synthetic video (no camera needed)
```bash
python tests/make_synthetic_video.py
python tests/test_stage1.py
```
Expected output:
```
Detected 2 events, expected 2
  checking cup_moved ~ expected cup_moved: 2.03-3.17s vs 2.0-3.0s
  checking cup_returned ~ expected cup_returned: 5.03-6.17s vs 5.0-6.0s

ALL STAGE 1 TESTS PASSED
```
This confirms the whole chain — tracking, state machine, timestamping —
works correctly against known ground truth before you trust it on real
footage.

### 3. Run it on your own video
Record a short clip with a fixed camera: cup sits still, you move it, it
sits, you return it. Then:

```bash
# with a display: drag a box around the cup when it pops up
python -m src.pipeline --video data/my_video.mp4

# or specify the cup's starting box directly (x,y,w,h in pixels)
python -m src.pipeline --video data/my_video.mp4 --bbox 310,220,60,80
```

Outputs land in `output/`:
- `events.csv` — human-readable event log
- `events.db` — SQLite table `events` (same data, queryable with SQL)
- `debug_annotated.mp4` — your video with the tracked box + state overlay,
  so you can visually confirm the tracker is actually following the cup

### Tuning
If events are missed or spuriously split, adjust `src/config.py`:
- Jittery/noisy tracking splitting one move into many → raise
  `movement_threshold_px` and/or `noise_floor_px`.
- Real movements not being detected as "settled" → lower `settle_frames`.
- "Returned" not firing even though the cup is visibly back → raise
  `return_radius_px` slightly (lighting/shadow can shift the tracked
  centroid a few pixels from the true original position).

## What Stage 1 does **not** yet do (by design — see spec section 10)

- It cannot yet distinguish "picked up" from "slid across the table" —
  both are reported as `cup_moved`; that distinction needs Stage 3's
  learned model. This is stated directly in `events.py`'s docstring and in
  the `notes` field of each event, rather than silently implied.
- It assumes a fixed camera and roughly consistent lighting; it has no
  occlusion-duration limit beyond `max_track_failures` frames, and a cup
  fully hidden for longer than that will be reported as tracking lost
  with no event recorded.
- Confidence scores are heuristic, not learned.

## Next step

Once you've run Stage 1 on a few real clips and are happy with the
detected events, say so and we'll move to Stage 2: building the
annotation workflow so you can start collecting labeled training data for
the action-recognition model.

## Automatic PDF report

Every successful `python -m src.pipeline --video "data/cup video.mp4"` run now creates
`output/event_report.pdf` automatically. It contains video metadata, detected events,
timestamps, descriptions and a summary, in a printable landscape layout.

To choose a different output name, add `--pdf "output/my_report.pdf"`.
To disable PDF creation, add `--pdf ""`.

Install the PDF dependency once with `python -m pip install reportlab` (or
`python -m pip install -r requirements.txt`). This report displays Stage 1
rule-based detections, **not** ground-truth action labels or trained-model
probabilities.


## Professional annotated video (LinkedIn demo)

Run from the project root:

```cmd
python -m src.pipeline --video "data/cup video.mp4" --annotated-video "output/annotated_video.mp4"
```

Select the cup in the first frame, then press SPACE or ENTER. The output
`output/annotated_video.mp4` includes a cup bounding box, tracked centroid,
original-position marker, motion arrow, event label, timestamp and event timeline.
The final event labels are overlaid in a SECOND PASS after detection finishes.
PDF report still saves to `output/event_report.pdf`.

Important: Stage 1 event labels are heuristic and may be incorrect; confidence
is tracker-quality heuristic, not model accuracy. This is not a trained AI model.
Use `--annotated-video ""` to disable the professional MP4 output.

## Tracking correction (dark cup demo)
The updated `src/tracker.py` uses conservative dark-object segmentation for the
provided **black cup on a light background**. Unlike the previous CSRT-only
approach, it shows `TRACKING LOST` rather than drawing a false box when the
cup leaves the frame. This is a scene-specific prototype, not a general cup
recognizer; other objects/backgrounds require tuning or a trained detector.

**Important:** Event labels and timings are still heuristic and have NOT been
validated against human-annotated ground truth. In particular, `cup_placed`
may appear while the cup is held in a hand. The printed `confidence` is a
tracking-quality heuristic, NOT an ML prediction probability. Review the
video and correct the event detector before presenting event accuracy claims.

Run from the project directory:
`python -m src.pipeline --video "data/cup video.mp4"`
Outputs: `output/annotated_video.mp4` and `output/event_report.pdf`.


## Clear labels for sharing
The annotated video and PDF use reader-friendly labels. `MOVEMENT STOPPED` means the tracker detected a pause; it does **not** prove the cup touched the table. `BACK NEAR START` indicates proximity to the initial location. The CSV retains original internal labels for compatibility. Tracking scores are not classification accuracy.
