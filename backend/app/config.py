"""Paths, model defaults and counting defaults."""
from __future__ import annotations

import os
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parent.parent
ROOT_DIR = BACKEND_DIR.parent
DATA_DIR = Path(os.environ.get("APCS_DATA_DIR", ROOT_DIR / "data"))
UPLOAD_DIR = DATA_DIR / "uploads"
OUTPUT_DIR = DATA_DIR / "outputs"
WEIGHTS_DIR = BACKEND_DIR / "weights"
FRONTEND_DIST = ROOT_DIR / "frontend" / "dist"
DB_PATH = DATA_DIR / "apcs.db"

for _d in (UPLOAD_DIR, OUTPUT_DIR, WEIGHTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# Stock weights auto-downloaded into WEIGHTS_DIR on first start.
DEFAULT_WEIGHTS = ["yolov8s-pose.pt", "yolov8m-pose.pt"]
DEFAULT_SLOTS = {"A": "yolov8s-pose.pt", "B": "yolov8m-pose.pt"}
SLOTS = ("A", "B")

ALLOWED_VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv"}

# Inference defaults (editable per job in the UI).
DEFAULT_INFERENCE = {"conf": 0.25, "iou": 0.5, "imgsz": 640}
PERSON_CLASS = 0
TRACKER_CFG = "bytetrack.yaml"

# Annotated videos are downscaled to this width (keeps re-renders fast).
RENDER_MAX_WIDTH = int(os.environ.get("APCS_RENDER_MAX_WIDTH", "960"))
TRAIL_LEN = 30

DEFAULT_COUNT_PARAMS = {
    "anchor": "hip_fallback",  # hip_fallback | bbox_center | bbox_bottom
    "kp_conf": 0.5,  # min keypoint conf for hips 11/12
    "anchor_stabilize": True,  # estimate hips from bbox when keypoints drop (no anchor jumps)
    "hysteresis": 0.03,  # fraction of frame height
    "min_track_age": 3,  # frames before a track may be counted
    "count_both_directions": True,  # IN then OUT on one track -> both logged
    "recovery_enabled": True,
    "recovery_frames": 10,
    "recovery_px": 60,
    "uncounted_min_frames": 5,  # min frames to report a track as "uncounted"
}

DEFAULT_ROI_CONFIG = {
    "roi": [[0.05, 0.05], [0.95, 0.05], [0.95, 0.95], [0.05, 0.95]],
    "line": [[0.05, 0.5], [0.95, 0.5]],
    # +1 / -1: sign of cross(line_b - line_a, p - line_a) that is "inside the bus".
    # With a left->right line and image y pointing down, +1 means "below the line".
    "in_side": 1,
    "params": DEFAULT_COUNT_PARAMS,
}
