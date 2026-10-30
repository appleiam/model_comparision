"""Annotated, browser-playable H.264 MP4 rendering (via imageio-ffmpeg's bundled ffmpeg)."""
from __future__ import annotations

import colorsys
import os
from collections import defaultdict, deque
from pathlib import Path
from typing import Callable, Optional

import cv2
import imageio_ffmpeg
import numpy as np

from .config import RENDER_MAX_WIDTH, TRAIL_LEN
from .counting import anchor_point, to_px

# COCO-17 skeleton
SKELETON = [
    (15, 13), (13, 11), (16, 14), (14, 12), (11, 12), (5, 11), (6, 12), (5, 6),
    (5, 7), (6, 8), (7, 9), (8, 10), (1, 2), (0, 1), (0, 2), (1, 3), (2, 4), (3, 5), (4, 6),
]
ORANGE = (0, 140, 255)  # BGR: detected but never counted
GREY = (150, 150, 150)
GREEN = (80, 220, 80)
RED = (60, 60, 240)
ROI_COLOR = (255, 200, 0)
LINE_COLOR = (255, 255, 255)
FLASH_FRAMES = 8


def _id_color(tid: int) -> tuple[int, int, int]:
    # Hues in the green/cyan/blue/purple range, avoiding orange/red.
    h = (0.25 + (tid * 0.137) % 0.55)
    r, g, b = colorsys.hsv_to_rgb(h, 0.75, 1.0)
    return int(b * 255), int(g * 255), int(r * 255)


def _track_color(cid: int, counted: set, uncounted: set) -> tuple[int, int, int]:
    if cid in counted:
        return _id_color(cid)
    if cid in uncounted:
        return ORANGE  # detected but never counted: the failure under investigation
    return GREY  # too short to be reported


def _even(v: int) -> int:
    return v - (v % 2)


def render_video(
    video_path: str,
    out_path: Path,
    raw: dict,
    count: dict,
    config: dict,
    label: str,
    progress: Optional[Callable[[float], None]] = None,
    should_abort: Optional[Callable[[], bool]] = None,
) -> Path:
    meta = raw["meta"]
    W, H, fps = meta["width"], meta["height"], meta["fps"] or 30.0
    scale = min(1.0, RENDER_MAX_WIDTH / W) if W else 1.0
    ow, oh = _even(int(round(W * scale))), _even(int(round(H * scale)))
    sx, sy = ow / W, oh / H
    p = count["params"]
    id_map = {int(k): v for k, v in count["id_map"].items()}
    anchors = count["anchors"]
    counted_ids = {tid for tid, t in count["tracks"].items() if t["counted"]}
    uncounted_ids = {u["track_id"] for u in count["uncounted"]}
    events_by_frame: dict[int, list[dict]] = defaultdict(list)
    for e in count["events"]:
        events_by_frame[e["frame"]].append(e)
    merges_by_frame: dict[int, list[dict]] = defaultdict(list)
    for m in count["merges"]:
        merges_by_frame[m["frame"]].append(m)

    roi = np.array([[x * sx, y * sy] for x, y in to_px(config.get("roi") or [], W, H)], np.int32)
    (ax, ay), (bx, by) = [(x * sx, y * sy) for x, y in to_px(config["line"], W, H)]
    in_side = 1 if config.get("in_side", 1) >= 0 else -1
    dx, dy = bx - ax, by - ay
    norm = max(np.hypot(dx, dy), 1e-6)
    nx, ny = -dy / norm * in_side, dx / norm * in_side  # unit normal pointing to IN side
    band = p["hysteresis"] * H * sy
    mx, my = (ax + bx) / 2, (ay + by) / 2

    # Static overlay (ROI fill) blended once per frame.
    overlay_mask = np.zeros((oh, ow, 3), np.uint8)
    if len(roi) >= 3:
        cv2.fillPoly(overlay_mask, [roi], ROI_COLOR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".tmp.mp4")
    writer = imageio_ffmpeg.write_frames(
        str(tmp),
        (ow, oh),
        fps=fps,
        codec="libx264",
        pix_fmt_in="bgr24",
        pix_fmt_out="yuv420p",
        macro_block_size=1,
        output_params=["-preset", "veryfast", "-crf", "23", "-movflags", "+faststart"],
    )
    writer.send(None)

    cap = cv2.VideoCapture(video_path)
    frames = raw["frames"]
    n = len(frames)
    trails: dict[int, deque] = defaultdict(lambda: deque(maxlen=TRAIL_LEN))
    cin = cout = 0
    flash: list[tuple[int, dict]] = []  # (frames_left, event)
    merge_flash: list[tuple[int, dict]] = []
    aborted = False
    try:
        for fi in range(n):
            ok, frame = cap.read()
            if not ok:
                frame = np.zeros((H, W, 3), np.uint8)
            if scale != 1.0 or frame.shape[1] != ow or frame.shape[0] != oh:
                frame = cv2.resize(frame, (ow, oh), interpolation=cv2.INTER_AREA)
            if len(roi) >= 3:
                cv2.addWeighted(overlay_mask, 0.12, frame, 1.0, 0, frame)
                cv2.polylines(frame, [roi], True, ROI_COLOR, 2, cv2.LINE_AA)

            for e in events_by_frame.get(fi, []):
                if e["direction"] == "in":
                    cin += 1
                else:
                    cout += 1
                flash.append((FLASH_FRAMES, e))
            for m in merges_by_frame.get(fi, []):
                merge_flash.append((FLASH_FRAMES * 2, m))

            # counting line + hysteresis band + IN arrow
            flash_col = None
            if flash:
                flash_col = GREEN if flash[-1][1]["direction"] == "in" else RED
            for off in (-band, band):
                cv2.line(frame, _pt(ax + nx * off, ay + ny * off), _pt(bx + nx * off, by + ny * off), (120, 120, 120), 1, cv2.LINE_AA)
            cv2.line(frame, _pt(ax, ay), _pt(bx, by), flash_col or LINE_COLOR, 5 if flash_col else 2, cv2.LINE_AA)
            arrow_len = max(30, 0.08 * oh)
            cv2.arrowedLine(frame, _pt(mx, my), _pt(mx + nx * arrow_len, my + ny * arrow_len), GREEN, 2, cv2.LINE_AA, tipLength=0.3)
            _text(frame, "IN", _pt(mx + nx * arrow_len * 1.25 - 10, my + ny * arrow_len * 1.25 + 5), GREEN, 0.55)

            for d in frames[fi]["dets"]:
                cid = id_map.get(d["id"], d["id"])
                col = _track_color(cid, counted_ids, uncounted_ids)
                x1, y1, x2, y2 = d["bbox"]
                cv2.rectangle(frame, _pt(x1 * sx, y1 * sy), _pt(x2 * sx, y2 * sy), col, 2)
                tag = f"#{cid}" + (f"({d['id']})" if cid != d["id"] else "") + f" {d['conf']:.2f}"
                _label(frame, tag, _pt(x1 * sx, y1 * sy), col)
                k = d.get("kpts")
                if k:
                    for a, b in SKELETON:
                        if k[a][2] > 0.5 and k[b][2] > 0.5:
                            cv2.line(frame, _pt(k[a][0] * sx, k[a][1] * sy), _pt(k[b][0] * sx, k[b][1] * sy), col, 1, cv2.LINE_AA)
                    for x, y, c in k:
                        if c > 0.5:
                            cv2.circle(frame, _pt(x * sx, y * sy), 2, col, -1, cv2.LINE_AA)
                a = anchors.get((fi, d["id"]))
                axp, ayp = (a[0], a[1]) if a else anchor_point(d, p["anchor"], p["kp_conf"])[0]
                ap = _pt(axp * sx, ayp * sy)
                trails[cid].append(ap)
                cv2.circle(frame, ap, 5, (255, 255, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, ap, 5, col, 2, cv2.LINE_AA)

            # trails (only for tracks present in the last TRAIL_LEN frames)
            present = {id_map.get(d["id"], d["id"]) for d in frames[fi]["dets"]}
            for cid, tr in trails.items():
                if cid in present and len(tr) > 1:
                    col = _track_color(cid, counted_ids, uncounted_ids)
                    cv2.polylines(frame, [np.array(tr, np.int32)], False, col, 2, cv2.LINE_AA)

            for left, e in flash:
                col = GREEN if e["direction"] == "in" else RED
                ex, ey = e["anchor"]
                _text(frame, f"{e['direction'].upper()} +1 #{e['track_id']}", _pt(ex * sx + 8, ey * sy - 8), col, 0.7, 2)
            for left, m in merge_flash:
                _text(frame, f"ID {m['from_id']}->{m['to_id']} merged", (10, oh - 40), (255, 0, 255), 0.5)
            flash = [(l - 1, e) for l, e in flash if l > 1]
            merge_flash = [(l - 1, m) for l, m in merge_flash if l > 1]

            # HUD
            cv2.rectangle(frame, (0, 0), (260, 62), (0, 0, 0), -1)
            _text(frame, label, (8, 20), (255, 255, 255), 0.5)
            _text(frame, f"IN {cin}", (8, 50), GREEN, 0.8, 2)
            _text(frame, f"OUT {cout}", (120, 50), RED, 0.8, 2)
            _text(frame, f"f {fi}  t {fi / fps:.2f}s", (8, oh - 12), (255, 255, 255), 0.5)
            if flash_col:
                cv2.rectangle(frame, (0, 0), (ow - 1, oh - 1), flash_col, 4)

            writer.send(frame.tobytes())
            if progress and fi % 10 == 0:
                progress(fi / n)
            if should_abort and should_abort():
                aborted = True
                break
    finally:
        cap.release()
        writer.close()
    if aborted:
        tmp.unlink(missing_ok=True)
        return out_path
    os.replace(tmp, out_path)
    if progress:
        progress(1.0)
    return out_path


def _pt(x: float, y: float) -> tuple[int, int]:
    return int(round(x)), int(round(y))


def _text(img, s, org, col, scale=0.5, thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 2, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, col, thick, cv2.LINE_AA)


def _label(img, s, org, col):
    (tw, th), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    x, y = org
    y = max(th + 4, y)
    cv2.rectangle(img, (x, y - th - 4), (x + tw + 4, y), col, -1)
    cv2.putText(img, s, (x + 2, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1, cv2.LINE_AA)


def probe_video(path: str) -> dict:
    """Read fps / resolution / frame count. Raises ValueError for unreadable files."""
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("OpenCV could not open the video (corrupt or unsupported codec)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    ok, frame = cap.read()
    cap.release()
    if not ok or frame is None:
        raise ValueError("Video contains no decodable frames")
    if not fps or fps != fps or fps > 240:
        fps = 30.0
    return {"fps": float(fps), "width": w or frame.shape[1], "height": h or frame.shape[0], "frame_count": max(n, 0), "first_frame": frame}


def read_frame(path: str, index: int) -> Optional[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
    ok, frame = cap.read()
    cap.release()
    return frame if ok else None
