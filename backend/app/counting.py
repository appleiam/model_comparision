"""Pure IN/OUT counting logic over cached tracks. No model code in here.

Input frames format (as cached in {slot}_raw.json):
    frames[i] = {"i": i, "dets": [{"id": int, "bbox": [x1,y1,x2,y2], "conf": float,
                                   "kpts": [[x,y,c]*17] | None}, ...]}

ROI config (normalized 0..1 coordinates):
    {"roi": [[x,y],...], "line": [[x1,y1],[x2,y2]], "in_side": 1|-1, "params": {...}}

Side convention: s = sign(cross(B - A, P - A)) in pixel coordinates (y down).
A crossing into side `in_side` is IN, into the opposite side is OUT.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

from .config import DEFAULT_COUNT_PARAMS

L_HIP, R_HIP = 11, 12

REASON_NEVER_CROSSED = "never_crossed"
REASON_TOO_SHORT = "too_short_lived"
REASON_OUTSIDE_ROI = "outside_roi"
REASON_JITTER = "jitter_suppressed"


def merged_params(params: Optional[dict]) -> dict:
    p = dict(DEFAULT_COUNT_PARAMS)
    if params:
        p.update({k: v for k, v in params.items() if v is not None})
    return p


# --------------------------------------------------------------------- geometry
def anchor_point(det: dict, mode: str = "hip_fallback", kp_conf: float = 0.5) -> tuple[tuple[float, float], str]:
    """Return ((x, y), source) where source is 'hip', 'bottom' or 'center'."""
    x1, y1, x2, y2 = det["bbox"]
    if mode == "bbox_center":
        return ((x1 + x2) / 2, (y1 + y2) / 2), "center"
    if mode == "hip_fallback":
        k = det.get("kpts")
        if k and len(k) > R_HIP:
            l, r = k[L_HIP], k[R_HIP]
            if l[2] > kp_conf and r[2] > kp_conf:
                return ((l[0] + r[0]) / 2, (l[1] + r[1]) / 2), "hip"
    return ((x1 + x2) / 2, y2), "bottom"


def point_in_polygon(x: float, y: float, poly: list[list[float]]) -> bool:
    """Ray casting; poly in the same coordinate space as (x, y)."""
    n = len(poly)
    if n < 3:
        return True  # no ROI -> everything counts
    inside = False
    j = n - 1
    for i in range(n):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            xin = (xj - xi) * (y - yi) / ((yj - yi) or 1e-12) + xi
            if x < xin:
                inside = not inside
        j = i
    return inside


def signed_distance(p: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
    """Signed perpendicular distance (px) of p to line a->b. Positive = cross > 0 side."""
    dx, dy = b[0] - a[0], b[1] - a[1]
    norm = math.hypot(dx, dy) or 1e-9
    return (dx * (p[1] - a[1]) - dy * (p[0] - a[0])) / norm


def to_px(pts: list[list[float]], w: int, h: int) -> list[list[float]]:
    return [[x * w, y * h] for x, y in pts]


# --------------------------------------------------------------------- tracks
@dataclass
class TrackObs:
    frame: int
    raw_id: int
    anchor: tuple[float, float]
    source: str
    conf: float
    bbox: tuple[float, float, float, float] = (0, 0, 0, 0)


@dataclass
class Track:
    id: int  # canonical id (after lost-track recovery merges)
    obs: list[TrackObs] = field(default_factory=list)

    @property
    def first(self) -> int:
        return self.obs[0].frame

    @property
    def last(self) -> int:
        return self.obs[-1].frame


def build_tracks(frames: list[dict], mode: str, kp_conf: float) -> dict[int, Track]:
    tracks: dict[int, Track] = {}
    for fi, fr in enumerate(frames):
        f = fr.get("i", fi)
        for d in fr.get("dets", []):
            tid = d.get("id")
            if tid is None or tid < 0:
                continue
            pt, src = anchor_point(d, mode, kp_conf)
            tracks.setdefault(tid, Track(tid)).obs.append(
                TrackObs(f, tid, pt, src, d.get("conf", 0.0), tuple(d["bbox"]))
            )
    return tracks


def stabilize_anchors(track: Track, alpha: float = 0.5) -> None:
    """Avoid hip <-> bbox-bottom anchor jumps (they look like line crossings).

    Once a track has shown its hips, frames where the hip keypoints are weak use the hip
    position estimated from the bbox (last known hip location relative to the bbox, EMA).
    Online-friendly: only past observations are used.
    """
    ratio: Optional[tuple[float, float]] = None
    for o in track.obs:
        x1, y1, x2, y2 = o.bbox
        w, h = max(x2 - x1, 1e-6), max(y2 - y1, 1e-6)
        if o.source == "hip":
            r = ((o.anchor[0] - x1) / w, (o.anchor[1] - y1) / h)
            ratio = r if ratio is None else (ratio[0] + alpha * (r[0] - ratio[0]), ratio[1] + alpha * (r[1] - ratio[1]))
        elif o.source == "bottom" and ratio is not None:
            o.anchor = (x1 + ratio[0] * w, y1 + ratio[1] * h)
            o.source = "hip_est"


def _family(source: str) -> str:
    return "hip" if source in ("hip", "hip_est") else source


def find_recoveries(tracks: dict[int, Track], max_gap: int, max_px: float) -> list[dict]:
    """Find (vanished track -> new track) pairs within max_gap frames and max_px pixels.

    Greedy in order of the new track's start frame, nearest candidate wins; each vanished track
    can be continued at most once. Chains (1 -> 5 -> 9) resolve to the root id.
    """
    ordered = sorted(tracks.values(), key=lambda t: t.first)
    used_pred: set[int] = set()
    merges: list[dict] = []
    for t in ordered:
        best = None
        for u in ordered:
            if u.id == t.id or u.id in used_pred:
                continue
            gap = t.first - u.last
            if gap <= 0 or gap > max_gap:
                continue
            ua, ta = u.obs[-1].anchor, t.obs[0].anchor
            dist = math.hypot(ua[0] - ta[0], ua[1] - ta[1])
            if dist <= max_px and (best is None or dist < best[1]):
                best = (u, dist, gap)
        if best:
            u, dist, gap = best
            used_pred.add(u.id)
            merges.append({"frame": t.first, "from_id": t.id, "to_id": u.id, "gap": gap, "dist": round(dist, 1)})
    return merges


def apply_merges(tracks: dict[int, Track], merges: list[dict]) -> tuple[dict[int, Track], dict[int, int]]:
    parent = {m["from_id"]: m["to_id"] for m in merges}

    def root(i: int) -> int:
        seen = set()
        while i in parent and i not in seen:
            seen.add(i)
            i = parent[i]
        return i

    id_map = {tid: root(tid) for tid in tracks}
    merged: dict[int, Track] = {}
    for tid in sorted(tracks, key=lambda k: tracks[k].first):
        cid = id_map[tid]
        merged.setdefault(cid, Track(cid)).obs.extend(tracks[tid].obs)
    for t in merged.values():
        t.obs.sort(key=lambda o: o.frame)
    for m in merges:
        m["to_id"] = id_map[m["to_id"]]
    return merged, id_map


# --------------------------------------------------------------------- counting
def count_tracks(
    frames: list[dict],
    width: int,
    height: int,
    fps: float,
    config: dict,
) -> dict[str, Any]:
    """Count IN/OUT crossings. Returns events, totals, uncounted tracks, merges, id map."""
    p = merged_params(config.get("params"))
    roi = to_px(config.get("roi") or [], width, height)
    (ax, ay), (bx, by) = to_px(config["line"], width, height)
    in_side = 1 if config.get("in_side", 1) >= 0 else -1
    band = float(p["hysteresis"]) * height
    min_age = int(p["min_track_age"])
    fps = fps or 30.0

    raw_tracks = build_tracks(frames, p["anchor"], float(p["kp_conf"]))
    candidates = find_recoveries(raw_tracks, int(p["recovery_frames"]), float(p["recovery_px"]))
    merges = [dict(m) for m in candidates] if p["recovery_enabled"] else []
    tracks, id_map = apply_merges(raw_tracks, merges)
    stabilize = bool(p["anchor_stabilize"]) and p["anchor"] == "hip_fallback"
    if stabilize:
        for t in tracks.values():
            stabilize_anchors(t)

    events: list[dict] = []
    suppressed: list[dict] = []
    track_info: dict[int, dict] = {}

    for t in sorted(tracks.values(), key=lambda t: t.first):
        side = 0  # last confirmed side (outside the hysteresis band), 0 = unknown
        raw_side = 0
        age = 0
        in_roi = 0
        raw_flips = 0
        young_cross = 0
        counted: set[str] = set()
        family = None
        for o in t.obs:
            age += 1
            if not point_in_polygon(o.anchor[0], o.anchor[1], roi):
                continue
            in_roi += 1
            d = signed_distance(o.anchor, (ax, ay), (bx, by))
            if stabilize and family is not None and _family(o.source) != family:
                side = 0  # anchor switched bottom -> hip: re-seed the side, don't count the jump
            family = _family(o.source)
            s_raw = 1 if d > 0 else -1
            if raw_side and s_raw != raw_side:
                raw_flips += 1
            raw_side = s_raw
            if abs(d) <= band:
                continue  # inside hysteresis band: no decision
            s = 1 if d > 0 else -1
            if side and s != side:
                direction = "in" if s == in_side else "out"
                ev = {
                    "frame": o.frame,
                    "time": round(o.frame / fps, 3),
                    "track_id": t.id,
                    "raw_track_id": o.raw_id,
                    "direction": direction,
                    "anchor": [round(o.anchor[0], 1), round(o.anchor[1], 1)],
                    "anchor_source": o.source,
                }
                if age < min_age:
                    young_cross += 1
                    suppressed.append({**ev, "reason": "too_young"})
                elif direction in counted or (counted and not p["count_both_directions"]):
                    suppressed.append({**ev, "reason": "duplicate"})
                else:
                    counted.add(direction)
                    events.append(ev)
            side = s
        track_info[t.id] = {
            "track_id": t.id,
            "raw_ids": sorted({o.raw_id for o in t.obs}),
            "first_frame": t.first,
            "last_frame": t.last,
            "frames": len(t.obs),
            "frames_in_roi": in_roi,
            "counted": sorted(counted),
            "_raw_flips": raw_flips,
            "_young_cross": young_cross,
            "last_anchor": [round(t.obs[-1].anchor[0], 1), round(t.obs[-1].anchor[1], 1)],
        }

    min_frames = int(p["uncounted_min_frames"])
    uncounted = []
    for info in track_info.values():
        if info["counted"] or info["frames"] < min_frames:
            continue
        uncounted.append({**_public(info), "reason": _reason(info, min_frames, min_age)})

    events.sort(key=lambda e: (e["frame"], e["track_id"]))
    anchors: dict[tuple[int, int], tuple[float, float, str]] = {
        (o.frame, o.raw_id): (o.anchor[0], o.anchor[1], o.source) for t in tracks.values() for o in t.obs
    }
    return {
        "anchors": anchors,  # (frame, raw_id) -> (x, y, source); used by the renderer
        "in": sum(e["direction"] == "in" for e in events),
        "out": sum(e["direction"] == "out" for e in events),
        "events": events,
        "suppressed": suppressed,
        "uncounted": sorted(uncounted, key=lambda u: u["first_frame"]),
        "merges": merges,
        "id_switch_candidates": len(candidates),
        "tracks": {tid: _public(i) for tid, i in track_info.items()},
        "id_map": id_map,
        "params": p,
    }


def _public(info: dict) -> dict:
    return {k: v for k, v in info.items() if not k.startswith("_")}


def _reason(info: dict, min_frames: int, min_age: int) -> str:
    if info["frames_in_roi"] < max(1, min(min_frames, min_age)):
        return REASON_OUTSIDE_ROI
    if info["_young_cross"]:
        return REASON_TOO_SHORT
    if info["_raw_flips"]:
        return REASON_JITTER
    if info["frames_in_roi"] < min_age:
        return REASON_TOO_SHORT
    return REASON_NEVER_CROSSED


def cumulative_counts(events: list[dict], n_frames: int) -> tuple[list[int], list[int]]:
    """Per-frame running totals of IN and OUT (inclusive of the event frame)."""
    cin, cout = [0] * n_frames, [0] * n_frames
    by_frame: dict[int, list[str]] = {}
    for e in events:
        by_frame.setdefault(e["frame"], []).append(e["direction"])
    i = o = 0
    for f in range(n_frames):
        for d in by_frame.get(f, []):
            if d == "in":
                i += 1
            else:
                o += 1
        cin[f], cout[f] = i, o
    return cin, cout
