"""Synthetic-track tests for app.counting (no model code involved)."""
from __future__ import annotations

import pytest

from app.counting import (
    REASON_JITTER,
    REASON_OUTSIDE_ROI,
    REASON_TOO_SHORT,
    anchor_point,
    count_tracks,
    cumulative_counts,
    point_in_polygon,
)

W, H, FPS = 1000, 1000, 10.0

# Horizontal line at y=0.5; in_side=+1 means "below the line" is inside the bus.
BASE = {
    "roi": [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]],
    "line": [[0.0, 0.5], [1.0, 0.5]],
    "in_side": 1,
    "params": {"anchor": "bbox_bottom", "hysteresis": 0.03, "min_track_age": 3, "uncounted_min_frames": 3},
}


def cfg(**params):
    c = {**BASE, "params": {**BASE["params"], **params}}
    return c


def det(tid, x, y, conf=0.9):
    """Detection whose bbox bottom-center is (x, y)."""
    return {"id": tid, "bbox": [x - 20, y - 100, x + 20, y], "conf": conf, "kpts": None}


def make_frames(tracks: dict[int, list[tuple[int, float, float]]], n: int | None = None):
    """tracks: id -> [(frame, x, y)]"""
    last = max(f for pts in tracks.values() for f, _, _ in pts)
    n = n or last + 1
    frames = [{"i": i, "dets": []} for i in range(n)]
    for tid, pts in tracks.items():
        for f, x, y in pts:
            frames[f]["dets"].append(det(tid, x, y))
    return frames


def linear(f0, f1, y0, y1, x=500):
    steps = f1 - f0
    return [(f0 + k, x, y0 + (y1 - y0) * k / steps) for k in range(steps + 1)]


def run(frames, c=None):
    return count_tracks(frames, W, H, FPS, c or cfg())


# ------------------------------------------------------------------ basics
def test_point_in_polygon():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    assert point_in_polygon(5, 5, sq)
    assert not point_in_polygon(15, 5, sq)


def test_anchor_hip_and_fallback():
    kpts = [[0, 0, 0.0]] * 17
    kpts = [list(k) for k in kpts]
    kpts[11] = [100, 200, 0.9]
    kpts[12] = [120, 220, 0.9]
    d = {"bbox": [50, 50, 150, 300], "kpts": kpts}
    assert anchor_point(d, "hip_fallback") == ((110, 210), "hip")
    kpts[12][2] = 0.3  # one hip weak -> bbox bottom-center fallback
    assert anchor_point(d, "hip_fallback") == ((100, 300), "bottom")
    assert anchor_point(d, "bbox_center") == ((100, 175), "center")
    assert anchor_point({"bbox": [0, 0, 10, 10], "kpts": None}, "hip_fallback") == ((5, 10), "bottom")


# ------------------------------------------------------------------ crossings
def test_clean_crossing_in():
    r = run(make_frames({1: linear(0, 20, 200, 800)}))
    assert (r["in"], r["out"]) == (1, 0)
    ev = r["events"][0]
    assert ev["track_id"] == 1 and ev["direction"] == "in"
    assert ev["frame"] > 10  # counted once it clears the band below the line
    assert r["uncounted"] == []


def test_clean_crossing_out():
    r = run(make_frames({1: linear(0, 20, 800, 200)}))
    assert (r["in"], r["out"]) == (0, 1)


def test_in_side_flip_inverts_direction():
    c = cfg()
    c["in_side"] = -1
    r = count_tracks(make_frames({1: linear(0, 20, 200, 800)}), W, H, FPS, c)
    assert (r["in"], r["out"]) == (0, 1)


def test_jitter_on_line_not_counted():
    # Oscillates +-20px around y=500, band is 30px -> never leaves band on the other side.
    pts = [(0, 500, 300), (1, 500, 350), (2, 500, 400)]
    pts += [(3 + k, 500, 500 + (20 if k % 2 else -20)) for k in range(20)]
    r = run(make_frames({1: pts}))
    assert (r["in"], r["out"]) == (0, 0)
    assert r["uncounted"][0]["reason"] == REASON_JITTER


def test_jitter_without_hysteresis_double_counts():
    pts = [(k, 500, 500 + (40 if k % 2 else -40)) for k in range(10)]
    r = run(make_frames({1: pts}), cfg(hysteresis=0.0, min_track_age=1))
    # one count per direction per track
    assert (r["in"], r["out"]) == (1, 1)
    assert len(r["suppressed"]) > 0


def test_in_then_out_counts_both_or_first_only():
    pts = linear(0, 10, 200, 800) + linear(11, 21, 800, 200)
    frames = make_frames({1: pts})
    assert (run(frames)["in"], run(frames)["out"]) == (1, 1)
    r = run(frames, cfg(count_both_directions=False))
    assert (r["in"], r["out"]) == (1, 0)


def test_short_track_not_counted():
    # Crosses in 2 frames: too young at crossing time.
    pts = [(0, 500, 400), (1, 500, 600), (2, 500, 650)]
    r = run(make_frames({1: pts}), cfg(min_track_age=5))
    assert (r["in"], r["out"]) == (0, 0)
    assert r["suppressed"][0]["reason"] == "too_young"
    assert r["uncounted"][0]["reason"] == REASON_TOO_SHORT


def test_id_switch_mid_crossing_recovered():
    # Track 1 walks down to the line and vanishes; track 2 appears 4 frames later 30px away.
    t1 = linear(0, 10, 200, 490)
    t2 = linear(14, 24, 520, 800)
    frames = make_frames({1: t1, 2: t2})

    r = run(frames, cfg(recovery_enabled=False))
    assert (r["in"], r["out"]) == (0, 0)  # the failure mode: nobody counted
    assert r["id_switch_candidates"] == 1
    assert {u["track_id"] for u in r["uncounted"]} == {1, 2}

    r = run(frames, cfg(recovery_enabled=True, recovery_frames=10, recovery_px=60))
    assert (r["in"], r["out"]) == (1, 0)
    assert r["merges"] == [{"frame": 14, "from_id": 2, "to_id": 1, "gap": 4, "dist": 30.0}]
    assert r["events"][0]["track_id"] == 1 and r["events"][0]["raw_track_id"] == 2
    assert r["id_map"] == {1: 1, 2: 1}


def test_recovery_respects_distance_and_gap():
    t1 = linear(0, 10, 200, 490)
    far = make_frames({1: t1, 2: [(f, 900, y) for f, _, y in linear(14, 24, 520, 800)]})
    assert run(far)["merges"] == []
    late = make_frames({1: t1, 2: linear(30, 40, 520, 800)})
    assert run(late, cfg(recovery_frames=10))["merges"] == []


def test_track_outside_roi_ignored():
    c = cfg()
    c["roi"] = [[0.0, 0.0], [0.4, 0.0], [0.4, 1.0], [0.0, 1.0]]  # left 40% only
    r = count_tracks(make_frames({1: linear(0, 20, 200, 800, x=800)}), W, H, FPS, c)
    assert (r["in"], r["out"]) == (0, 0)
    assert r["uncounted"][0]["reason"] == REASON_OUTSIDE_ROI


def test_two_people_cross_simultaneously():
    frames = make_frames({1: linear(0, 20, 200, 800, x=300), 2: linear(0, 20, 800, 200, x=700)})
    r = run(frames)
    assert (r["in"], r["out"]) == (1, 1)


def test_cumulative_counts():
    events = [{"frame": 2, "direction": "in"}, {"frame": 4, "direction": "out"}, {"frame": 4, "direction": "in"}]
    cin, cout = cumulative_counts(events, 6)
    assert cin == [0, 0, 1, 1, 2, 2]
    assert cout == [0, 0, 0, 0, 1, 1]


def _pose_det(tid, y_top, hip_conf):
    """Person bbox 240px tall starting at y_top; hips 120px below the top."""
    k = [[0.0, 0.0, 0.0] for _ in range(17)]
    k[11] = [490.0, y_top + 120, 0.9]
    k[12] = [510.0, y_top + 120, hip_conf]
    return {"id": tid, "bbox": [460, y_top, 540, y_top + 240], "conf": 0.9, "kpts": k}


def test_hip_flicker_does_not_create_phantom_crossings():
    # Hips stay ~60px ABOVE the line (y=500) while bbox bottom is ~60px BELOW it.
    # Right-hip conf flickers around 0.5 -> raw anchor jumps across the line every frame.
    frames = []
    for i in range(40):
        y_top = 300 + i * 0.5
        frames.append({"i": i, "dets": [_pose_det(1, y_top, 0.6 if i % 2 else 0.4)]})
    c = cfg(anchor="hip_fallback", anchor_stabilize=False)
    assert run(frames, c)["in"] + run(frames, c)["out"] >= 1  # the artifact
    r = run(frames, cfg(anchor="hip_fallback", anchor_stabilize=True))
    assert (r["in"], r["out"]) == (0, 0)


def test_hip_stabilized_real_crossing_still_counted():
    frames = []
    for i in range(40):
        y_top = 100 + i * 15  # hips go from 220 to 805: real crossing
        frames.append({"i": i, "dets": [_pose_det(1, y_top, 0.6 if i % 3 else 0.3)]})
    r = run(frames, cfg(anchor="hip_fallback"))
    assert (r["in"], r["out"]) == (1, 0)


@pytest.mark.parametrize("mode", ["hip_fallback", "bbox_center", "bbox_bottom"])
def test_all_anchor_modes_count(mode):
    r = run(make_frames({1: linear(0, 30, 100, 900)}), cfg(anchor=mode))
    assert r["in"] == 1
