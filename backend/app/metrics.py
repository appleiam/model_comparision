"""Per-frame metrics, per-model summaries and ground-truth evaluation."""
from __future__ import annotations

from typing import Any, Optional

from .counting import cumulative_counts


def per_frame(raw: dict, count: dict) -> dict[str, list]:
    frames = raw["frames"]
    n = len(frames)
    id_map = count["id_map"]
    spans = [(t["first_frame"], t["last_frame"]) for t in count["tracks"].values()]
    active = [0] * n
    for a, b in spans:
        for f in range(max(0, a), min(n - 1, b) + 1):
            active[f] += 1
    n_dets, mean_conf, ms = [], [], []
    for fr in frames:
        dets = fr["dets"]
        n_dets.append(len(dets))
        mean_conf.append(round(sum(d["conf"] for d in dets) / len(dets), 3) if dets else None)
        ms.append(round(fr.get("ms", 0.0), 1))
    cin, cout = cumulative_counts(count["events"], n)
    return {
        "n_dets": n_dets,
        "mean_conf": mean_conf,
        "infer_ms": ms,
        "active_tracks": active,
        "cum_in": cin,
        "cum_out": cout,
        "_ids": len(id_map),
    }


def summary(raw: dict, count: dict) -> dict[str, Any]:
    frames = raw["frames"]
    confs = [d["conf"] for fr in frames for d in fr["dets"]]
    ms = [fr.get("ms", 0.0) for fr in frames if fr.get("ms")]
    wall = [fr.get("wall_ms", 0.0) for fr in frames if fr.get("wall_ms")]
    avg_ms = sum(ms) / len(ms) if ms else 0.0
    avg_wall = sum(wall) / len(wall) if wall else avg_ms
    return {
        "avg_conf": round(sum(confs) / len(confs), 3) if confs else 0.0,
        "avg_dets": round(len(confs) / len(frames), 2) if frames else 0.0,
        "avg_infer_ms": round(avg_ms, 1),
        "avg_wall_ms": round(avg_wall, 1),
        "fps": round(1000 / avg_wall, 1) if avg_wall else 0.0,
        "n_raw_tracks": len(count["id_map"]),
        "n_tracks": len(count["tracks"]),
        "id_switches": count["id_switch_candidates"],
        "merges": len(count["merges"]),
        "uncounted": len(count["uncounted"]),
        "device": raw["meta"].get("device"),
    }


def gt_eval(pred_in: int, pred_out: int, gt_in: Optional[int], gt_out: Optional[int]) -> Optional[dict]:
    if gt_in is None or gt_out is None:
        return None
    return {
        "err_in": pred_in - gt_in,
        "err_out": pred_out - gt_out,
        "abs_error": abs(pred_in - gt_in) + abs(pred_out - gt_out),
        "correct": pred_in == gt_in and pred_out == gt_out,
    }
