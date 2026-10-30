"""Single GPU worker (asyncio queue) + inference pipeline + results/render helpers.

Jobs are processed strictly one at a time so the two models never run in parallel.
Re-renders triggered by a recount run on a separate single-thread CPU pool and are
superseded (aborted) when a newer recount for the same clip arrives.
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Optional

import cv2
from sqlmodel import Session, select

from . import inference, metrics
from .config import DEFAULT_COUNT_PARAMS, DEFAULT_INFERENCE, DEFAULT_ROI_CONFIG, OUTPUT_DIR, SLOTS
from .counting import count_tracks
from .db import engine
from .models import Clip, Job, RoiPreset, Setting
from .render import render_video

log = logging.getLogger("apcs.worker")

# --------------------------------------------------------------------- state
_queue: Optional[asyncio.Queue] = None
job_state: dict[int, dict[str, Any]] = {}  # live progress, read by the WebSocket
_state_lock = threading.Lock()
_render_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="render")
_render_tokens: dict[int, int] = {}
_clip_locks: dict[int, threading.Lock] = {}


def clip_dir(clip_id: int) -> Path:
    d = OUTPUT_DIR / str(clip_id)
    d.mkdir(parents=True, exist_ok=True)
    return d


def raw_path(clip_id: int, weights: str) -> Path:
    return clip_dir(clip_id) / f"{Path(weights).stem}_raw.json"


def video_path(clip_id: int, slot: str) -> Path:
    return clip_dir(clip_id) / f"{slot}.mp4"


def results_path(clip_id: int) -> Path:
    return clip_dir(clip_id) / "results.json"


ACTIVE_ROI_KEY = "active_roi"


def get_active_roi() -> dict:
    """The ROI applied to every clip (new uploads, jobs without a preset, Apply in the editor)."""
    with Session(engine) as s:
        row = s.get(Setting, ACTIVE_ROI_KEY)
        cfg = json.loads(row.value) if row else json.loads(json.dumps(DEFAULT_ROI_CONFIG))
    cfg["params"] = {**DEFAULT_COUNT_PARAMS, **cfg.get("params", {})}
    return cfg


def set_active_roi(cfg: dict) -> None:
    with Session(engine) as s:
        row = s.get(Setting, ACTIVE_ROI_KEY) or Setting(key=ACTIVE_ROI_KEY, value="")
        row.value = json.dumps(cfg)
        s.add(row)
        s.commit()


def _clip_lock(clip_id: int) -> threading.Lock:
    with _state_lock:
        return _clip_locks.setdefault(clip_id, threading.Lock())


def _update_state(job_id: int, **kw) -> None:
    with _state_lock:
        st = job_state.setdefault(job_id, {"clips": {}})
        clips = kw.pop("clips", None)
        st.update(kw)
        if clips:
            for cid, v in clips.items():
                st["clips"].setdefault(cid, {}).update(v)
        st["version"] = st.get("version", 0) + 1
        st["updated"] = time.time()


def snapshot(job_id: int) -> Optional[dict]:
    with _state_lock:
        st = job_state.get(job_id)
        return json.loads(json.dumps(st)) if st else None


# --------------------------------------------------------------------- queue
def start(loop_ready: asyncio.AbstractEventLoop) -> asyncio.Task:
    global _queue
    _queue = asyncio.Queue()
    return loop_ready.create_task(_worker_loop())


async def _worker_loop() -> None:
    assert _queue is not None
    while True:
        job_id = await _queue.get()
        try:
            await asyncio.to_thread(run_job, job_id)
        except Exception:  # never let the worker die
            log.exception("Job %s crashed", job_id)
        finally:
            _queue.task_done()


async def enqueue(job_id: int) -> None:
    assert _queue is not None
    await _queue.put(job_id)


# --------------------------------------------------------------------- jobs
def run_job(job_id: int) -> None:
    with Session(engine) as s:
        job = s.get(Job, job_id)
        if not job:
            return
        job.status = "running"
        s.add(job)
        s.commit()
        clip_ids = job.clip_ids()
        slots = {"A": job.slot_a, "B": job.slot_b}
        settings = {**DEFAULT_INFERENCE, **json.loads(job.settings_json or "{}")}
        preset_cfg = None
        if job.preset_id:
            preset = s.get(RoiPreset, job.preset_id)
            preset_cfg = preset.config() if preset else None
        preset_id = job.preset_id

    _update_state(job_id, status="running", progress=0.0, message=None)
    errors = []
    for idx, cid in enumerate(clip_ids):
        def prog(frac: float, stage: str, _idx=idx, _cid=cid) -> None:
            _update_state(
                job_id,
                progress=round((_idx + frac) / len(clip_ids) * 100, 1),
                clips={str(_cid): {"progress": round(frac * 100, 1), "stage": stage, "status": "processing"}},
            )
            _set_clip(_cid, progress=round(frac * 100, 1))

        try:
            process_clip(cid, slots, settings, preset_cfg, preset_id, prog)
            _update_state(job_id, clips={str(cid): {"status": "done", "progress": 100.0, "stage": "done"}})
        except Exception as e:
            msg = _friendly_error(e, settings)
            log.error("Clip %s failed: %s\n%s", cid, msg, traceback.format_exc())
            errors.append(f"clip {cid}: {msg}")
            _set_clip(cid, status="error", error=msg)
            _update_state(job_id, clips={str(cid): {"status": "error", "error": msg, "stage": "error"}})
            inference.free_memory()

    status = "error" if errors and len(errors) == len(clip_ids) else "done"
    message = "; ".join(errors) if errors else None
    with Session(engine) as s:
        job = s.get(Job, job_id)
        job.status, job.progress, job.message = status, 100.0, message
        s.add(job)
        s.commit()
    _update_state(job_id, status=status, progress=100.0, message=message)


def _friendly_error(e: Exception, settings: dict) -> str:
    if inference.is_oom(e):
        return f"Out of memory at imgsz={settings.get('imgsz')}. Try a lower imgsz (e.g. 480 or 320)."
    return f"{type(e).__name__}: {e}"


def _set_clip(cid: int, **fields) -> None:
    with Session(engine) as s:
        clip = s.get(Clip, cid)
        if clip:
            for k, v in fields.items():
                setattr(clip, k, v)
            s.add(clip)
            s.commit()


def process_clip(cid: int, slots: dict[str, str], settings: dict, preset_cfg: Optional[dict],
                 preset_id: Optional[int], prog) -> None:
    with Session(engine) as s:
        clip = s.get(Clip, cid)
        if not clip:
            raise ValueError("clip not found")
        clip.status, clip.error, clip.progress = "processing", None, 0.0
        clip.slot_a, clip.slot_b = slots["A"], slots["B"]
        clip.roi_json = json.dumps(preset_cfg or get_active_roi())
        clip.preset_id = preset_id if preset_cfg else None
        s.add(clip)
        s.commit()
        s.refresh(clip)
        path, meta = clip.path, {"fps": clip.fps, "width": clip.width, "height": clip.height}

    # 1-3: inference only for weights without a valid cache
    needed = [w for w in dict.fromkeys(slots.values()) if not _cache_valid(cid, w, settings)]
    if needed:
        run_inference(cid, path, meta, slots, needed, settings, lambda f: prog(f * 0.75, "inference"))
    prog(0.75, "counting")

    # 4, 6: counting + metrics from cache (re-read the active ROI: it may have changed mid-job)
    compute_results(cid, preset_cfg or get_active_roi())
    prog(0.8, "rendering")

    # 5: render
    with Session(engine) as s:
        clip = s.get(Clip, cid)
        clip.render_status = "rendering"
        s.add(clip)
        s.commit()
    token = _next_render_token(cid)
    _render_all(cid, token, lambda f: prog(0.8 + 0.2 * f, "rendering"))
    _set_clip(cid, status="done", progress=100.0)


def _cache_valid(cid: int, weights: str, settings: dict) -> bool:
    p = raw_path(cid, weights)
    if not p.exists():
        return False
    try:
        meta = json.loads(p.read_text())["meta"]
        return all(meta.get(k) == settings[k] for k in ("conf", "iou", "imgsz"))
    except Exception:
        return False


def run_inference(cid: int, path: str, meta: dict, slots: dict[str, str], weights_list: list[str],
                  settings: dict, progress) -> None:
    """Decode each frame once and feed the SAME frame to every model.

    Each weights file runs on its slot's dedicated model instance (if both slots use the same
    weights, it runs once).
    """
    models = {}
    for slot in SLOTS:
        w = slots[slot]
        if w in weights_list and w not in models:
            models[w] = inference.get_model(slot, w)
    for m in models.values():
        inference.reset_tracker(m)

    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("Cannot open video (corrupt or unsupported codec)")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
    frames: dict[str, list] = {w: [] for w in models}
    remap: dict[str, dict[int, int]] = {w: {} for w in models}
    i = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            for w, m in models.items():
                dets, ms, wall = inference.track_frame(m, frame, settings["conf"], settings["iou"], settings["imgsz"])
                for d in dets:  # contiguous IDs per model, in order of first appearance
                    d["id"] = remap[w].setdefault(d["id"], len(remap[w]) + 1)
                frames[w].append({"i": i, "ms": round(ms, 2), "wall_ms": round(wall, 2), "dets": dets})
            i += 1
            if i % 5 == 0:
                progress(min(i / total, 1.0))
    finally:
        cap.release()
    if i == 0:
        raise ValueError("Video contains no decodable frames")

    h, w_ = int(meta["height"]), int(meta["width"])
    for w in models:
        out = {
            "meta": {
                "conf": settings["conf"], "iou": settings["iou"], "imgsz": settings["imgsz"],
                "weights": w, "device": inference.select_device(),
                "fps": meta["fps"], "width": w_, "height": h, "n_frames": i,
                "created": time.time(),
            },
            "frames": frames[w],
        }
        tmp = raw_path(cid, w).with_suffix(".tmp")
        tmp.write_text(json.dumps(out, separators=(",", ":")))
        tmp.replace(raw_path(cid, w))
    progress(1.0)


def load_raw(cid: int, weights: str) -> dict:
    p = raw_path(cid, weights)
    if not p.exists():
        raise FileNotFoundError(f"No cached inference for {weights}; run a job first")
    return json.loads(p.read_text())


# --------------------------------------------------------------------- results
def compute_results(cid: int, config: Optional[dict] = None) -> dict:
    """Counting + metrics for both slots from cached raw results (no inference)."""
    with Session(engine) as s:
        clip = s.get(Clip, cid)
        if config is not None:
            clip.roi_json = json.dumps(config)
            s.add(clip)
            s.commit()
            s.refresh(clip)
        config = clip.roi() or DEFAULT_ROI_CONFIG
        slots = {"A": clip.slot_a, "B": clip.slot_b}
        gt = (clip.gt_in, clip.gt_out)

    out: dict[str, Any] = {"clip_id": cid, "config": config, "updated": time.time(), "models": {}}
    summary: dict[str, Any] = {}
    for slot, w in slots.items():
        if not w:
            continue
        raw = load_raw(cid, w)
        m = raw["meta"]
        c = count_tracks(raw["frames"], m["width"], m["height"], m["fps"], config)
        out["models"][slot] = {
            "slot": slot,
            "weights": w,
            "inference": {k: m[k] for k in ("conf", "iou", "imgsz", "device")},
            "in": c["in"],
            "out": c["out"],
            "events": c["events"],
            "suppressed": c["suppressed"],
            "uncounted": c["uncounted"],
            "merges": c["merges"],
            "summary": metrics.summary(raw, c),
            "per_frame": {k: v for k, v in metrics.per_frame(raw, c).items() if not k.startswith("_")},
        }
        summary[slot] = {"weights": w, "in": c["in"], "out": c["out"], "uncounted": len(c["uncounted"]),
                         "fps": out["models"][slot]["summary"]["fps"]}
        out["models"][slot]["gt"] = metrics.gt_eval(c["in"], c["out"], *gt)
    results_path(cid).write_text(json.dumps(out, separators=(",", ":")))
    _set_clip(cid, summary_json=json.dumps(summary))
    return out


def load_results(cid: int) -> Optional[dict]:
    p = results_path(cid)
    if not p.exists():
        return None
    return json.loads(p.read_text())


# --------------------------------------------------------------------- rendering
def _next_render_token(cid: int) -> int:
    with _state_lock:
        _render_tokens[cid] = _render_tokens.get(cid, 0) + 1
        return _render_tokens[cid]


def _render_all(cid: int, token: int, progress=None) -> bool:
    with Session(engine) as s:
        clip = s.get(Clip, cid)
        config = clip.roi() or DEFAULT_ROI_CONFIG
        slots = [(sl, w) for sl, w in (("A", clip.slot_a), ("B", clip.slot_b)) if w]
        path = clip.path

    def stale() -> bool:
        return _render_tokens.get(cid) != token

    try:
        for k, (slot, w) in enumerate(slots):
            raw = load_raw(cid, w)
            m = raw["meta"]
            c = count_tracks(raw["frames"], m["width"], m["height"], m["fps"], config)
            label = f"{slot}: {Path(w).stem}  conf {m['conf']} imgsz {m['imgsz']}"
            out = video_path(cid, slot)
            tmp_target = out.with_name(f"{slot}.r{token}.mp4")
            render_video(
                path, tmp_target, raw, c, config, label,
                progress=(lambda f, k=k: progress((k + f) / len(slots))) if progress else None,
                should_abort=stale,
            )
            if stale():
                tmp_target.unlink(missing_ok=True)
                return False
            with _clip_lock(cid):
                tmp_target.replace(out)
    except Exception as e:
        log.exception("Render failed for clip %s", cid)
        _set_clip(cid, render_status="error", error=f"render: {e}")
        return False
    with Session(engine) as s:
        clip = s.get(Clip, cid)
        if _render_tokens.get(cid) == token:
            clip.render_status = "done"
            clip.render_version += 1
            s.add(clip)
            s.commit()
    return True


def schedule_render(cid: int) -> None:
    token = _next_render_token(cid)
    _set_clip(cid, render_status="rendering")
    _render_pool.submit(_render_all, cid, token)


def schedule_recount(cid: int, config: dict, preset_id: Optional[int] = None) -> None:
    """Background recount + re-render (CPU pool); superseded by any newer recount of the clip."""
    token = _next_render_token(cid)
    _set_clip(cid, render_status="rendering")

    def task() -> None:
        if _render_tokens.get(cid) != token:
            return
        try:
            compute_results(cid, config)
            _set_clip(cid, preset_id=preset_id)
        except Exception as e:
            log.exception("Recount failed for clip %s", cid)
            _set_clip(cid, render_status="error", error=f"recount: {e}")
            return
        _render_all(cid, token)

    _render_pool.submit(task)


def recount_all(config: dict, preset_id: Optional[int] = None, exclude: Optional[int] = None) -> list[int]:
    """Apply a ROI to every clip: processed clips are recounted in the background, the rest
    just store it (queued/processing clips read the active ROI when they reach counting)."""
    scheduled = []
    with Session(engine) as s:
        clips = s.exec(select(Clip)).all()
        for c in clips:
            if c.id == exclude:
                continue
            if c.status == "done":
                scheduled.append(c.id)
            elif c.status != "processing":
                c.roi_json, c.preset_id = json.dumps(config), preset_id
                s.add(c)
        s.commit()
    for cid in scheduled:
        schedule_recount(cid, config, preset_id)
    return scheduled


def make_thumbnail(cid: int, frame) -> Path:
    p = clip_dir(cid) / "thumb.jpg"
    h, w = frame.shape[:2]
    scale = 240 / w
    small = cv2.resize(frame, (240, max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
    cv2.imwrite(str(p), small, [cv2.IMWRITE_JPEG_QUALITY, 80])
    return p
