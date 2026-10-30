"""FastAPI app: routes, CORS, static serving."""
from __future__ import annotations

import asyncio
import csv
import io
import json
import logging
import shutil
import threading
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Literal, Optional

import cv2
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlmodel import Session, select

from . import inference, metrics, worker
from .config import (
    ALLOWED_VIDEO_EXT, DEFAULT_COUNT_PARAMS, DEFAULT_INFERENCE, DEFAULT_ROI_CONFIG, DEFAULT_SLOTS,
    FRONTEND_DIST, OUTPUT_DIR, UPLOAD_DIR,
)
from .db import engine, get_session, init_db
from .models import Clip, Job, RoiPreset
from .render import probe_video, read_frame

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger("apcs")

startup_state = {"models_ready": False, "error": None}


def _load_models_bg() -> None:
    try:
        inference.ensure_default_weights()
        inference.load_default_models()
        startup_state["models_ready"] = True
    except Exception as e:  # pragma: no cover
        log.exception("Model loading failed")
        startup_state["error"] = str(e)


@asynccontextmanager
async def lifespan(app: FastAPI):
    init_db()
    with Session(engine) as s:  # recover from a crash/restart mid-job
        for c in s.exec(select(Clip).where(Clip.status.in_(["queued", "processing"]))).all():
            c.status, c.error = "error", "interrupted (server restarted)"
            s.add(c)
        for c in s.exec(select(Clip).where(Clip.render_status == "rendering")).all():
            c.render_status = "error"
            s.add(c)
        for j in s.exec(select(Job).where(Job.status.in_(["queued", "running"]))).all():
            j.status, j.message = "error", "interrupted (server restarted)"
            s.add(j)
        s.commit()
    threading.Thread(target=_load_models_bg, daemon=True).start()
    task = worker.start(asyncio.get_running_loop())
    yield
    task.cancel()


app = FastAPI(title="APCS Model Comparator", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


# ------------------------------------------------------------------ helpers
def _clip_or_404(s: Session, clip_id: int) -> Clip:
    clip = s.get(Clip, clip_id)
    if not clip:
        raise HTTPException(404, "clip not found")
    return clip


def _clip_dict(c: Clip) -> dict[str, Any]:
    summ = c.summary() or {}
    gt = {}
    for slot, v in summ.items():
        gt[slot] = metrics.gt_eval(v["in"], v["out"], c.gt_in, c.gt_out)
    return {
        "id": c.id,
        "name": c.original_name,
        "fps": c.fps,
        "width": c.width,
        "height": c.height,
        "frame_count": c.frame_count,
        "duration": c.duration,
        "status": c.status,
        "progress": c.progress,
        "error": c.error,
        "created_at": c.created_at.isoformat(),
        "gt_in": c.gt_in,
        "gt_out": c.gt_out,
        "gt_note": c.gt_note,
        "slot_a": c.slot_a,
        "slot_b": c.slot_b,
        "preset_id": c.preset_id,
        "roi": c.roi(),
        "summary": summ,
        "gt_eval": gt,
        "render_status": c.render_status,
        "render_version": c.render_version,
    }


# ------------------------------------------------------------------ system
@app.get("/system")
def system():
    info = inference.system_info()
    info.update(
        models_ready=startup_state["models_ready"],
        startup_error=startup_state["error"],
        defaults={"inference": DEFAULT_INFERENCE, "count_params": DEFAULT_COUNT_PARAMS, "roi": DEFAULT_ROI_CONFIG},
    )
    return info


# ------------------------------------------------------------------ clips
@app.post("/clips/upload")
async def upload_clips(files: list[UploadFile] = File(...), s: Session = Depends(get_session)):
    out = []
    for f in files:
        ext = Path(f.filename or "").suffix.lower()
        if ext not in ALLOWED_VIDEO_EXT:
            out.append({"name": f.filename, "error": f"unsupported extension {ext}"})
            continue
        dest = UPLOAD_DIR / f"{uuid.uuid4().hex[:12]}{ext}"
        with open(dest, "wb") as fh:
            shutil.copyfileobj(f.file, fh)
        try:
            info = probe_video(str(dest))
        except ValueError as e:
            dest.unlink(missing_ok=True)
            out.append({"name": f.filename, "error": str(e)})
            continue
        n = info["frame_count"]
        clip = Clip(
            original_name=f.filename or dest.name, path=str(dest), fps=info["fps"], width=info["width"],
            height=info["height"], frame_count=n, duration=round(n / info["fps"], 2) if n else 0.0,
            roi_json=json.dumps(worker.get_active_roi()),
        )
        s.add(clip)
        s.commit()
        s.refresh(clip)
        worker.make_thumbnail(clip.id, info["first_frame"])
        out.append(_clip_dict(clip))
    return out


@app.get("/clips")
def list_clips(s: Session = Depends(get_session)):
    return [_clip_dict(c) for c in s.exec(select(Clip).order_by(Clip.id.desc())).all()]


@app.get("/clips/{clip_id}")
def get_clip(clip_id: int, s: Session = Depends(get_session)):
    return _clip_dict(_clip_or_404(s, clip_id))


@app.delete("/clips/{clip_id}")
def delete_clip(clip_id: int, s: Session = Depends(get_session)):
    clip = _clip_or_404(s, clip_id)
    if clip.status == "processing":
        raise HTTPException(409, "clip is being processed")
    Path(clip.path).unlink(missing_ok=True)
    shutil.rmtree(OUTPUT_DIR / str(clip_id), ignore_errors=True)
    s.delete(clip)
    s.commit()
    return {"ok": True}


class GroundTruth(BaseModel):
    gt_in: Optional[int] = Field(None, ge=0)
    gt_out: Optional[int] = Field(None, ge=0)
    note: Optional[str] = None


@app.post("/clips/{clip_id}/ground-truth")
def set_ground_truth(clip_id: int, body: GroundTruth, s: Session = Depends(get_session)):
    clip = _clip_or_404(s, clip_id)
    clip.gt_in, clip.gt_out, clip.gt_note = body.gt_in, body.gt_out, body.note
    s.add(clip)
    s.commit()
    s.refresh(clip)
    return _clip_dict(clip)


@app.get("/clips/{clip_id}/thumbnail")
def thumbnail(clip_id: int):
    p = OUTPUT_DIR / str(clip_id) / "thumb.jpg"
    if not p.exists():
        raise HTTPException(404)
    return FileResponse(p, media_type="image/jpeg")


@app.get("/clips/{clip_id}/frame")
def frame(clip_id: int, i: int = 0, s: Session = Depends(get_session)):
    clip = _clip_or_404(s, clip_id)
    img = read_frame(clip.path, max(0, i))
    if img is None:
        raise HTTPException(404, "frame not readable")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 85])
    return Response(buf.tobytes(), media_type="image/jpeg")


@app.get("/clips/{clip_id}/results")
def results(clip_id: int, s: Session = Depends(get_session)):
    clip = _clip_or_404(s, clip_id)
    r = worker.load_results(clip_id)
    if not r:
        raise HTTPException(404, "no results yet")
    for m in r["models"].values():  # GT may have changed since results were computed
        m["gt"] = metrics.gt_eval(m["in"], m["out"], clip.gt_in, clip.gt_out)
    r["clip"] = _clip_dict(clip)
    return r


class RoiConfig(BaseModel):
    roi: list[list[float]]
    line: list[list[float]] = Field(..., min_length=2, max_length=2)
    in_side: int = 1
    params: dict[str, Any] = {}


class RecountBody(RoiConfig):
    preset_id: Optional[int] = None
    # "all": save as the active ROI and recount every clip; "clip": preview on this clip only
    scope: Literal["all", "clip"] = "all"


def _recount(clip_id: int, cfg: dict, preset_id: Optional[int] = None) -> dict:
    try:
        r = worker.compute_results(clip_id, cfg)
    except FileNotFoundError as e:
        raise HTTPException(409, str(e))
    with Session(engine) as s:
        clip = s.get(Clip, clip_id)
        clip.preset_id = preset_id
        s.add(clip)
        s.commit()
    worker.schedule_render(clip_id)
    return r


@app.post("/clips/{clip_id}/recount")
def recount(clip_id: int, body: RecountBody, s: Session = Depends(get_session)):
    clip = _clip_or_404(s, clip_id)
    if clip.status == "processing":
        raise HTTPException(409, "clip is still processing")
    cfg = body.model_dump(exclude={"preset_id", "scope"})
    cfg["params"] = {**DEFAULT_COUNT_PARAMS, **cfg["params"]}
    _recount(clip_id, cfg, body.preset_id)
    others: list[int] = []
    if body.scope == "all":
        worker.set_active_roi(cfg)
        others = worker.recount_all(cfg, body.preset_id, exclude=clip_id)
    s.expire_all()  # pick up render_status written by the worker
    out = results(clip_id, s)
    out["recount_scheduled"] = others
    return out


@app.get("/roi/active")
def get_active_roi():
    return worker.get_active_roi()


@app.post("/roi/active")
def set_active_roi(body: RoiConfig):
    """Save the ROI used by every clip and recount all processed clips in the background."""
    cfg = body.model_dump()
    cfg["params"] = {**DEFAULT_COUNT_PARAMS, **cfg["params"]}
    worker.set_active_roi(cfg)
    return {"config": cfg, "recount_scheduled": worker.recount_all(cfg)}


@app.get("/roi/active/export")
def export_active_roi(clip_id: Optional[int] = None, s: Session = Depends(get_session)):
    clip = s.get(Clip, clip_id) if clip_id else s.exec(select(Clip).order_by(Clip.id.desc())).first()
    return _export(worker.get_active_roi(), "active-roi", clip)


@app.get("/clips/{clip_id}/video/{model}")
def video(clip_id: int, model: str, request: Request):
    slot = model.upper()
    if slot not in ("A", "B"):
        raise HTTPException(400, "model must be A or B")
    p = worker.video_path(clip_id, slot)
    if not p.exists():
        raise HTTPException(404, "video not rendered yet")
    return _range_response(p, request, "video/mp4")


def _range_response(path: Path, request: Request, media_type: str):
    size = path.stat().st_size
    rng = request.headers.get("range")
    headers = {"Accept-Ranges": "bytes", "Cache-Control": "no-cache"}
    if not rng or not rng.startswith("bytes="):
        return FileResponse(path, media_type=media_type, headers=headers)
    start_s, _, end_s = rng[6:].split(",")[0].partition("-")
    try:
        if start_s:
            start = int(start_s)
            end = int(end_s) if end_s else size - 1
        else:  # suffix range: last N bytes
            start, end = size - int(end_s), size - 1
    except ValueError:
        raise HTTPException(416)
    end = min(end, size - 1)
    if start < 0 or start > end:
        return Response(status_code=416, headers={"Content-Range": f"bytes */{size}"})
    length = end - start + 1

    def chunks():
        with open(path, "rb") as f:
            f.seek(start)
            left = length
            while left > 0:
                data = f.read(min(1 << 16, left))
                if not data:
                    break
                left -= len(data)
                yield data

    headers.update({"Content-Range": f"bytes {start}-{end}/{size}", "Content-Length": str(length)})
    return StreamingResponse(chunks(), status_code=206, media_type=media_type, headers=headers)


@app.get("/clips/{clip_id}/roi-export")
def clip_roi_export(clip_id: int, s: Session = Depends(get_session)):
    clip = _clip_or_404(s, clip_id)
    return _export(clip.roi() or DEFAULT_ROI_CONFIG, f"clip-{clip_id}", clip)


# ------------------------------------------------------------------ jobs
class JobBody(BaseModel):
    clip_ids: list[int]
    slot_a: str = DEFAULT_SLOTS["A"]
    slot_b: str = DEFAULT_SLOTS["B"]
    preset_id: Optional[int] = None
    conf: float = Field(DEFAULT_INFERENCE["conf"], gt=0, lt=1)
    iou: float = Field(DEFAULT_INFERENCE["iou"], gt=0, lt=1)
    imgsz: int = Field(DEFAULT_INFERENCE["imgsz"], ge=160, le=1920)


@app.post("/jobs")
async def create_job(body: JobBody, s: Session = Depends(get_session)):
    weights = inference.list_weights()
    for w in (body.slot_a, body.slot_b):
        if w not in weights:
            raise HTTPException(400, f"weights {w} not found in backend/weights")
    if body.preset_id and not s.get(RoiPreset, body.preset_id):
        raise HTTPException(400, "preset not found")
    clips = [s.get(Clip, cid) for cid in body.clip_ids]
    if not body.clip_ids or any(c is None for c in clips):
        raise HTTPException(400, "unknown clip id")
    busy = [c.id for c in clips if c.status in ("queued", "processing")]
    if busy:
        raise HTTPException(409, f"clips already queued/processing: {busy}")
    imgsz = body.imgsz - body.imgsz % 32
    job = Job(
        clip_ids_json=json.dumps(body.clip_ids), slot_a=body.slot_a, slot_b=body.slot_b, preset_id=body.preset_id,
        settings_json=json.dumps({"conf": body.conf, "iou": body.iou, "imgsz": imgsz}),
    )
    s.add(job)
    for c in clips:
        c.status, c.progress, c.error = "queued", 0.0, None
        s.add(c)
    s.commit()
    s.refresh(job)
    worker._update_state(job.id, status="queued", progress=0.0,
                         clips={str(c): {"progress": 0.0, "status": "queued", "stage": "queued"} for c in body.clip_ids})
    await worker.enqueue(job.id)
    return _job_dict(job)


def _job_dict(j: Job) -> dict:
    live = worker.snapshot(j.id) or {}
    return {
        "id": j.id, "clip_ids": j.clip_ids(), "slot_a": j.slot_a, "slot_b": j.slot_b, "preset_id": j.preset_id,
        "settings": json.loads(j.settings_json), "status": live.get("status", j.status),
        "progress": live.get("progress", j.progress), "message": live.get("message", j.message),
        "clips": live.get("clips", {}), "created_at": j.created_at.isoformat(),
    }


@app.get("/jobs")
def list_jobs(limit: int = 20, s: Session = Depends(get_session)):
    return [_job_dict(j) for j in s.exec(select(Job).order_by(Job.id.desc()).limit(limit)).all()]


@app.get("/jobs/{job_id}")
def get_job(job_id: int, s: Session = Depends(get_session)):
    j = s.get(Job, job_id)
    if not j:
        raise HTTPException(404, "job not found")
    return _job_dict(j)


@app.websocket("/ws/jobs/{job_id}")
async def ws_job(ws: WebSocket, job_id: int):
    await ws.accept()
    last = -1
    try:
        while True:
            with Session(engine) as s:
                j = s.get(Job, job_id)
                if not j:
                    await ws.send_json({"error": "job not found"})
                    break
                d = _job_dict(j)
            st = worker.snapshot(job_id) or {}
            ver = st.get("version", 0)
            if ver != last:
                last = ver
                await ws.send_json(d)
            if d["status"] in ("done", "error"):
                break
            await asyncio.sleep(0.3)
    except WebSocketDisconnect:
        return
    await ws.close()


# ------------------------------------------------------------------ ROI presets
class PresetBody(RoiConfig):
    name: str


@app.get("/roi-presets")
def list_presets(s: Session = Depends(get_session)):
    return [{"id": p.id, "name": p.name, "config": p.config(), "created_at": p.created_at.isoformat()}
            for p in s.exec(select(RoiPreset).order_by(RoiPreset.name)).all()]


@app.post("/roi-presets")
def create_preset(body: dict, s: Session = Depends(get_session)):
    """Create/overwrite (by name) a preset. Accepts either the editor format or an exported JSON."""
    name = (body.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "name required")
    src = body.get("normalized") or body  # exported files carry coordinates under "normalized"
    try:
        cfg = RoiConfig(
            roi=src["roi"], line=src["line"], in_side=src.get("in_side", body.get("in_side", 1)),
            params=body.get("params") or src.get("params") or {},
        ).model_dump()
    except Exception as e:
        raise HTTPException(400, f"invalid ROI config: {e}")
    cfg["params"] = {**DEFAULT_COUNT_PARAMS, **cfg["params"]}
    existing = s.exec(select(RoiPreset).where(RoiPreset.name == name)).first()
    p = existing or RoiPreset(name=name, config_json="{}")
    p.config_json = json.dumps(cfg)
    s.add(p)
    s.commit()
    s.refresh(p)
    return {"id": p.id, "name": p.name, "config": p.config()}


@app.delete("/roi-presets/{preset_id}")
def delete_preset(preset_id: int, s: Session = Depends(get_session)):
    p = s.get(RoiPreset, preset_id)
    if not p:
        raise HTTPException(404)
    s.delete(p)
    s.commit()
    return {"ok": True}


def _export(cfg: dict, name: str, clip: Optional[Clip], width: Optional[int] = None, height: Optional[int] = None):
    w = width or (clip.width if clip else None)
    h = height or (clip.height if clip else None)
    norm = {"roi": cfg["roi"], "line": cfg["line"], "in_side": cfg.get("in_side", 1)}
    doc = {
        "name": name,
        "format": "apcs-roi/v1",
        "side_convention": "in_side = sign of cross(line[1]-line[0], p-line[0]) in image coords (y down) that is inside the bus",
        "normalized": norm,
        "params": cfg.get("params", {}),
    }
    if w and h:
        doc["resolution"] = {"width": w, "height": h}
        doc["pixels"] = {
            "roi": [[round(x * w), round(y * h)] for x, y in cfg["roi"]],
            "line": [[round(x * w), round(y * h)] for x, y in cfg["line"]],
            "in_side": norm["in_side"],
            "hysteresis_px": round(cfg.get("params", {}).get("hysteresis", DEFAULT_COUNT_PARAMS["hysteresis"]) * h, 1),
        }
    return JSONResponse(doc, headers={"Content-Disposition": f'attachment; filename="{name}.roi.json"'})


@app.get("/roi-presets/{preset_id}/export")
def export_preset(preset_id: int, clip_id: Optional[int] = None, width: Optional[int] = None,
                  height: Optional[int] = None, s: Session = Depends(get_session)):
    p = s.get(RoiPreset, preset_id)
    if not p:
        raise HTTPException(404)
    clip = s.get(Clip, clip_id) if clip_id else s.exec(select(Clip).order_by(Clip.id.desc())).first()
    safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in p.name)
    return _export(p.config(), safe, clip, width, height)


@app.post("/roi-presets/{preset_id}/apply-all")
def apply_preset_all(preset_id: int, s: Session = Depends(get_session)):
    """Make the preset the active ROI and recount every clip with it."""
    p = s.get(RoiPreset, preset_id)
    if not p:
        raise HTTPException(404)
    worker.set_active_roi(p.config())
    return {"recount_scheduled": worker.recount_all(p.config(), preset_id)}


# ------------------------------------------------------------------ report
def _report_rows(s: Session) -> list[dict]:
    rows = []
    for c in s.exec(select(Clip).order_by(Clip.id)).all():
        r = worker.load_results(c.id) if c.status == "done" else None
        if not r:
            continue
        row = {"clip_id": c.id, "name": c.original_name, "duration": c.duration, "gt_in": c.gt_in, "gt_out": c.gt_out,
               "gt_note": c.gt_note, "models": {}}
        for slot, m in r["models"].items():
            reasons: dict[str, int] = {}
            for u in m["uncounted"]:
                reasons[u["reason"]] = reasons.get(u["reason"], 0) + 1
            row["models"][slot] = {
                "weights": m["weights"], "in": m["in"], "out": m["out"],
                "gt": metrics.gt_eval(m["in"], m["out"], c.gt_in, c.gt_out),
                "fps": m["summary"]["fps"], "avg_conf": m["summary"]["avg_conf"],
                "uncounted": len(m["uncounted"]), "id_switches": m["summary"]["id_switches"], "reasons": reasons,
            }
        rows.append(row)
    return rows


@app.get("/report")
def report(s: Session = Depends(get_session)):
    rows = _report_rows(s)
    agg: dict[str, dict] = {}
    for row in rows:
        for slot, m in row["models"].items():
            key = f"{slot}:{m['weights']}"
            a = agg.setdefault(key, {"slot": slot, "weights": m["weights"], "clips": 0, "clips_with_gt": 0,
                                     "correct": 0, "total_abs_error": 0, "err_in": 0, "err_out": 0,
                                     "fps_sum": 0.0, "uncounted": 0, "id_switches": 0, "reasons": {}})
            a["clips"] += 1
            a["fps_sum"] += m["fps"]
            a["uncounted"] += m["uncounted"]
            a["id_switches"] += m["id_switches"]
            for k, v in m["reasons"].items():
                a["reasons"][k] = a["reasons"].get(k, 0) + v
            if m["gt"]:
                a["clips_with_gt"] += 1
                a["correct"] += int(m["gt"]["correct"])
                a["total_abs_error"] += m["gt"]["abs_error"]
                a["err_in"] += m["gt"]["err_in"]
                a["err_out"] += m["gt"]["err_out"]
    for a in agg.values():
        a["avg_fps"] = round(a.pop("fps_sum") / a["clips"], 1) if a["clips"] else 0
        a["accuracy"] = round(a["correct"] / a["clips_with_gt"], 3) if a["clips_with_gt"] else None
    both_wrong = only_a = only_b = both_right = 0
    for row in rows:
        ga, gb = (row["models"].get(k, {}).get("gt") for k in ("A", "B"))
        if ga and gb:
            if not ga["correct"] and not gb["correct"]:
                both_wrong += 1
            elif not ga["correct"]:
                only_a += 1
            elif not gb["correct"]:
                only_b += 1
            else:
                both_right += 1
    return {"rows": rows, "models": list(agg.values()),
            "agreement": {"both_right": both_right, "both_wrong": both_wrong, "only_a_wrong": only_a, "only_b_wrong": only_b},
            "note": "FPS is laptop PyTorch speed, not Hailo-8L device speed."}


@app.get("/report.csv")
def report_csv(s: Session = Depends(get_session)):
    rows = _report_rows(s)
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["clip_id", "name", "duration_s", "gt_in", "gt_out", "gt_note",
                 "A_weights", "A_in", "A_out", "A_abs_error", "A_correct", "A_fps", "A_uncounted", "A_id_switches",
                 "B_weights", "B_in", "B_out", "B_abs_error", "B_correct", "B_fps", "B_uncounted", "B_id_switches"])
    for r in rows:
        line = [r["clip_id"], r["name"], r["duration"], r["gt_in"], r["gt_out"], r["gt_note"] or ""]
        for slot in ("A", "B"):
            m = r["models"].get(slot)
            if not m:
                line += [""] * 8
                continue
            g = m["gt"] or {}
            line += [m["weights"], m["in"], m["out"], g.get("abs_error", ""), g.get("correct", ""), m["fps"],
                     m["uncounted"], m["id_switches"]]
        wr.writerow(line)
    return Response(buf.getvalue(), media_type="text/csv",
                    headers={"Content-Disposition": 'attachment; filename="apcs_report.csv"'})


# ------------------------------------------------------------------ static frontend (production build)
if FRONTEND_DIST.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
