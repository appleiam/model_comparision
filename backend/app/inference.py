"""Model loading, device selection and per-frame inference + ByteTrack."""
from __future__ import annotations

import logging
import platform
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .config import DEFAULT_SLOTS, DEFAULT_WEIGHTS, PERSON_CLASS, SLOTS, TRACKER_CFG, WEIGHTS_DIR

log = logging.getLogger("apcs.inference")

_device: Optional[str] = None
_models: dict[str, Any] = {}  # slot -> YOLO instance
_model_weights: dict[str, str] = {}  # slot -> weights filename
_lock = threading.Lock()


class InferenceError(RuntimeError):
    pass


def select_device() -> str:
    global _device
    if _device is None:
        import torch

        if torch.cuda.is_available():
            _device = "cuda:0"
        elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            _device = "mps"
        else:
            _device = "cpu"
    return _device


def system_info() -> dict[str, Any]:
    import torch

    dev = select_device()
    info: dict[str, Any] = {
        "device": dev,
        "device_name": platform.processor() or platform.machine(),
        "cuda_available": torch.cuda.is_available(),
        "cuda_name": None,
        "vram_total_mb": None,
        "vram_used_mb": None,
        "torch": torch.__version__,
        "platform": platform.platform(),
        "weights": list_weights(),
        "slots": dict(_model_weights),
        "default_slots": DEFAULT_SLOTS,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        info["cuda_name"] = props.name
        info["device_name"] = props.name
        info["vram_total_mb"] = round(props.total_memory / 2**20)
        info["vram_used_mb"] = round(torch.cuda.memory_allocated(0) / 2**20)
    elif dev == "mps":
        info["device_name"] = "Apple MPS"
    return info


def list_weights() -> list[str]:
    return sorted(p.name for p in WEIGHTS_DIR.glob("*.pt"))


def ensure_default_weights() -> None:
    """Download stock pose weights into backend/weights if missing."""
    from ultralytics import YOLO

    for name in DEFAULT_WEIGHTS:
        target = WEIGHTS_DIR / name
        if target.exists():
            continue
        log.info("Downloading %s ...", name)
        try:
            YOLO(str(target))  # ultralytics downloads known assets to this path
        except Exception as e:  # pragma: no cover - network dependent
            log.error("Could not download %s: %s", name, e)
            continue
        if not target.exists() and Path(name).exists():  # older versions download to CWD
            shutil.move(name, target)


def _load(weights: str):
    from ultralytics import YOLO

    path = WEIGHTS_DIR / weights
    if not path.exists():
        raise InferenceError(f"Weights not found: {path}")
    model = YOLO(str(path))
    if getattr(model, "task", None) != "pose":
        log.warning("%s is task=%s, keypoints unavailable (bbox anchors only)", weights, model.task)
    return model


def get_model(slot: str, weights: str):
    """Return a dedicated YOLO instance for this slot, (re)loading if weights changed.

    Each slot owns its own instance so ByteTrack state never mixes between models.
    """
    with _lock:
        if _model_weights.get(slot) != weights or slot not in _models:
            log.info("Loading slot %s <- %s", slot, weights)
            _models[slot] = _load(weights)
            _model_weights[slot] = weights
            _warmup(_models[slot])
        return _models[slot]


def _warmup(model) -> None:
    try:
        model.predict(np.zeros((320, 320, 3), dtype=np.uint8), device=select_device(), verbose=False, imgsz=320)
    except Exception as e:  # pragma: no cover
        log.warning("Warmup failed: %s", e)


def load_default_models() -> None:
    weights = list_weights()
    for slot in SLOTS:
        w = DEFAULT_SLOTS[slot]
        if w in weights:
            try:
                get_model(slot, w)
            except Exception as e:  # pragma: no cover
                log.error("Failed to load %s: %s", w, e)


def reset_tracker(model) -> None:
    predictor = getattr(model, "predictor", None)
    for t in getattr(predictor, "trackers", None) or []:
        t.reset()


def track_frame(model, frame: np.ndarray, conf: float, iou: float, imgsz: int) -> tuple[list[dict], float, float]:
    """Run detection + ByteTrack on one BGR frame.

    Returns (detections, model_inference_ms, wall_ms). Only tracked boxes are returned.
    """
    t0 = time.perf_counter()
    res = model.track(
        frame,
        persist=True,
        tracker=TRACKER_CFG,
        conf=conf,
        iou=iou,
        imgsz=imgsz,
        classes=[PERSON_CLASS],
        device=select_device(),
        verbose=False,
    )[0]
    wall_ms = (time.perf_counter() - t0) * 1000
    infer_ms = float(res.speed.get("inference", wall_ms)) if getattr(res, "speed", None) else wall_ms

    dets: list[dict] = []
    boxes = res.boxes
    if boxes is not None and boxes.id is not None and len(boxes):
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy().astype(int)
        kpts = None
        if getattr(res, "keypoints", None) is not None and res.keypoints.data is not None:
            kpts = res.keypoints.data.cpu().numpy()  # (n, 17, 3)
            if kpts.ndim != 3 or kpts.shape[0] != len(ids):
                kpts = None
        for i in range(len(ids)):
            d = {
                "id": int(ids[i]),
                "bbox": [round(float(v), 1) for v in xyxy[i]],
                "conf": round(float(confs[i]), 3),
                "kpts": None,
            }
            if kpts is not None:
                d["kpts"] = [[round(float(x), 1), round(float(y), 1), round(float(c), 3)] for x, y, c in kpts[i]]
            dets.append(d)
    return dets, infer_ms, wall_ms


def is_oom(e: BaseException) -> bool:
    msg = str(e).lower()
    return "out of memory" in msg or type(e).__name__ == "OutOfMemoryError"


def free_memory() -> None:
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
