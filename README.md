# APCS Model Comparator

Local web app that runs **two YOLOv8-pose models** (default `yolov8s-pose` vs `yolov8m-pose`) on the same
bus-door clips, tracks people with ByteTrack, counts IN/OUT across an editable ROI + counting line, and shows
side by side where each model succeeds or fails.

- Backend: FastAPI + Ultralytics + OpenCV + SQLite (`backend/`)
- Frontend: React + Vite + TypeScript + Tailwind + react-konva + Recharts (`frontend/`)
- Device is auto-selected: CUDA → Apple MPS → CPU (shown in the header).
- One GPU worker: jobs run one at a time, so the two models never run in parallel.

> FPS shown in the UI is **laptop PyTorch speed**, not Hailo-8L speed on the device.

## Setup

Requirements: Python 3.10+, Node.js 18+. No system ffmpeg is needed (the bundled `imageio-ffmpeg` binary is used).

```bash
# backend
cd backend
python -m venv .venv
# Windows: .venv\Scripts\activate      Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt

# frontend
cd ../frontend
npm install
```

**NVIDIA GPU:** `pip install ultralytics` installs a CPU build of PyTorch on Windows. For CUDA, install the
matching wheel first, e.g. `pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124`
(see pytorch.org for your CUDA version), then `pip install -r requirements.txt`.

## Run

- Windows: double-click `run.bat`
- Linux/macOS: `./run.sh`

Both start the backend on http://127.0.0.1:8000 and the frontend on http://localhost:5173. The first start
downloads `yolov8s-pose.pt` and `yolov8m-pose.pt` into `backend/weights/`.

Manual:
```bash
cd backend && .venv/Scripts/python -m uvicorn app.main:app --port 8000   # (.venv/bin/python on Linux)
cd frontend && npm run dev
```
Alternatively, `npm run build` in `frontend/` and open http://127.0.0.1:8000 (FastAPI serves `frontend/dist`).

Tests: `cd backend && python -m pytest`

Test clip: `python backend/scripts/make_sample_clip.py` writes `data/sample_bus.mp4`, a synthetic 10 s
clip in which 4 people move down across the middle line (ground truth: 4 IN / 0 OUT).

## Workflow

1. **Upload**: drop clips, pick the model in slot A and slot B, an optional ROI preset, and conf/IoU/imgsz, then run.
   Progress is pushed over a WebSocket.
2. **Clips**: type the ground-truth IN/OUT inline. Filter by "both wrong", "only A wrong", or "only B wrong".
3. **Comparison**: the two annotated videos play in sync (Space = play/pause, ←/→ = 1 frame, Shift = 10).
   Click the timeline or any table row to seek. Orange boxes are tracks that were detected but never counted.
4. **ROI editor**: there is one **active ROI** shared by every clip. Drag the ROI vertices and line endpoints.
   Double-click a vertex to delete it; click a ◇ midpoint to insert one. Flip the IN side, edit parameters, and
   press **Apply to ALL clips**. The ROI is saved, every processed clip is recounted from cached inference (the
   current clip immediately, the rest in the background), and **new uploads use it automatically**. "Preview on
   this clip only" tries a change on one clip without changing the active ROI. Choosing a preset on the Upload
   page overrides the active ROI for that job only.
5. **Batch report**: accuracy against ground truth, count error, FPS, and failure reasons, with CSV/JSON export.

Inference results are cached at `data/outputs/{clip_id}/{weights}_raw.json`. A clip is only re-inferred
when the weights or conf/IoU/imgsz change.

## Custom weights

Copy any `.pt` file (e.g. `my-retrained-pose.pt`) into `backend/weights/` and reload the page. It appears in
the slot dropdowns. Pose models are recommended. Detection-only models still work, but they fall back to bbox anchors.

## Exporting the ROI for the device

ROI editor → **Export preset JSON** (or **Export clip ROI** for the last applied ROI of the clip), or
`GET /roi-presets/{id}/export?clip_id=N` (or `?width=1280&height=720`). The file contains:

```json
{
  "format": "apcs-roi/v1",
  "normalized": { "roi": [[x,y],...], "line": [[x1,y1],[x2,y2]], "in_side": 1 },
  "resolution": { "width": 640, "height": 720 },
  "pixels":     { "roi": [[32,36],...], "line": [[32,360],[608,360]], "in_side": 1, "hysteresis_px": 21.6 },
  "params": { "anchor": "hip_fallback", "hysteresis": 0.03, "min_track_age": 3, ... }
}
```

`in_side` is the sign of `cross(line[1]-line[0], p-line[0])` in image coordinates (y pointing down) for points
inside the bus. **Import…** accepts the same file.

## Counting logic (`backend/app/counting.py`)

- Anchor point: mid-hip (keypoints 11/12, both conf > `kp_conf`) with a bbox bottom-center fallback, or bbox
  center, or bbox bottom-center only.
- **Anchor stabilization** (on by default): once a track's hips have been seen, frames with weak hip keypoints
  use the hip position estimated from the bbox instead of jumping to the feet. Without it, a hip confidence that
  hovers around the threshold makes the anchor jump about half a body height every frame, which shows up as
  phantom IN/OUT counts (found on the sample clip with yolov8m).
- Only anchors inside the ROI are considered. A crossing is counted when the anchor leaves the hysteresis band
  on the opposite side, after `min_track_age` frames, once per track per direction.
- Lost-track recovery merges a vanished track with a new ID that appears within N frames and D px of where it
  vanished. Merges are logged.
- Uncounted tracks (≥ X frames) get a reason: never crossed line / too short-lived / outside ROI /
  jitter-suppressed.

## API

`POST /clips/upload`, `GET /clips`, `GET|DELETE /clips/{id}`, `POST /clips/{id}/ground-truth`,
`POST /jobs`, `GET /jobs`, `GET /jobs/{id}`, `WS /ws/jobs/{id}`, `POST /clips/{id}/recount`,
`GET|POST /roi-presets`, `DELETE /roi-presets/{id}`, `GET /roi-presets/{id}/export`,
`POST /roi-presets/{id}/apply-all`, `GET /clips/{id}/video/{A|B}` (HTTP range), `GET /clips/{id}/results`,
`GET /clips/{id}/frame?i=N`, `GET /clips/{id}/thumbnail`, `GET /clips/{id}/roi-export`, `GET /report`,
`GET /report.csv`, `GET /system`. Interactive docs: http://127.0.0.1:8000/docs

The frontend uses hash routes (`/#/clips/1`), so these API paths never collide with page URLs.

## Troubleshooting

- **Out of memory**: the job reports it and suggests a lower `imgsz` (480/320). The worker keeps running.
- **Corrupt/unsupported video**: rejected at upload with a message.
- **Server restarted mid-job**: the affected clips are marked `error (interrupted)`. Re-run them from the Upload page.
- Annotated videos are downscaled to 960 px wide to keep re-renders fast. Override with `APCS_RENDER_MAX_WIDTH`.
