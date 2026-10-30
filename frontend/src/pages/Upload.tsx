import { useEffect, useRef, useState } from "react";
import { Link } from "react-router-dom";
import { api, modelLabel, urls, type Clip, type Job, type Preset } from "../api";
import { useSystem } from "../App";
import { ErrorBox, ProgressBar, SlotTag, StatusPill } from "../components/ui";

const EXT = [".mp4", ".avi", ".mov", ".mkv"];

export default function UploadPage() {
  const sys = useSystem();
  const [files, setFiles] = useState<File[]>([]);
  const [drag, setDrag] = useState(false);
  const [presets, setPresets] = useState<Preset[]>([]);
  const [slotA, setSlotA] = useState("");
  const [slotB, setSlotB] = useState("");
  const [presetId, setPresetId] = useState<number | null>(null);
  const [conf, setConf] = useState(0.25);
  const [iou, setIou] = useState(0.5);
  const [imgsz, setImgsz] = useState(640);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [clips, setClips] = useState<Record<number, Clip>>({});
  const [pending, setPending] = useState<Clip[]>([]);
  const sockets = useRef<Record<number, WebSocket>>({});

  useEffect(() => {
    api.presets().then(setPresets).catch(() => {});
    api.jobs().then((js) => {
      setJobs(js);
      js.filter((j) => j.status === "queued" || j.status === "running").forEach(watch);
    });
    api.clips().then((cs) => {
      setClips(Object.fromEntries(cs.map((c) => [c.id, c])));
      setPending(cs.filter((c) => c.status === "uploaded" || c.status === "error"));
    });
    const socks = sockets.current;
    return () => Object.values(socks).forEach((s) => s.close());
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (sys && !slotA) {
      setSlotA(sys.default_slots.A);
      setSlotB(sys.default_slots.B);
    }
  }, [sys, slotA]);

  function watch(job: Job) {
    if (sockets.current[job.id]) return;
    const ws = new WebSocket(urls.ws(job.id));
    sockets.current[job.id] = ws;
    ws.onmessage = (ev) => {
      const j = JSON.parse(ev.data) as Job;
      if (!j.id) return;
      setJobs((prev) => [j, ...prev.filter((p) => p.id !== j.id)].sort((a, b) => b.id - a.id));
      if (j.status === "done" || j.status === "error") {
        api.clips().then((cs) => setClips(Object.fromEntries(cs.map((c) => [c.id, c]))));
      }
    };
    ws.onclose = () => delete sockets.current[job.id];
  }

  function addFiles(list: FileList | null) {
    if (!list) return;
    const ok = Array.from(list).filter((f) => EXT.some((e) => f.name.toLowerCase().endsWith(e)));
    setFiles((prev) => [...prev, ...ok]);
    if (ok.length < list.length) setError(`Skipped ${list.length - ok.length} file(s): only ${EXT.join(", ")} supported`);
  }

  async function start(clipIds: number[]) {
    const job = await api.createJob({ clip_ids: clipIds, slot_a: slotA, slot_b: slotB, preset_id: presetId, conf, iou, imgsz });
    setJobs((prev) => [job, ...prev]);
    watch(job);
  }

  async function uploadAndStart() {
    setBusy(true);
    setError(null);
    try {
      const res = await api.upload(files);
      const ok = res.filter((r): r is Clip => "id" in r && r.id != null);
      const bad = res.filter((r) => "error" in r && r.error && !("id" in r && r.id));
      if (bad.length) setError(bad.map((b) => `${b.name}: ${(b as { error: string }).error}`).join("\n"));
      setClips((prev) => ({ ...prev, ...Object.fromEntries(ok.map((c) => [c.id, c])) }));
      setFiles([]);
      if (ok.length) await start(ok.map((c) => c.id));
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  }

  async function rerunPending() {
    setError(null);
    try {
      await start(pending.map((c) => c.id));
      setPending([]);
    } catch (e) {
      setError(String((e as Error).message));
    }
  }

  const weights = sys?.weights ?? [];

  return (
    <div className="mx-auto grid max-w-6xl grid-cols-1 gap-4 lg:grid-cols-[1fr_340px]">
      <div className="space-y-4">
        <div
          onDragOver={(e) => {
            e.preventDefault();
            setDrag(true);
          }}
          onDragLeave={() => setDrag(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDrag(false);
            addFiles(e.dataTransfer.files);
          }}
          className={`flex h-44 flex-col items-center justify-center rounded-lg border-2 border-dashed ${drag ? "border-sky-500 bg-sky-950/30" : "border-zinc-700 bg-zinc-900/50"}`}
        >
          <div className="text-sm text-zinc-300">Drop bus-door clips here</div>
          <div className="mt-1 text-xs text-zinc-500">{EXT.join(" · ")} — multiple files OK</div>
          <label className="btn mt-3 cursor-pointer">
            Browse…
            <input type="file" multiple accept={EXT.join(",")} className="hidden" onChange={(e) => addFiles(e.target.files)} />
          </label>
        </div>

        {files.length > 0 && (
          <div className="card">
            <div className="label mb-2">Selected ({files.length})</div>
            <ul className="space-y-1 text-xs">
              {files.map((f, i) => (
                <li key={i} className="flex justify-between">
                  <span className="truncate">{f.name}</span>
                  <span className="flex gap-3 text-zinc-500">
                    {(f.size / 1e6).toFixed(1)} MB
                    <button className="text-zinc-500 hover:text-red-400" onClick={() => setFiles(files.filter((_, k) => k !== i))}>✕</button>
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
        <ErrorBox error={error} />

        <div className="card">
          <div className="label mb-2">Queue</div>
          {jobs.length === 0 && <div className="text-xs text-zinc-500">No jobs yet.</div>}
          <div className="space-y-3">
            {jobs.map((j) => (
              <div key={j.id} className="rounded border border-zinc-800 p-2">
                <div className="mb-1 flex items-center gap-2 text-xs">
                  <span className="font-mono text-zinc-400">job #{j.id}</span>
                  <StatusPill status={j.status} />
                  <SlotTag slot="A">{modelLabel(j.slot_a)}</SlotTag>
                  <span className="text-zinc-600">vs</span>
                  <SlotTag slot="B">{modelLabel(j.slot_b)}</SlotTag>
                  <span className="text-zinc-500">conf {j.settings.conf} · iou {j.settings.iou} · {j.settings.imgsz}px</span>
                  <span className="ml-auto font-mono">{j.progress.toFixed(0)}%</span>
                </div>
                <ProgressBar value={j.progress} status={j.status} />
                {j.message && <div className="mt-1 text-xs text-red-400">{j.message}</div>}
                <div className="mt-2 space-y-1">
                  {j.clip_ids.map((cid) => {
                    const st = j.clips[String(cid)];
                    const c = clips[cid];
                    return (
                      <div key={cid} className="grid grid-cols-[40px_1fr_90px_160px] items-center gap-2 text-xs">
                        <img src={urls.thumb(cid)} className="h-6 w-10 rounded object-cover" alt="" />
                        <Link to={`/clips/${cid}`} className="truncate hover:text-sky-400">{c?.name ?? `clip ${cid}`}</Link>
                        <span className="text-zinc-500">{st?.stage ?? (j.status === "done" ? "done" : "—")}</span>
                        <ProgressBar value={st?.progress ?? (j.status === "done" ? 100 : 0)} status={st?.status} />
                      </div>
                    );
                  })}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>

      <div className="space-y-4">
        <div className="card space-y-3">
          <div className="label">Model slots</div>
          {(["A", "B"] as const).map((slot) => (
            <label key={slot} className="flex items-center gap-2">
              <SlotTag slot={slot} />
              <select className="input flex-1" value={slot === "A" ? slotA : slotB} onChange={(e) => (slot === "A" ? setSlotA : setSlotB)(e.target.value)}>
                {weights.map((w) => (
                  <option key={w} value={w}>{w}</option>
                ))}
              </select>
            </label>
          ))}
          <div className="text-[11px] text-zinc-500">Add custom .pt files to backend/weights/ and reload.</div>

          <div className="label pt-2">ROI</div>
          <select className="input w-full" value={presetId ?? ""} onChange={(e) => setPresetId(e.target.value ? Number(e.target.value) : null)}>
            <option value="">Active ROI (shared by all clips)</option>
            {presets.map((p) => (
              <option key={p.id} value={p.id}>{p.name}</option>
            ))}
          </select>

          <div className="label pt-2">Inference</div>
          <div className="grid grid-cols-3 gap-2">
            <label className="text-xs text-zinc-400">conf<input type="number" step={0.05} min={0.01} max={0.99} className="input w-full" value={conf} onChange={(e) => setConf(Number(e.target.value))} /></label>
            <label className="text-xs text-zinc-400">IoU<input type="number" step={0.05} min={0.05} max={0.95} className="input w-full" value={iou} onChange={(e) => setIou(Number(e.target.value))} /></label>
            <label className="text-xs text-zinc-400">imgsz<select className="input w-full" value={imgsz} onChange={(e) => setImgsz(Number(e.target.value))}>{[320, 416, 480, 640, 800, 960, 1280].map((s) => <option key={s}>{s}</option>)}</select></label>
          </div>
          <div className="text-[11px] text-zinc-500">Class: person only. Inference is cached per clip/model/settings.</div>

          <button className="btn-primary w-full py-2" disabled={!files.length || busy || !sys?.models_ready} onClick={uploadAndStart}>
            {busy ? "Uploading…" : `Upload & run ${files.length || ""} clip${files.length === 1 ? "" : "s"}`}
          </button>
          {!sys?.models_ready && <div className="text-[11px] text-amber-400">Waiting for models to load…</div>}
        </div>

        {pending.length > 0 && (
          <div className="card space-y-2">
            <div className="label">Not yet processed ({pending.length})</div>
            <ul className="max-h-32 space-y-0.5 overflow-auto text-xs text-zinc-400">
              {pending.map((c) => <li key={c.id} className="truncate">{c.name} {c.error && <span className="text-red-400">— {c.error}</span>}</li>)}
            </ul>
            <button className="btn w-full" onClick={rerunPending}>Run these with current settings</button>
          </div>
        )}
      </div>
    </div>
  );
}
