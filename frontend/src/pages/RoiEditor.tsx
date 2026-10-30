import { useEffect, useRef, useState, type ReactNode } from "react";
import { Arrow, Circle, Image as KImage, Layer, Line, Rect, Stage, Text } from "react-konva";
import type { KonvaEventObject } from "konva/lib/Node";
import { Link, useParams } from "react-router-dom";
import { api, modelLabel, urls, type Clip, type CountParams, type Preset, type Pt, type Results, type RoiConfig } from "../api";
import { useSystem } from "../App";
import { Check, ErrorBox, SlotTag } from "../components/ui";

type Mode = "edit" | "add";

const clamp01 = (v: number) => Math.min(1, Math.max(0, v));

function useImage(src: string) {
  const [img, setImg] = useState<HTMLImageElement | null>(null);
  useEffect(() => {
    const i = new window.Image();
    i.onload = () => setImg(i);
    i.src = src;
  }, [src]);
  return img;
}

function useWidth<T extends HTMLElement>() {
  const ref = useRef<T>(null);
  const [w, setW] = useState(800);
  useEffect(() => {
    if (!ref.current) return;
    const ro = new ResizeObserver(([e]) => setW(e.contentRect.width));
    ro.observe(ref.current);
    return () => ro.disconnect();
  }, []);
  return [ref, w] as const;
}

export default function RoiEditorPage() {
  const id = Number(useParams().id);
  const sys = useSystem();
  const [clip, setClip] = useState<Clip | null>(null);
  const [cfg, setCfg] = useState<RoiConfig | null>(null);
  const [frameIdx, setFrameIdx] = useState(0);
  const [frameSrcIdx, setFrameSrcIdx] = useState(0);
  const [mode, setMode] = useState<Mode>("edit");
  const [presets, setPresets] = useState<Preset[]>([]);
  const [presetId, setPresetId] = useState<number | null>(null);
  const [presetName, setPresetName] = useState("");
  const [res, setRes] = useState<Results | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [info, setInfo] = useState<string | null>(null);
  const [dirty, setDirty] = useState(false);
  const [wrapRef, wrapW] = useWidth<HTMLDivElement>();

  useEffect(() => {
    api.clip(id).then((c) => {
      setClip(c);
      setPresetId(c.preset_id);
    });
    // edit the active ROI (shared by every clip), drawn over this clip's frames
    api.activeRoi().then(setCfg).catch(() => {});
    api.presets().then(setPresets);
    api.results(id).then(setRes).catch(() => {});
  }, [id]);
  useEffect(() => {
    if (!cfg && clip && sys) setCfg(sys.defaults.roi);
  }, [cfg, clip, sys]);
  // debounce frame fetches while scrubbing
  useEffect(() => {
    const t = setTimeout(() => setFrameSrcIdx(frameIdx), 120);
    return () => clearTimeout(t);
  }, [frameIdx]);

  const img = useImage(urls.frame(id, frameSrcIdx));

  if (!clip || !cfg) return <div className="text-zinc-500">Loading…</div>;
  const W = clip.width || img?.width || 1280;
  const H = clip.height || img?.height || 720;
  const scale = Math.min(wrapW / W, 640 / H);
  const sw = W * scale, sh = H * scale;
  const toS = (p: Pt): [number, number] => [p[0] * sw, p[1] * sh];
  const fromS = (x: number, y: number): Pt => [clamp01(x / sw), clamp01(y / sh)];

  const update = (patch: Partial<RoiConfig>) => {
    setCfg({ ...cfg, ...patch });
    setDirty(true);
  };
  const setParam = <K extends keyof CountParams>(k: K, v: CountParams[K]) => update({ params: { ...cfg.params, [k]: v } });

  const onStageClick = (e: KonvaEventObject<MouseEvent>) => {
    if (mode !== "add") return;
    const pos = e.target.getStage()?.getPointerPosition();
    if (pos) update({ roi: [...cfg.roi, fromS(pos.x, pos.y)] });
  };

  // line geometry in screen space
  const [a, b] = [toS(cfg.line[0]), toS(cfg.line[1])];
  const dx = b[0] - a[0], dy = b[1] - a[1];
  const len = Math.hypot(dx, dy) || 1;
  // normal toward the IN side: sign(cross(B-A, P-A)) == in_side
  const nx = (-dy / len) * cfg.in_side, ny = (dx / len) * cfg.in_side;
  const mid = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2];
  const band = cfg.params.hysteresis * sh;
  const arrowLen = Math.max(30, 0.1 * sh);

  async function apply(scope: "all" | "clip") {
    setBusy(true);
    setError(null);
    try {
      const r = await api.recount(id, cfg!, presetId, scope);
      setRes(r);
      setDirty(scope === "clip");
      const others = r.recount_scheduled?.length ?? 0;
      setInfo(
        scope === "all"
          ? `Saved as the active ROI for all clips (incl. future uploads). This clip recounted; ${others} other clip(s) recounting in the background.`
          : "Preview on this clip only — other clips and future uploads still use the active ROI.",
      );
    } catch (e) {
      setError(String((e as Error).message));
    } finally {
      setBusy(false);
    }
  }

  async function savePreset() {
    const name = presetName.trim() || presets.find((p) => p.id === presetId)?.name;
    if (!name) return setError("Enter a preset name");
    try {
      const p = await api.savePreset({ name, ...cfg });
      setPresets(await api.presets());
      setPresetId(p.id);
      setPresetName("");
      setInfo(`Saved preset "${p.name}"`);
    } catch (e) {
      setError(String((e as Error).message));
    }
  }

  function loadPreset(pid: number | null) {
    setPresetId(pid);
    const p = presets.find((x) => x.id === pid);
    if (p) {
      setCfg({ ...p.config, params: { ...sys!.defaults.count_params, ...p.config.params } });
      setDirty(true);
    }
  }

  async function importPreset(f: File | undefined) {
    if (!f) return;
    try {
      const doc = JSON.parse(await f.text());
      const p = await api.savePreset({ ...doc, name: doc.name || f.name.replace(/\.(roi\.)?json$/, "") });
      setPresets(await api.presets());
      loadPreset(p.id);
      setCfg({ ...p.config });
      setInfo(`Imported preset "${p.name}"`);
    } catch (e) {
      setError(`Import failed: ${(e as Error).message}`);
    }
  }

  const roiFlat = cfg.roi.flatMap(toS);

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-3">
        <Link to={`/clips/${id}`} className="text-zinc-500 hover:text-zinc-300">← Comparison</Link>
        <h1 className="text-base font-semibold text-white">ROI editor <span className="font-normal text-zinc-400">· shared by all clips · previewing on {clip.name}</span></h1>
        <span className="text-xs text-zinc-500">{W}×{H}</span>
      </div>
      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_360px]">
        <div className="space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            <button className={`btn ${mode === "edit" ? "!bg-sky-700" : ""}`} onClick={() => setMode("edit")}>Edit / drag</button>
            <button className={`btn ${mode === "add" ? "!bg-sky-700" : ""}`} onClick={() => setMode("add")}>Add ROI points (click)</button>
            <button className="btn" onClick={() => { update({ roi: [] }); setMode("add"); }}>Redraw ROI</button>
            <button className="btn" onClick={() => update({ in_side: cfg.in_side === 1 ? -1 : 1 })}>⇅ Flip IN side</button>
            <button className="btn" onClick={() => update({ line: [cfg.line[1], cfg.line[0]], in_side: cfg.in_side === 1 ? -1 : 1 })} title="Swap endpoints (same IN side)">Swap line ends</button>
            <span className="text-zinc-500">Drag vertices · double-click a vertex to delete · click a ◇ midpoint to insert</span>
          </div>
          <div ref={wrapRef} className="w-full">
            <Stage width={sw} height={sh} onClick={onStageClick} className={`rounded border border-zinc-800 ${mode === "add" ? "cursor-crosshair" : ""}`}>
              <Layer>
                {img ? <KImage image={img} width={sw} height={sh} /> : <Rect width={sw} height={sh} fill="#111" />}
                {cfg.roi.length >= 2 && (
                  <Line points={roiFlat} closed={cfg.roi.length >= 3} stroke="#00c8ff" strokeWidth={2} fill="rgba(0,200,255,0.10)" listening={false} />
                )}
                {/* hysteresis band */}
                {[-band, band].map((o) => (
                  <Line key={o} points={[a[0] + nx * o, a[1] + ny * o, b[0] + nx * o, b[1] + ny * o]} stroke="#a1a1aa" strokeWidth={1} dash={[4, 4]} listening={false} />
                ))}
                <Line points={[...a, ...b]} stroke="#ffffff" strokeWidth={3} listening={false} />
                <Arrow points={[mid[0], mid[1], mid[0] + nx * arrowLen, mid[1] + ny * arrowLen]} stroke="#4ade80" fill="#4ade80" strokeWidth={3} pointerLength={10} pointerWidth={10} listening={false} />
                <Text x={mid[0] + nx * arrowLen * 1.3 - 10} y={mid[1] + ny * arrowLen * 1.3 - 7} text="IN" fill="#4ade80" fontStyle="bold" fontSize={14} listening={false} />
                <Text x={mid[0] - nx * arrowLen * 0.9 - 14} y={mid[1] - ny * arrowLen * 0.9 - 7} text="OUT" fill="#f87171" fontStyle="bold" fontSize={12} listening={false} />

                {/* midpoint insert handles */}
                {mode === "edit" && cfg.roi.length >= 3 &&
                  cfg.roi.map((p, i) => {
                    const q = cfg.roi[(i + 1) % cfg.roi.length];
                    const [mx, my] = toS([(p[0] + q[0]) / 2, (p[1] + q[1]) / 2]);
                    return (
                      <Rect key={`m${i}`} x={mx - 4} y={my - 4} width={8} height={8} rotation={0} fill="#0e7490" stroke="#67e8f9" strokeWidth={1}
                        onClick={(e) => { e.cancelBubble = true; const r = [...cfg.roi]; r.splice(i + 1, 0, [(p[0] + q[0]) / 2, (p[1] + q[1]) / 2]); update({ roi: r }); }} />
                    );
                  })}
                {/* ROI vertices */}
                {cfg.roi.map((p, i) => {
                  const [x, y] = toS(p);
                  return (
                    <Circle key={`v${i}`} x={x} y={y} radius={6} fill="#00c8ff" stroke="#000" strokeWidth={1} draggable
                      onDragMove={(e) => { const r = [...cfg.roi]; r[i] = fromS(e.target.x(), e.target.y()); update({ roi: r }); }}
                      onDblClick={(e) => { e.cancelBubble = true; update({ roi: cfg.roi.filter((_, k) => k !== i) }); }}
                      onClick={(e) => (e.cancelBubble = true)} />
                  );
                })}
                {/* line endpoints */}
                {[0, 1].map((k) => {
                  const [x, y] = toS(cfg.line[k]);
                  return (
                    <Circle key={`l${k}`} x={x} y={y} radius={8} fill="#fff" stroke="#000" strokeWidth={2} draggable
                      onDragMove={(e) => { const l = [...cfg.line] as [Pt, Pt]; l[k] = fromS(e.target.x(), e.target.y()); update({ line: l }); }}
                      onClick={(e) => (e.cancelBubble = true)} />
                  );
                })}
              </Layer>
            </Stage>
          </div>
          <div className="flex items-center gap-2 text-xs">
            <span className="label">Frame</span>
            <input type="range" min={0} max={Math.max(0, clip.frame_count - 1)} value={frameIdx} onChange={(e) => setFrameIdx(Number(e.target.value))} className="flex-1 accent-sky-500" />
            <span className="w-28 text-right font-mono">{frameIdx} · {(frameIdx / clip.fps).toFixed(2)}s</span>
          </div>
          <div className="font-mono text-[11px] text-zinc-500">
            ROI ({cfg.roi.length} pts): {cfg.roi.map((p) => `(${Math.round(p[0] * W)},${Math.round(p[1] * H)})`).join(" ")} · line: ({Math.round(cfg.line[0][0] * W)},{Math.round(cfg.line[0][1] * H)})→({Math.round(cfg.line[1][0] * W)},{Math.round(cfg.line[1][1] * H)}) · in_side {cfg.in_side}
          </div>
        </div>

        <div className="space-y-3">
          <div className="card space-y-2">
            <div className="flex items-center">
              <div className="label">Counting parameters</div>
              {dirty && <span className="ml-auto text-[11px] text-amber-400">unapplied changes</span>}
            </div>
            <Field label="Anchor point">
              <select className="input w-full" value={cfg.params.anchor} onChange={(e) => setParam("anchor", e.target.value as CountParams["anchor"])}>
                <option value="hip_fallback">mid-hip (kp 11/12), fallback bbox bottom</option>
                <option value="bbox_bottom">bbox bottom-center only</option>
                <option value="bbox_center">bbox center</option>
              </select>
            </Field>
            {cfg.params.anchor === "hip_fallback" && (
              <>
                <Num label="Hip keypoint min conf" v={cfg.params.kp_conf} step={0.05} on={(v) => setParam("kp_conf", v)} />
                <Bool label="Stabilize anchor (estimate hips from bbox when keypoints drop)" v={cfg.params.anchor_stabilize} on={(v) => setParam("anchor_stabilize", v)} />
              </>
            )}
            <Num label="Hysteresis band (% of frame height)" v={+(cfg.params.hysteresis * 100).toFixed(2)} step={0.5} on={(v) => setParam("hysteresis", v / 100)} />
            <Num label="Min track age before counting (frames)" v={cfg.params.min_track_age} step={1} on={(v) => setParam("min_track_age", Math.round(v))} />
            <Bool label="Count IN and OUT on the same track" v={cfg.params.count_both_directions} on={(v) => setParam("count_both_directions", v)} />
            <Bool label="Lost-track recovery (merge IDs)" v={cfg.params.recovery_enabled} on={(v) => setParam("recovery_enabled", v)} />
            {cfg.params.recovery_enabled && (
              <div className="grid grid-cols-2 gap-2">
                <Num label="within N frames" v={cfg.params.recovery_frames} step={1} on={(v) => setParam("recovery_frames", Math.round(v))} />
                <Num label="within D px" v={cfg.params.recovery_px} step={5} on={(v) => setParam("recovery_px", v)} />
              </div>
            )}
            <Num label="Report uncounted tracks with ≥ frames" v={cfg.params.uncounted_min_frames} step={1} on={(v) => setParam("uncounted_min_frames", Math.round(v))} />
            <button className="btn-primary w-full py-2" disabled={busy || cfg.roi.length < 3} onClick={() => apply("all")}>
              {busy ? "Recounting…" : "Apply to ALL clips (and future uploads)"}
            </button>
            <button className="w-full text-[11px] text-zinc-500 hover:text-zinc-300 disabled:opacity-40" disabled={busy || cfg.roi.length < 3} onClick={() => apply("clip")}>
              or preview on this clip only
            </button>
            {cfg.roi.length < 3 && <div className="text-[11px] text-amber-400">ROI needs at least 3 points.</div>}
          </div>

          {res && (
            <div className="card">
              <div className="label mb-2">Counts {clip.gt_in != null && <span className="normal-case text-zinc-400">· GT {clip.gt_in} in / {clip.gt_out} out</span>}</div>
              <div className="grid grid-cols-2 gap-2">
                {(["A", "B"] as const).map((slot) => {
                  const m = res.models[slot];
                  return (
                    <div key={slot} className="rounded bg-zinc-950 p-2 text-center">
                      <SlotTag slot={slot}>{modelLabel(m?.weights)}</SlotTag>
                      <div className="font-mono text-xl"><span className="text-emerald-400">{m?.in ?? "–"}</span> / <span className="text-red-400">{m?.out ?? "–"}</span></div>
                      <div className="text-xs"><Check ok={m?.gt?.correct} /> <span className="text-orange-400">{m?.uncounted.length ?? 0} uncounted</span></div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
          <ErrorBox error={error} />
          {info && <div className="text-[11px] text-sky-300">{info}</div>}

          <div className="card space-y-2">
            <div className="label">Presets</div>
            <div className="flex gap-2">
              <select className="input flex-1" value={presetId ?? ""} onChange={(e) => loadPreset(e.target.value ? Number(e.target.value) : null)}>
                <option value="">(none)</option>
                {presets.map((p) => <option key={p.id} value={p.id}>{p.name}</option>)}
              </select>
              <button className="btn" disabled={!presetId} onClick={async () => { if (presetId && confirm("Delete preset?")) { await api.deletePreset(presetId); setPresets(await api.presets()); setPresetId(null); } }}>Delete</button>
            </div>
            <div className="flex gap-2">
              <input className="input flex-1" placeholder='new name, e.g. "Bus front door"' value={presetName} onChange={(e) => setPresetName(e.target.value)} />
              <button className="btn" onClick={savePreset}>{presetName.trim() ? "Save as" : "Save / overwrite"}</button>
            </div>
            <div className="text-[11px] text-zinc-500">Selecting a preset loads it into the editor; press Apply to use it for all clips.</div>
            <div className="flex gap-2">
              <a className={`btn flex-1 text-center ${presetId ? "" : "pointer-events-none opacity-40"}`} href={presetId ? urls.presetExport(presetId, id) : undefined} download>Export preset JSON</a>
              <a className="btn flex-1 text-center" href={urls.activeRoiExport(id)} download title="Exports the last APPLIED active ROI, with pixels at this clip's resolution">Export active ROI</a>
              <label className="btn flex-1 cursor-pointer text-center">
                Import…
                <input type="file" accept=".json" className="hidden" onChange={(e) => { importPreset(e.target.files?.[0]); e.target.value = ""; }} />
              </label>
            </div>
            <div className="text-[11px] text-zinc-500">Export contains normalized and pixel coordinates (at {W}×{H}) for the device.</div>
          </div>
        </div>
      </div>
    </div>
  );
}

function Field({ label, children }: { label: string; children: ReactNode }) {
  return (
    <label className="block text-xs text-zinc-400">
      <div className="mb-0.5">{label}</div>
      {children}
    </label>
  );
}

function Num({ label, v, step, on }: { label: string; v: number; step: number; on: (v: number) => void }) {
  return (
    <label className="flex items-center justify-between gap-2 text-xs text-zinc-400">
      <span>{label}</span>
      <input type="number" step={step} min={0} className="input w-20 text-right" value={v} onChange={(e) => on(Number(e.target.value))} />
    </label>
  );
}

function Bool({ label, v, on }: { label: string; v: boolean; on: (v: boolean) => void }) {
  return (
    <label className="flex items-center gap-2 text-xs text-zinc-400">
      <input type="checkbox" checked={v} onChange={(e) => on(e.target.checked)} />
      {label}
    </label>
  );
}

