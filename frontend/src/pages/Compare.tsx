import { useCallback, useEffect, useRef, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api, modelLabel, REASON_LABEL, urls, type ModelResult, type Results, type Slot } from "../api";
import SyncedVideos, { type SyncedHandle } from "../components/SyncedVideos";
import Timeline from "../components/Timeline";
import { Check, ErrorBox, SlotTag } from "../components/ui";

export default function ComparePage() {
  const id = Number(useParams().id);
  const [res, setRes] = useState<Results | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [frame, setFrame] = useState(0);
  const [shadeDets, setShadeDets] = useState(false);
  const player = useRef<SyncedHandle>(null);

  const load = useCallback(() => api.results(id).then(setRes).catch((e) => setError(String(e.message))), [id]);
  useEffect(() => {
    load();
  }, [load]);

  // poll while a re-render is in progress, then reload videos (render_version busts the cache)
  useEffect(() => {
    if (res?.clip.render_status !== "rendering") return;
    const t = setInterval(async () => {
      const c = await api.clip(id);
      if (c.render_status !== "rendering") load();
    }, 1500);
    return () => clearInterval(t);
  }, [res?.clip.render_status, id, load]);

  const seek = useCallback((f: number) => player.current?.seekFrame(f), []);

  if (error && !res) return <ErrorBox error={error} />;
  if (!res) return <div className="text-zinc-500">Loading…</div>;
  const clip = res.clip;
  const rendering = clip.render_status === "rendering";
  const srcs = rendering ? {} : { A: urls.video(id, "A", clip.render_version), B: urls.video(id, "B", clip.render_version) };

  return (
    <div className="space-y-3">
      <div className="flex flex-wrap items-center gap-3">
        <Link to="/clips" className="text-zinc-500 hover:text-zinc-300">← Clips</Link>
        <h1 className="text-base font-semibold text-white">{clip.name}</h1>
        <span className="text-xs text-zinc-500">
          #{clip.id} · {clip.width}×{clip.height} · {clip.fps.toFixed(2)} fps · {clip.frame_count} frames · {clip.duration}s
        </span>
        {rendering && <span className="animate-pulse text-xs text-amber-400">re-rendering videos…</span>}
        <div className="ml-auto flex gap-2">
          <Link to={`/clips/${id}/roi`} className="btn-primary">Edit ROI / line</Link>
        </div>
      </div>
      <ErrorBox error={error} />

      <div className="grid grid-cols-1 gap-3 xl:grid-cols-[1fr_330px]">
        <div className="space-y-3">
          <SyncedVideos
            ref={player}
            srcs={srcs}
            labels={{ A: modelLabel(res.models.A?.weights), B: modelLabel(res.models.B?.weights) }}
            fps={clip.fps}
            frameCount={clip.frame_count || res.models.A?.per_frame.n_dets.length || 1}
            onFrame={setFrame}
          />
          <div className="card">
            <div className="mb-1 flex items-center gap-3 text-xs">
              <span className="label">Detections per frame</span>
              <span className="text-zinc-500">solid = IN, dashed = OUT · <span className="text-red-400">red band</span> = models' running counts disagree · click to seek</span>
              <label className="ml-auto flex items-center gap-1 text-zinc-400">
                <input type="checkbox" checked={shadeDets} onChange={(e) => setShadeDets(e.target.checked)} /> shade detection-count differences
              </label>
            </div>
            <Timeline models={res.models} fps={clip.fps} current={frame} onSeek={seek} shadeDetections={shadeDets} />
          </div>
          <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
            {(["A", "B"] as const).map((slot) =>
              res.models[slot] ? <EventTables key={slot} slot={slot} m={res.models[slot]!} fps={clip.fps} frame={frame} onSeek={seek} /> : null,
            )}
          </div>
        </div>

        <div className="space-y-3">
          <GroundTruthCard res={res} onSaved={load} />
          {(["A", "B"] as const).map((slot) => (res.models[slot] ? <SummaryCard key={slot} slot={slot} m={res.models[slot]!} /> : null))}
          <div className="card text-[11px] text-zinc-500">
            Box colours: per-ID colour = counted track · <span className="text-orange-400">orange = detected ≥ N frames but never counted</span> · grey = short-lived.
            White dot = anchor point; grey lines = hysteresis band.
          </div>
        </div>
      </div>
    </div>
  );
}

function GroundTruthCard({ res, onSaved }: { res: Results; onSaved: () => void }) {
  const c = res.clip;
  const [gi, setGi] = useState(c.gt_in?.toString() ?? "");
  const [go, setGo] = useState(c.gt_out?.toString() ?? "");
  const [note, setNote] = useState(c.gt_note ?? "");
  const [saving, setSaving] = useState(false);
  const p = (v: string) => (v.trim() === "" ? null : Math.max(0, parseInt(v, 10) || 0));
  const save = async () => {
    setSaving(true);
    await api.groundTruth(c.id, p(gi), p(go), note || null).finally(() => setSaving(false));
    onSaved();
  };
  return (
    <div className="card space-y-2">
      <div className="label">Ground truth</div>
      <div className="flex items-center gap-2">
        <label className="text-xs text-zinc-400">IN <input className="input w-14" value={gi} onChange={(e) => setGi(e.target.value)} /></label>
        <label className="text-xs text-zinc-400">OUT <input className="input w-14" value={go} onChange={(e) => setGo(e.target.value)} /></label>
        <button className="btn-primary ml-auto" onClick={save} disabled={saving}>Save</button>
      </div>
      <input className="input w-full" placeholder="note (e.g. 2 people side by side, one hidden)" value={note} onChange={(e) => setNote(e.target.value)} />
      <div className="grid grid-cols-2 gap-2 pt-1">
        {(["A", "B"] as const).map((slot) => {
          const m = res.models[slot];
          return (
            <div key={slot} className="rounded bg-zinc-950 p-2 text-center">
              <SlotTag slot={slot} />
              <div className="mt-1 font-mono text-lg">
                <span className="text-emerald-400">{m?.in ?? "–"}</span>
                <span className="text-zinc-600"> / </span>
                <span className="text-red-400">{m?.out ?? "–"}</span>
              </div>
              <div className="text-xs">
                <Check ok={m?.gt?.correct} />{" "}
                {m?.gt && !m.gt.correct && (
                  <span className="text-zinc-400">
                    err {fmtErr(m.gt.err_in)} in / {fmtErr(m.gt.err_out)} out
                  </span>
                )}
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}

const fmtErr = (v: number) => (v > 0 ? `+${v}` : `${v}`);

function SummaryCard({ slot, m }: { slot: Slot; m: ModelResult }) {
  const s = m.summary;
  const rows: [string, string | number][] = [
    ["Avg confidence", s.avg_conf.toFixed(3)],
    ["Avg detections / frame", s.avg_dets],
    ["Inference ms (model)", s.avg_infer_ms],
    ["ms / frame incl. tracking", s.avg_wall_ms],
    ["FPS (laptop)", s.fps],
    ["Tracks (after merge)", `${s.n_tracks} (${s.n_raw_tracks} raw IDs)`],
    ["ID switches (est.)", s.id_switches],
    ["Lost-track merges", s.merges],
    ["Uncounted tracks", s.uncounted],
    ["Suppressed crossings", m.suppressed.length],
  ];
  return (
    <div className={`card border-l-2 ${slot === "A" ? "border-l-slot-a" : "border-l-slot-b"}`}>
      <div className="mb-1 flex items-center justify-between">
        <SlotTag slot={slot}>{modelLabel(m.weights)}</SlotTag>
        <span className="text-[10px] text-zinc-500">conf {m.inference.conf} · iou {m.inference.iou} · {m.inference.imgsz}px · {m.inference.device}</span>
      </div>
      <table className="w-full text-xs">
        <tbody>
          {rows.map(([k, v]) => (
            <tr key={k}>
              <td className="py-0.5 text-zinc-500">{k}</td>
              <td className="py-0.5 text-right font-mono">{v}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function EventTables({ slot, m, fps, frame, onSeek }: { slot: Slot; m: ModelResult; fps: number; frame: number; onSeek: (f: number) => void }) {
  const [showSupp, setShowSupp] = useState(false);
  const near = (f: number) => Math.abs(f - frame) <= 2;
  return (
    <div className="card space-y-3 p-2">
      <SlotTag slot={slot}>{modelLabel(m.weights)}</SlotTag>
      <div>
        <div className="label mb-1">Count events ({m.events.length})</div>
        <table className="dense w-full text-xs">
          <thead>
            <tr><th>Frame</th><th>Time</th><th>Track</th><th>Dir</th><th>Anchor</th></tr>
          </thead>
          <tbody>
            {m.events.map((e, i) => (
              <tr key={i} onClick={() => onSeek(e.frame)} className={`cursor-pointer hover:bg-zinc-800 ${near(e.frame) ? "bg-zinc-800" : ""}`}>
                <td className="font-mono">{e.frame}</td>
                <td className="font-mono">{e.time.toFixed(2)}s</td>
                <td>#{e.track_id}{e.raw_track_id !== e.track_id ? ` (raw ${e.raw_track_id})` : ""}</td>
                <td className={e.direction === "in" ? "text-emerald-400" : "text-red-400"}>{e.direction.toUpperCase()}</td>
                <td className="font-mono text-zinc-500">{e.anchor.map((v) => v.toFixed(0)).join(",")} <span className="text-zinc-600">{e.anchor_source}</span></td>
              </tr>
            ))}
            {!m.events.length && <tr><td colSpan={5} className="text-zinc-600">none</td></tr>}
          </tbody>
        </table>
      </div>
      <div>
        <div className="label mb-1 text-orange-400">Uncounted tracks ({m.uncounted.length})</div>
        <table className="dense w-full text-xs">
          <thead>
            <tr><th>Track</th><th>Frames</th><th>In ROI</th><th>Span</th><th>Reason</th></tr>
          </thead>
          <tbody>
            {m.uncounted.map((u) => (
              <tr key={u.track_id} onClick={() => onSeek(u.first_frame)} className={`cursor-pointer hover:bg-zinc-800 ${frame >= u.first_frame && frame <= u.last_frame ? "bg-zinc-800/70" : ""}`}>
                <td>#{u.track_id}{u.raw_ids.length > 1 ? ` (${u.raw_ids.join("+")})` : ""}</td>
                <td className="font-mono">{u.frames}</td>
                <td className="font-mono">{u.frames_in_roi}</td>
                <td className="font-mono text-zinc-400">{u.first_frame}–{u.last_frame} <span className="text-zinc-600">({(u.first_frame / fps).toFixed(1)}s)</span></td>
                <td className="text-orange-300">{REASON_LABEL[u.reason] ?? u.reason}</td>
              </tr>
            ))}
            {!m.uncounted.length && <tr><td colSpan={5} className="text-zinc-600">none</td></tr>}
          </tbody>
        </table>
      </div>
      {m.merges.length > 0 && (
        <div>
          <div className="label mb-1">Lost-track merges ({m.merges.length})</div>
          <table className="dense w-full text-xs">
            <tbody>
              {m.merges.map((g, i) => (
                <tr key={i} onClick={() => onSeek(g.frame)} className="cursor-pointer hover:bg-zinc-800">
                  <td className="font-mono">f {g.frame}</td>
                  <td>#{g.from_id} → #{g.to_id}</td>
                  <td className="text-zinc-500">gap {g.gap}f · {g.dist}px</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
      {m.suppressed.length > 0 && (
        <div>
          <button className="label mb-1 hover:text-zinc-300" onClick={() => setShowSupp(!showSupp)}>
            {showSupp ? "▾" : "▸"} Suppressed crossings ({m.suppressed.length})
          </button>
          {showSupp && (
            <table className="dense w-full text-xs">
              <tbody>
                {m.suppressed.map((e, i) => (
                  <tr key={i} onClick={() => onSeek(e.frame)} className="cursor-pointer hover:bg-zinc-800">
                    <td className="font-mono">f {e.frame}</td>
                    <td>#{e.track_id}</td>
                    <td>{e.direction.toUpperCase()}</td>
                    <td className="text-zinc-500">{e.reason}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          )}
        </div>
      )}
    </div>
  );
}
