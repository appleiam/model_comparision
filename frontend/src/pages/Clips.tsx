import { useEffect, useMemo, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { api, modelLabel, urls, type Clip } from "../api";
import { Check, ErrorBox, ProgressBar, SlotTag, StatusPill } from "../components/ui";

type Filter = "all" | "both_wrong" | "only_a" | "only_b" | "no_gt";

const FILTERS: [Filter, string][] = [
  ["all", "All"],
  ["both_wrong", "Both wrong"],
  ["only_a", "Only A wrong"],
  ["only_b", "Only B wrong"],
  ["no_gt", "No ground truth"],
];

function matches(c: Clip, f: Filter) {
  const a = c.gt_eval.A?.correct;
  const b = c.gt_eval.B?.correct;
  switch (f) {
    case "both_wrong":
      return a === false && b === false;
    case "only_a":
      return a === false && b === true;
    case "only_b":
      return a === true && b === false;
    case "no_gt":
      return c.gt_in == null || c.gt_out == null;
    default:
      return true;
  }
}

export default function ClipsPage() {
  const [clips, setClips] = useState<Clip[]>([]);
  const [filter, setFilter] = useState<Filter>("all");
  const [error, setError] = useState<string | null>(null);
  const nav = useNavigate();

  const load = () => api.clips().then(setClips).catch((e) => setError(String(e.message)));
  useEffect(() => {
    load();
    const t = setInterval(load, 4000);
    return () => clearInterval(t);
  }, []);

  const shown = useMemo(() => clips.filter((c) => matches(c, filter)), [clips, filter]);
  const counts = useMemo(() => Object.fromEntries(FILTERS.map(([f]) => [f, clips.filter((c) => matches(c, f)).length])), [clips]);

  async function saveGt(c: Clip, gt_in: number | null, gt_out: number | null) {
    try {
      const upd = await api.groundTruth(c.id, gt_in, gt_out, c.gt_note);
      setClips((prev) => prev.map((p) => (p.id === c.id ? upd : p)));
    } catch (e) {
      setError(String((e as Error).message));
    }
  }

  async function del(c: Clip) {
    if (!confirm(`Delete ${c.name} and its results?`)) return;
    await api.deleteClip(c.id).catch((e) => setError(String(e.message)));
    load();
  }

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        {FILTERS.map(([f, label]) => (
          <button key={f} className={`btn ${filter === f ? "!bg-sky-700 !border-sky-600 text-white" : ""}`} onClick={() => setFilter(f)}>
            {label} <span className="text-zinc-400">{counts[f]}</span>
          </button>
        ))}
        {clips.some((c) => c.status === "done") && (
          <Link to={`/clips/${clips.find((c) => c.status === "done")!.id}/roi`} className="btn ml-auto">Edit ROI (all clips)</Link>
        )}
        <Link to="/upload" className={`btn-primary ${clips.some((c) => c.status === "done") ? "" : "ml-auto"}`}>+ Upload clips</Link>
      </div>
      <ErrorBox error={error} />
      <div className="card overflow-x-auto p-0">
        <table className="dense w-full">
          <thead>
            <tr>
              <th></th>
              <th>Clip</th>
              <th>Dur</th>
              <th>Status</th>
              <th>Ground truth IN / OUT</th>
              <th><SlotTag slot="A" /> IN / OUT</th>
              <th></th>
              <th><SlotTag slot="B" /> IN / OUT</th>
              <th></th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {shown.map((c) => (
              <tr key={c.id} className="cursor-pointer hover:bg-zinc-800/40" onClick={() => c.status === "done" && nav(`/clips/${c.id}`)}>
                <td className="w-20"><img src={urls.thumb(c.id)} alt="" className="h-10 w-16 rounded object-cover" /></td>
                <td>
                  <div className="font-medium text-zinc-100">{c.name}</div>
                  <div className="text-[11px] text-zinc-500">#{c.id} · {c.width}×{c.height} @ {c.fps.toFixed(1)} fps{c.gt_note ? ` · ${c.gt_note}` : ""}</div>
                </td>
                <td className="font-mono text-xs">{c.duration.toFixed(1)}s</td>
                <td className="w-32">
                  <StatusPill status={c.status} />
                  {(c.status === "processing" || c.status === "queued") && <div className="mt-1"><ProgressBar value={c.progress} /></div>}
                  {c.error && <div className="mt-0.5 max-w-48 truncate text-[11px] text-red-400" title={c.error}>{c.error}</div>}
                </td>
                <td onClick={(e) => e.stopPropagation()}>
                  <GtInputs clip={c} onSave={saveGt} />
                </td>
                {(["A", "B"] as const).map((slot) => {
                  const s = c.summary[slot];
                  return [
                    <td key={slot} className="font-mono">
                      {s ? (
                        <>
                          <span className="text-emerald-400">{s.in}</span> / <span className="text-red-400">{s.out}</span>
                          <div className="text-[10px] text-zinc-500">{modelLabel(s.weights)}{s.uncounted ? ` · ${s.uncounted} uncounted` : ""}</div>
                        </>
                      ) : (
                        <span className="text-zinc-600">—</span>
                      )}
                    </td>,
                    <td key={slot + "ok"} className="text-center"><Check ok={c.gt_eval[slot]?.correct} /></td>,
                  ];
                })}
                <td onClick={(e) => e.stopPropagation()} className="whitespace-nowrap text-right">
                  {c.status === "done" && <Link className="btn mr-1" to={`/clips/${c.id}/roi`}>ROI</Link>}
                  <button className="btn hover:!bg-red-900" onClick={() => del(c)}>✕</button>
                </td>
              </tr>
            ))}
            {shown.length === 0 && (
              <tr><td colSpan={10} className="py-6 text-center text-zinc-500">No clips match.</td></tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}

export function GtInputs({ clip, onSave }: { clip: Clip; onSave: (c: Clip, i: number | null, o: number | null) => void }) {
  const [gi, setGi] = useState(clip.gt_in?.toString() ?? "");
  const [go, setGo] = useState(clip.gt_out?.toString() ?? "");
  useEffect(() => {
    setGi(clip.gt_in?.toString() ?? "");
    setGo(clip.gt_out?.toString() ?? "");
  }, [clip.gt_in, clip.gt_out]);
  const parse = (v: string) => (v.trim() === "" ? null : Math.max(0, parseInt(v, 10) || 0));
  const commit = () => {
    const i = parse(gi), o = parse(go);
    if (i !== clip.gt_in || o !== clip.gt_out) onSave(clip, i, o);
  };
  return (
    <div className="flex items-center gap-1">
      <input className="input w-12 text-center" placeholder="in" value={gi} onChange={(e) => setGi(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === "Enter" && commit()} />
      <span className="text-zinc-600">/</span>
      <input className="input w-12 text-center" placeholder="out" value={go} onChange={(e) => setGo(e.target.value)} onBlur={commit} onKeyDown={(e) => e.key === "Enter" && commit()} />
    </div>
  );
}
