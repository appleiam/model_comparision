import { useEffect, useMemo, useState } from "react";
import { Link } from "react-router-dom";
import { Bar, BarChart, CartesianGrid, Legend, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { api, modelLabel, REASON_LABEL, urls, type Report } from "../api";
import { Check, ErrorBox, SLOT_COLOR, SlotTag } from "../components/ui";

const tooltipStyle = { background: "#18181b", border: "1px solid #3f3f46", fontSize: 11 };
const axis = { tick: { fill: "#71717a", fontSize: 10 }, stroke: "#3f3f46" };

export default function ReportPage() {
  const [rep, setRep] = useState<Report | null>(null);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    api.report().then(setRep).catch((e) => setError(String(e.message)));
  }, []);

  const perClip = useMemo(
    () =>
      (rep?.rows ?? []).map((r) => ({
        name: `#${r.clip_id}`,
        A: r.models.A?.gt ? r.models.A.gt.abs_error : null,
        B: r.models.B?.gt ? r.models.B.gt.abs_error : null,
      })),
    [rep],
  );
  const reasons = useMemo(() => {
    const keys = new Set<string>();
    rep?.models.forEach((m) => Object.keys(m.reasons).forEach((k) => keys.add(k)));
    return [...keys].map((k) => ({
      reason: REASON_LABEL[k] ?? k,
      ...Object.fromEntries((rep?.models ?? []).map((m) => [`${m.slot}:${modelLabel(m.weights)}`, m.reasons[k] ?? 0])),
    }));
  }, [rep]);

  function downloadJson() {
    const blob = new Blob([JSON.stringify(rep, null, 2)], { type: "application/json" });
    const a = document.createElement("a");
    a.href = URL.createObjectURL(blob);
    a.download = "apcs_report.json";
    a.click();
    URL.revokeObjectURL(a.href);
  }

  if (error) return <ErrorBox error={error} />;
  if (!rep) return <div className="text-zinc-500">Loading…</div>;
  const withGt = rep.rows.filter((r) => r.gt_in != null && r.gt_out != null).length;

  return (
    <div className="space-y-3">
      <div className="flex items-center gap-2">
        <h1 className="text-base font-semibold text-white">Batch report</h1>
        <span className="text-xs text-zinc-500">{rep.rows.length} processed clips · {withGt} with ground truth</span>
        <div className="ml-auto flex gap-2">
          <a className="btn" href={urls.reportCsv} download>Export CSV</a>
          <button className="btn" onClick={downloadJson}>Export JSON</button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 md:grid-cols-2 xl:grid-cols-4">
        {rep.models.map((m) => (
          <div key={`${m.slot}${m.weights}`} className={`card border-l-2 ${m.slot === "A" ? "border-l-slot-a" : "border-l-slot-b"}`}>
            <SlotTag slot={m.slot}>{modelLabel(m.weights)}</SlotTag>
            <div className="mt-2 grid grid-cols-2 gap-y-1 text-xs">
              <span className="text-zinc-500">Accuracy</span>
              <span className="text-right font-mono text-lg text-white">{m.accuracy == null ? "–" : `${(m.accuracy * 100).toFixed(0)}%`}</span>
              <span className="text-zinc-500">Correct clips</span><span className="text-right font-mono">{m.correct}/{m.clips_with_gt}</span>
              <span className="text-zinc-500">Total |error|</span><span className="text-right font-mono">{m.total_abs_error}</span>
              <span className="text-zinc-500">Net err IN / OUT</span><span className="text-right font-mono">{m.err_in} / {m.err_out}</span>
              <span className="text-zinc-500">Avg FPS (laptop)</span><span className="text-right font-mono">{m.avg_fps}</span>
              <span className="text-zinc-500">Uncounted tracks</span><span className="text-right font-mono text-orange-400">{m.uncounted}</span>
              <span className="text-zinc-500">ID switches (est.)</span><span className="text-right font-mono">{m.id_switches}</span>
            </div>
          </div>
        ))}
        <div className="card text-xs">
          <div className="label mb-2">Agreement (clips with GT)</div>
          <div className="grid grid-cols-2 gap-y-1">
            <span className="text-zinc-500">Both right</span><span className="text-right font-mono text-emerald-400">{rep.agreement.both_right}</span>
            <span className="text-zinc-500">Both wrong</span><span className="text-right font-mono text-red-400">{rep.agreement.both_wrong}</span>
            <span className="text-zinc-500">Only A wrong</span><span className="text-right font-mono">{rep.agreement.only_a_wrong}</span>
            <span className="text-zinc-500">Only B wrong</span><span className="text-right font-mono">{rep.agreement.only_b_wrong}</span>
          </div>
          <div className="mt-3 text-[11px] text-zinc-600">{rep.note}</div>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-3 lg:grid-cols-2">
        <div className="card">
          <div className="label mb-1">|count error| per clip (vs ground truth)</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={perClip} margin={{ left: -20 }}>
              <CartesianGrid stroke="#27272a" vertical={false} />
              <XAxis dataKey="name" {...axis} />
              <YAxis allowDecimals={false} {...axis} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#27272a" }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              <Bar dataKey="A" fill={SLOT_COLOR.A} isAnimationActive={false} />
              <Bar dataKey="B" fill={SLOT_COLOR.B} isAnimationActive={false} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="card">
          <div className="label mb-1">Uncounted-track reasons</div>
          <ResponsiveContainer width="100%" height={220}>
            <BarChart data={reasons} margin={{ left: -20 }}>
              <CartesianGrid stroke="#27272a" vertical={false} />
              <XAxis dataKey="reason" {...axis} />
              <YAxis allowDecimals={false} {...axis} />
              <Tooltip contentStyle={tooltipStyle} cursor={{ fill: "#27272a" }} />
              <Legend wrapperStyle={{ fontSize: 11 }} />
              {rep.models.map((m) => (
                <Bar key={m.slot + m.weights} dataKey={`${m.slot}:${modelLabel(m.weights)}`} fill={SLOT_COLOR[m.slot]} isAnimationActive={false} />
              ))}
            </BarChart>
          </ResponsiveContainer>
        </div>
      </div>

      <div className="card overflow-x-auto p-0">
        <table className="dense w-full text-xs">
          <thead>
            <tr>
              <th>Clip</th><th>GT in/out</th>
              <th><SlotTag slot="A" /> in/out</th><th>err</th><th></th><th>FPS</th><th>uncounted</th>
              <th><SlotTag slot="B" /> in/out</th><th>err</th><th></th><th>FPS</th><th>uncounted</th>
              <th>Note</th>
            </tr>
          </thead>
          <tbody>
            {rep.rows.map((r) => (
              <tr key={r.clip_id} className="hover:bg-zinc-800/40">
                <td><Link to={`/clips/${r.clip_id}`} className="hover:text-sky-400">#{r.clip_id} {r.name}</Link></td>
                <td className="font-mono">{r.gt_in ?? "–"} / {r.gt_out ?? "–"}</td>
                {(["A", "B"] as const).map((slot) => {
                  const m = r.models[slot];
                  return [
                    <td key={slot + "c"} className="font-mono">{m ? `${m.in} / ${m.out}` : "–"}</td>,
                    <td key={slot + "e"} className="font-mono">{m?.gt?.abs_error ?? "–"}</td>,
                    <td key={slot + "k"}><Check ok={m?.gt?.correct} /></td>,
                    <td key={slot + "f"} className="font-mono">{m?.fps ?? "–"}</td>,
                    <td key={slot + "u"} className="font-mono text-orange-400">{m?.uncounted ?? "–"}</td>,
                  ];
                })}
                <td className="text-zinc-500">{r.gt_note}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
