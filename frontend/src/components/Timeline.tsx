import { memo, useMemo } from "react";
import { CartesianGrid, ComposedChart, Line, ReferenceArea, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import type { ModelResult, Slot } from "../api";
import { SLOT_COLOR } from "./ui";

interface Props {
  models: Partial<Record<Slot, ModelResult>>;
  fps: number;
  current: number;
  onSeek: (f: number) => void;
  shadeDetections: boolean;
}

function ranges(n: number, pred: (i: number) => boolean): [number, number][] {
  const out: [number, number][] = [];
  let start = -1;
  for (let i = 0; i < n; i++) {
    if (pred(i)) {
      if (start < 0) start = i;
    } else if (start >= 0) {
      out.push([start, i - 1]);
      start = -1;
    }
  }
  if (start >= 0) out.push([start, n - 1]);
  return out;
}

// Plot-area insets (YAxis width / right margin); shared by the cursor overlay and click-to-seek.
const PLOT_LEFT = 30;
const PLOT_RIGHT = 12;

const Chart = memo(function Chart({ models, fps, shadeDetections }: Omit<Props, "current" | "onSeek">) {
  const a = models.A, b = models.B;
  const n = Math.max(a?.per_frame.n_dets.length ?? 0, b?.per_frame.n_dets.length ?? 0);
  const data = useMemo(
    () => Array.from({ length: n }, (_, f) => ({ f, A: a?.per_frame.n_dets[f], B: b?.per_frame.n_dets[f] })),
    [a, b, n],
  );
  const countDisagree = useMemo(
    () =>
      a && b
        ? ranges(n, (i) => a.per_frame.cum_in[i] !== b.per_frame.cum_in[i] || a.per_frame.cum_out[i] !== b.per_frame.cum_out[i])
        : [],
    [a, b, n],
  );
  const detDisagree = useMemo(
    () => (a && b && shadeDetections ? ranges(n, (i) => a.per_frame.n_dets[i] !== b.per_frame.n_dets[i]) : []),
    [a, b, n, shadeDetections],
  );

  return (
    <ResponsiveContainer width="100%" height={180}>
      <ComposedChart data={data} margin={{ top: 16, right: PLOT_RIGHT, bottom: 0, left: 0 }}>
        <CartesianGrid stroke="#27272a" vertical={false} />
        <XAxis dataKey="f" type="number" domain={[0, Math.max(0, n - 1)]} tick={{ fill: "#71717a", fontSize: 10 }}
          tickFormatter={(f) => `${(f / fps).toFixed(1)}s`} stroke="#3f3f46" />
        <YAxis width={PLOT_LEFT} allowDecimals={false} tick={{ fill: "#71717a", fontSize: 10 }} stroke="#3f3f46" />
        <Tooltip
          contentStyle={{ background: "#18181b", border: "1px solid #3f3f46", fontSize: 11 }}
          labelFormatter={(f) => `frame ${f} · ${(Number(f) / fps).toFixed(2)}s`}
          formatter={(v, name) => [v, `${name} detections`]}
        />
        {detDisagree.map(([x1, x2]) => (
          <ReferenceArea key={`d${x1}`} x1={x1 - 0.5} x2={x2 + 0.5} fill="#f59e0b" fillOpacity={0.06} ifOverflow="hidden" />
        ))}
        {countDisagree.map(([x1, x2]) => (
          <ReferenceArea key={`c${x1}`} x1={x1 - 0.5} x2={x2 + 0.5} fill="#ef4444" fillOpacity={0.16} ifOverflow="hidden" />
        ))}
        {(["A", "B"] as const).flatMap((slot) =>
          (models[slot]?.events ?? []).map((e, k) => (
            <ReferenceLine
              key={`${slot}${k}`}
              x={e.frame}
              stroke={SLOT_COLOR[slot]}
              strokeDasharray={e.direction === "in" ? undefined : "4 2"}
              label={{ value: `${slot}${e.direction === "in" ? "↓IN" : "↑OUT"}`, fill: SLOT_COLOR[slot], fontSize: 9, position: slot === "A" ? "insideTopLeft" : "insideTopRight" }}
            />
          )),
        )}
        <Line type="stepAfter" dataKey="A" stroke={SLOT_COLOR.A} dot={false} strokeWidth={1.5} isAnimationActive={false} />
        <Line type="stepAfter" dataKey="B" stroke={SLOT_COLOR.B} dot={false} strokeWidth={1.5} isAnimationActive={false} />
      </ComposedChart>
    </ResponsiveContainer>
  );
});

export default function Timeline(props: Props) {
  const n = props.models.A?.per_frame.n_dets.length ?? props.models.B?.per_frame.n_dets.length ?? 1;
  // Current-frame cursor is an absolutely positioned overlay so the chart doesn't re-render per frame.
  const pct = n > 1 ? props.current / (n - 1) : 0;
  const onClick = (e: React.MouseEvent<HTMLDivElement>) => {
    const r = e.currentTarget.getBoundingClientRect();
    const frac = (e.clientX - r.left - PLOT_LEFT) / (r.width - PLOT_LEFT - PLOT_RIGHT);
    if (frac >= -0.02 && frac <= 1.02) props.onSeek(Math.round(Math.min(1, Math.max(0, frac)) * (n - 1)));
  };
  return (
    <div className="relative cursor-crosshair" onClick={onClick}>
      <Chart models={props.models} fps={props.fps} shadeDetections={props.shadeDetections} />
      <div className="pointer-events-none absolute top-4 bottom-[30px]" style={{ left: `calc(${PLOT_LEFT}px + (100% - ${PLOT_LEFT + PLOT_RIGHT}px) * ${pct})` }}>
        <div className="h-full w-px bg-white/70" />
      </div>
    </div>
  );
}
