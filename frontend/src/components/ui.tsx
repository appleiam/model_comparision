import type { ReactNode } from "react";

export function ProgressBar({ value, status }: { value: number; status?: string }) {
  const color = status === "error" ? "bg-red-500" : status === "done" ? "bg-emerald-500" : "bg-sky-500";
  return (
    <div className="h-1.5 w-full overflow-hidden rounded bg-zinc-800">
      <div className={`h-full ${color} transition-all`} style={{ width: `${Math.min(100, Math.max(0, value))}%` }} />
    </div>
  );
}

export function StatusPill({ status }: { status: string }) {
  const map: Record<string, string> = {
    uploaded: "bg-zinc-800 text-zinc-300",
    queued: "bg-zinc-700 text-zinc-200",
    processing: "bg-sky-900 text-sky-300",
    running: "bg-sky-900 text-sky-300",
    done: "bg-emerald-900/70 text-emerald-300",
    error: "bg-red-900/70 text-red-300",
  };
  return <span className={`rounded px-1.5 py-0.5 text-[11px] font-medium ${map[status] ?? "bg-zinc-800"}`}>{status}</span>;
}

export function Check({ ok }: { ok: boolean | null | undefined }) {
  if (ok === null || ok === undefined) return <span className="text-zinc-600">–</span>;
  return ok ? <span className="font-bold text-emerald-400">✓</span> : <span className="font-bold text-red-400">✗</span>;
}

export function SlotTag({ slot, children }: { slot: "A" | "B"; children?: ReactNode }) {
  return (
    <span className={`inline-flex items-center gap-1 font-medium ${slot === "A" ? "text-slot-a" : "text-slot-b"}`}>
      <span className={`rounded px-1 text-[10px] font-bold text-black ${slot === "A" ? "bg-slot-a" : "bg-slot-b"}`}>{slot}</span>
      {children}
    </span>
  );
}

export function ErrorBox({ error }: { error: string | null }) {
  if (!error) return null;
  return <div className="rounded border border-red-800 bg-red-950/50 px-3 py-2 text-xs text-red-300">{error}</div>;
}

export const SLOT_COLOR = { A: "#38bdf8", B: "#c084fc" } as const;
