import { forwardRef, useCallback, useEffect, useImperativeHandle, useRef, useState } from "react";
import type { Slot } from "../api";
import { SlotTag } from "./ui";

export interface SyncedHandle {
  seekFrame: (f: number) => void;
}

interface Props {
  srcs: Partial<Record<Slot, string>>;
  labels: Partial<Record<Slot, string>>;
  fps: number;
  frameCount: number;
  onFrame: (f: number) => void;
}

const SPEEDS = [0.25, 0.5, 1, 1.5, 2];

/** Two videos driven by one transport; B is slaved to A (master clock). */
const SyncedVideos = forwardRef<SyncedHandle, Props>(function SyncedVideos({ srcs, labels, fps, frameCount, onFrame }, ref) {
  const va = useRef<HTMLVideoElement>(null);
  const vb = useRef<HTMLVideoElement>(null);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [frame, setFrame] = useState(0);
  const raf = useRef(0);
  const lastReport = useRef(-1);

  const vids = () => [va.current, vb.current].filter((v): v is HTMLVideoElement => !!v);
  const master = () => va.current ?? vb.current;
  const maxFrame = Math.max(0, frameCount - 1);

  const report = useCallback(
    (t: number) => {
      const f = Math.min(maxFrame, Math.max(0, Math.floor(t * fps + 1e-3)));
      setFrame(f);
      if (f !== lastReport.current) {
        lastReport.current = f;
        onFrame(f);
      }
    },
    [fps, maxFrame, onFrame],
  );

  const seekFrame = useCallback(
    (f: number) => {
      const fr = Math.min(maxFrame, Math.max(0, Math.round(f)));
      const t = (fr + 0.5) / fps; // middle of the frame avoids rounding to the neighbour
      vids().forEach((v) => {
        v.pause();
        v.currentTime = t;
      });
      setPlaying(false);
      report(t);
    },
    [fps, maxFrame, report],
  );

  useImperativeHandle(ref, () => ({ seekFrame }), [seekFrame]);

  // master/slave sync loop while playing
  useEffect(() => {
    if (!playing) return;
    const tick = () => {
      const m = master();
      const s = m === va.current ? vb.current : null;
      if (m) {
        if (s && Math.abs(s.currentTime - m.currentTime) > 1.5 / fps) s.currentTime = m.currentTime;
        report(m.currentTime);
        if (m.ended) setPlaying(false);
      }
      raf.current = requestAnimationFrame(tick);
    };
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [playing, fps, report]);

  useEffect(() => {
    vids().forEach((v) => (v.playbackRate = speed));
  }, [speed, srcs.A, srcs.B]);

  const toggle = async () => {
    const vs = vids();
    if (playing) {
      vs.forEach((v) => v.pause());
      setPlaying(false);
      const m = master();
      if (m) seekFrame(Math.floor(m.currentTime * fps));
      return;
    }
    const m = master();
    if (m && m.currentTime >= m.duration - 0.05) vs.forEach((v) => (v.currentTime = 0));
    const t = m?.currentTime ?? 0;
    vs.forEach((v) => {
      v.currentTime = t;
      v.playbackRate = speed;
    });
    await Promise.all(vs.map((v) => v.play().catch(() => {})));
    setPlaying(true);
  };

  // keyboard: space play/pause, arrows frame step
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if ((e.target as HTMLElement).tagName === "INPUT" || (e.target as HTMLElement).tagName === "SELECT") return;
      if (e.code === "Space") {
        e.preventDefault();
        toggle();
      } else if (e.code === "ArrowRight") {
        e.preventDefault();
        seekFrame(frame + (e.shiftKey ? 10 : 1));
      } else if (e.code === "ArrowLeft") {
        e.preventDefault();
        seekFrame(frame - (e.shiftKey ? 10 : 1));
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  return (
    <div className="space-y-2">
      <div className="grid grid-cols-2 gap-2">
        {(["A", "B"] as const).map((slot) => (
          <div key={slot} className="overflow-hidden rounded border border-zinc-800 bg-black">
            <div className="flex items-center justify-between bg-zinc-900 px-2 py-1 text-xs">
              <SlotTag slot={slot}>{labels[slot]}</SlotTag>
            </div>
            {srcs[slot] ? (
              <video
                ref={slot === "A" ? va : vb}
                src={srcs[slot]}
                className="aspect-video w-full bg-black object-contain"
                muted
                playsInline
                preload="auto"
                onLoadedMetadata={(e) => {
                  e.currentTarget.playbackRate = speed;
                  const m = master();
                  if (m && m !== e.currentTarget) e.currentTarget.currentTime = m.currentTime;
                }}
              />
            ) : (
              <div className="flex aspect-video items-center justify-center text-xs text-zinc-500">rendering…</div>
            )}
          </div>
        ))}
      </div>
      <div className="flex items-center gap-2 rounded border border-zinc-800 bg-zinc-900 px-2 py-1.5">
        <button className="btn" onClick={() => seekFrame(0)} title="Start">⏮</button>
        <button className="btn" onClick={() => seekFrame(frame - 1)} title="Previous frame (←)">◀ 1f</button>
        <button className="btn-primary w-16" onClick={toggle} title="Space">{playing ? "Pause" : "Play"}</button>
        <button className="btn" onClick={() => seekFrame(frame + 1)} title="Next frame (→)">1f ▶</button>
        <input
          type="range"
          min={0}
          max={maxFrame}
          value={frame}
          onChange={(e) => seekFrame(Number(e.target.value))}
          className="mx-2 flex-1 accent-sky-500"
        />
        <span className="w-36 text-right font-mono text-xs text-zinc-300">
          f {frame}/{maxFrame} · {(frame / fps).toFixed(2)}s
        </span>
        <select className="input" value={speed} onChange={(e) => setSpeed(Number(e.target.value))}>
          {SPEEDS.map((s) => (
            <option key={s} value={s}>{s}×</option>
          ))}
        </select>
      </div>
    </div>
  );
});

export default SyncedVideos;
