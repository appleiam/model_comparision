// Typed client for the FastAPI backend (same origin; Vite proxies in dev).

export type Pt = [number, number];
export type Slot = "A" | "B";

export interface CountParams {
  anchor: "hip_fallback" | "bbox_center" | "bbox_bottom";
  kp_conf: number;
  anchor_stabilize: boolean;
  hysteresis: number;
  min_track_age: number;
  count_both_directions: boolean;
  recovery_enabled: boolean;
  recovery_frames: number;
  recovery_px: number;
  uncounted_min_frames: number;
}

export interface RoiConfig {
  roi: Pt[];
  line: [Pt, Pt];
  in_side: 1 | -1;
  params: CountParams;
}

export interface GtEval {
  err_in: number;
  err_out: number;
  abs_error: number;
  correct: boolean;
}

export interface SlotSummary {
  weights: string;
  in: number;
  out: number;
  uncounted: number;
  fps: number;
}

export interface Clip {
  id: number;
  name: string;
  fps: number;
  width: number;
  height: number;
  frame_count: number;
  duration: number;
  status: "uploaded" | "queued" | "processing" | "done" | "error";
  progress: number;
  error: string | null;
  created_at: string;
  gt_in: number | null;
  gt_out: number | null;
  gt_note: string | null;
  slot_a: string | null;
  slot_b: string | null;
  preset_id: number | null;
  roi: RoiConfig | null;
  summary: Partial<Record<Slot, SlotSummary>>;
  gt_eval: Partial<Record<Slot, GtEval | null>>;
  render_status: "none" | "rendering" | "done" | "error";
  render_version: number;
}

export interface CountEvent {
  frame: number;
  time: number;
  track_id: number;
  raw_track_id: number;
  direction: "in" | "out";
  anchor: Pt;
  anchor_source: string;
  reason?: string;
}

export interface Uncounted {
  track_id: number;
  raw_ids: number[];
  first_frame: number;
  last_frame: number;
  frames: number;
  frames_in_roi: number;
  reason: string;
  last_anchor: Pt;
}

export interface Merge {
  frame: number;
  from_id: number;
  to_id: number;
  gap: number;
  dist: number;
}

export interface ModelResult {
  slot: Slot;
  weights: string;
  inference: { conf: number; iou: number; imgsz: number; device: string };
  in: number;
  out: number;
  events: CountEvent[];
  suppressed: CountEvent[];
  uncounted: Uncounted[];
  merges: Merge[];
  summary: {
    avg_conf: number;
    avg_dets: number;
    avg_infer_ms: number;
    avg_wall_ms: number;
    fps: number;
    n_raw_tracks: number;
    n_tracks: number;
    id_switches: number;
    merges: number;
    uncounted: number;
    device: string;
  };
  per_frame: {
    n_dets: number[];
    mean_conf: (number | null)[];
    infer_ms: number[];
    active_tracks: number[];
    cum_in: number[];
    cum_out: number[];
  };
  gt: GtEval | null;
}

export interface Results {
  clip_id: number;
  config: RoiConfig;
  models: Partial<Record<Slot, ModelResult>>;
  clip: Clip;
  recount_scheduled?: number[];
}

export interface Job {
  id: number;
  clip_ids: number[];
  slot_a: string;
  slot_b: string;
  preset_id: number | null;
  settings: { conf: number; iou: number; imgsz: number };
  status: "queued" | "running" | "done" | "error";
  progress: number;
  message: string | null;
  clips: Record<string, { progress: number; stage: string; status: string; error?: string }>;
  created_at: string;
}

export interface Preset {
  id: number;
  name: string;
  config: RoiConfig;
}

export interface SystemInfo {
  device: string;
  device_name: string;
  cuda_available: boolean;
  cuda_name: string | null;
  vram_total_mb: number | null;
  vram_used_mb: number | null;
  torch: string;
  weights: string[];
  slots: Partial<Record<Slot, string>>;
  default_slots: Record<Slot, string>;
  models_ready: boolean;
  startup_error: string | null;
  defaults: {
    inference: { conf: number; iou: number; imgsz: number };
    count_params: CountParams;
    roi: RoiConfig;
  };
}

export interface ReportModel {
  slot: Slot;
  weights: string;
  clips: number;
  clips_with_gt: number;
  correct: number;
  accuracy: number | null;
  total_abs_error: number;
  err_in: number;
  err_out: number;
  avg_fps: number;
  uncounted: number;
  id_switches: number;
  reasons: Record<string, number>;
}

export interface ReportRow {
  clip_id: number;
  name: string;
  duration: number;
  gt_in: number | null;
  gt_out: number | null;
  gt_note: string | null;
  models: Partial<
    Record<
      Slot,
      {
        weights: string;
        in: number;
        out: number;
        gt: GtEval | null;
        fps: number;
        avg_conf: number;
        uncounted: number;
        id_switches: number;
        reasons: Record<string, number>;
      }
    >
  >;
}

export interface Report {
  rows: ReportRow[];
  models: ReportModel[];
  agreement: { both_right: number; both_wrong: number; only_a_wrong: number; only_b_wrong: number };
  note: string;
}

async function req<T>(url: string, init?: RequestInit): Promise<T> {
  const r = await fetch(url, init);
  if (!r.ok) {
    let msg = `${r.status} ${r.statusText}`;
    try {
      const j = await r.json();
      msg = typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j);
    } catch {
      /* not json */
    }
    throw new Error(msg);
  }
  return r.json() as Promise<T>;
}

const json = (method: string, body: unknown): RequestInit => ({
  method,
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const api = {
  system: () => req<SystemInfo>("/system"),
  clips: () => req<Clip[]>("/clips"),
  clip: (id: number) => req<Clip>(`/clips/${id}`),
  deleteClip: (id: number) => req<{ ok: boolean }>(`/clips/${id}`, { method: "DELETE" }),
  upload: (files: File[]) => {
    const fd = new FormData();
    files.forEach((f) => fd.append("files", f));
    return req<(Clip | { name: string; error: string })[]>("/clips/upload", { method: "POST", body: fd });
  },
  groundTruth: (id: number, gt_in: number | null, gt_out: number | null, note: string | null) =>
    req<Clip>(`/clips/${id}/ground-truth`, json("POST", { gt_in, gt_out, note })),
  results: (id: number) => req<Results>(`/clips/${id}/results`),
  recount: (id: number, cfg: RoiConfig, preset_id: number | null, scope: "all" | "clip" = "all") =>
    req<Results>(`/clips/${id}/recount`, json("POST", { ...cfg, preset_id, scope })),
  activeRoi: () => req<RoiConfig>("/roi/active"),
  createJob: (body: {
    clip_ids: number[];
    slot_a: string;
    slot_b: string;
    preset_id: number | null;
    conf: number;
    iou: number;
    imgsz: number;
  }) => req<Job>("/jobs", json("POST", body)),
  jobs: () => req<Job[]>("/jobs"),
  job: (id: number) => req<Job>(`/jobs/${id}`),
  presets: () => req<Preset[]>("/roi-presets"),
  savePreset: (body: Record<string, unknown>) => req<Preset>("/roi-presets", json("POST", body)),
  deletePreset: (id: number) => req<{ ok: boolean }>(`/roi-presets/${id}`, { method: "DELETE" }),
  applyPresetAll: (id: number) => req<{ recount_scheduled: number[] }>(`/roi-presets/${id}/apply-all`, { method: "POST" }),
  report: () => req<Report>("/report"),
};

export const urls = {
  thumb: (id: number) => `/clips/${id}/thumbnail`,
  frame: (id: number, i: number) => `/clips/${id}/frame?i=${i}`,
  video: (id: number, slot: Slot, v: number) => `/clips/${id}/video/${slot}?v=${v}`,
  presetExport: (id: number, clipId?: number) => `/roi-presets/${id}/export${clipId ? `?clip_id=${clipId}` : ""}`,
  clipRoiExport: (id: number) => `/clips/${id}/roi-export`,
  activeRoiExport: (clipId?: number) => `/roi/active/export${clipId ? `?clip_id=${clipId}` : ""}`,
  reportCsv: "/report.csv",
  ws: (jobId: number) => `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws/jobs/${jobId}`,
};

export const modelLabel = (w: string | null | undefined) => (w ? w.replace(/\.pt$/, "") : "—");

export const REASON_LABEL: Record<string, string> = {
  never_crossed: "never crossed line",
  too_short_lived: "too short-lived",
  outside_roi: "outside ROI",
  jitter_suppressed: "jitter-suppressed",
};
