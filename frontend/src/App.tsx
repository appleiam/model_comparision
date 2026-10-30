import { createContext, useContext, useEffect, useState } from "react";
import { HashRouter, NavLink, Navigate, Route, Routes } from "react-router-dom";
import { api, type SystemInfo } from "./api";
import UploadPage from "./pages/Upload";
import ClipsPage from "./pages/Clips";
import ComparePage from "./pages/Compare";
import RoiEditorPage from "./pages/RoiEditor";
import ReportPage from "./pages/Report";

const SystemCtx = createContext<SystemInfo | null>(null);
export const useSystem = () => useContext(SystemCtx);

function DeviceBadge({ sys }: { sys: SystemInfo | null }) {
  if (!sys) return <span className="text-zinc-500">connecting to backend…</span>;
  const gpu = sys.device.startsWith("cuda");
  const color = gpu ? "bg-emerald-900/60 text-emerald-300 border-emerald-700" : sys.device === "mps" ? "bg-sky-900/60 text-sky-300 border-sky-700" : "bg-amber-900/50 text-amber-300 border-amber-700";
  return (
    <div className="flex items-center gap-2 text-xs">
      {!sys.models_ready && <span className="text-amber-400">loading models…</span>}
      <span className={`rounded border px-2 py-0.5 font-mono ${color}`} title={`torch ${sys.torch}`}>
        {sys.device.toUpperCase()} · {sys.cuda_name ?? sys.device_name}
        {sys.vram_total_mb ? ` · ${(sys.vram_total_mb / 1024).toFixed(1)} GB` : ""}
      </span>
    </div>
  );
}

export default function App() {
  const [sys, setSys] = useState<SystemInfo | null>(null);
  useEffect(() => {
    let stop = false;
    const load = () =>
      api
        .system()
        .then((s) => {
          if (stop) return;
          setSys(s);
          if (!s.models_ready) setTimeout(load, 3000);
        })
        .catch(() => !stop && setTimeout(load, 3000));
    load();
    return () => {
      stop = true;
    };
  }, []);

  const link = ({ isActive }: { isActive: boolean }) =>
    `px-3 py-1.5 rounded text-xs font-medium ${isActive ? "bg-zinc-800 text-white" : "text-zinc-400 hover:text-zinc-200"}`;

  return (
    <SystemCtx.Provider value={sys}>
      <HashRouter>
        <div className="flex min-h-screen flex-col">
          <header className="flex items-center gap-4 border-b border-zinc-800 bg-zinc-900/80 px-4 py-2">
            <div className="font-semibold tracking-tight text-white">
              APCS <span className="text-zinc-400 font-normal">Model Comparator</span>
            </div>
            <nav className="flex gap-1">
              <NavLink to="/upload" className={link}>Upload</NavLink>
              <NavLink to="/clips" className={link}>Clips</NavLink>
              <NavLink to="/report" className={link}>Batch report</NavLink>
            </nav>
            <div className="ml-auto">
              <DeviceBadge sys={sys} />
            </div>
          </header>
          <main className="flex-1 p-4">
            <Routes>
              <Route path="/" element={<Navigate to="/clips" replace />} />
              <Route path="/upload" element={<UploadPage />} />
              <Route path="/clips" element={<ClipsPage />} />
              <Route path="/clips/:id" element={<ComparePage />} />
              <Route path="/clips/:id/roi" element={<RoiEditorPage />} />
              <Route path="/report" element={<ReportPage />} />
            </Routes>
          </main>
          <footer className="border-t border-zinc-900 px-4 py-1.5 text-[11px] text-zinc-600">
            FPS shown is laptop PyTorch speed ({sys?.device ?? "…"}), not Hailo-8L speed on the device.
          </footer>
        </div>
      </HashRouter>
    </SystemCtx.Provider>
  );
}
