import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const backend = process.env.APCS_BACKEND ?? "http://127.0.0.1:8000";
const api = ["/system", "/clips", "/jobs", "/roi-presets", "/roi/", "/report"];

export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: { chunkSizeWarningLimit: 1500 },
  server: {
    port: 5173,
    proxy: {
      ...Object.fromEntries(api.map((p) => [p, { target: backend, changeOrigin: true }])),
      "/ws": { target: backend.replace(/^http/, "ws"), ws: true },
    },
  },
});
