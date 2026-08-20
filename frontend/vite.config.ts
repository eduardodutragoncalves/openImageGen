import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

const API_ORIGIN = `http://127.0.0.1:${process.env.OIG_PORT ?? 8000}`;

// ADR-001: the studio is built into the FastAPI app and served from the same
// origin, so production needs no CORS at all. In development Vite proxies the
// API instead, which keeps cookies first-party.
export default defineConfig({
  plugins: [react(), tailwindcss()],
  build: {
    outDir: "../app/static",
    emptyOutDir: true,
    assetsDir: "assets",
  },
  server: {
    port: 5173,
    proxy: {
      // The port is read from the environment so `scripts/start.sh --api-port`
      // moves both halves at once; hardcoding it here would leave the studio
      // proxying to a port with nothing on it.
      "/v1": { target: API_ORIGIN, changeOrigin: false },
      "/healthz": { target: API_ORIGIN, changeOrigin: false },
    },
  },
});
