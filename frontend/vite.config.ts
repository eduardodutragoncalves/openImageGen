import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import tailwindcss from "@tailwindcss/vite";

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
      "/v1": { target: "http://127.0.0.1:8000", changeOrigin: false },
      "/healthz": { target: "http://127.0.0.1:8000", changeOrigin: false },
    },
  },
});
