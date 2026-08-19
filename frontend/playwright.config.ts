import { defineConfig } from "@playwright/test";

/**
 * The critical path runs against the real FastAPI app in dry-run mode: the
 * whole HTTP layer, the queue, the archive and the SPA, with no GPU. Weights
 * are the only thing mocked out.
 */
const PORT = 8021;

export default defineConfig({
  testDir: "./tests",
  timeout: 60_000,
  expect: { timeout: 15_000 },
  fullyParallel: false,
  workers: 1,
  reporter: process.env.CI ? "github" : "list",
  use: {
    baseURL: `http://127.0.0.1:${PORT}`,
    trace: "retain-on-failure",
  },
  webServer: {
    command: [
      "python -m uvicorn app.main:app",
      "--host 127.0.0.1",
      `--port ${PORT}`,
    ].join(" "),
    cwd: "..",
    url: `http://127.0.0.1:${PORT}/healthz`,
    reuseExistingServer: false,
    timeout: 120_000,
    env: {
      OIG_DRY_RUN: "true",
      OIG_DRY_RUN_STEP_SECONDS: "0.05",
      OIG_HOST: "127.0.0.1",
      OIG_API_KEYS: "e2e-key,other-key",
      OIG_ENABLE_NSFW_FILTER: "false",
      OIG_STATE_DIR: "./.e2e-state",
      OIG_OUTPUT_DIR: "./.e2e-output",
    },
  },
});
