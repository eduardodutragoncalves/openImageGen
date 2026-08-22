/**
 * Capture the studio for the README.
 *
 * Run against a live server; the images land in docs/screens/ at 2x so they
 * stay sharp on a retina display and readable when GitHub scales them down.
 *
 *   cd frontend && npm run screens -- [--base http://localhost:8000] [--only models]
 *
 * The API key is read from .env, the same one the studio would ask you for.
 * Nothing here generates anything: point it at a server whose archive already
 * holds what you want photographed.
 */
import { chromium } from "@playwright/test";
import { readFileSync, mkdirSync } from "node:fs";
import { dirname, resolve } from "node:path";
import { fileURLToPath } from "node:url";

const ROOT = resolve(dirname(fileURLToPath(import.meta.url)), "../..");
const OUT = resolve(ROOT, "docs/screens");

const args = process.argv.slice(2);
const argOf = (name, fallback) => {
  const index = args.indexOf(`--${name}`);
  return index >= 0 ? args[index + 1] : fallback;
};
const BASE = argOf("base", "http://localhost:8000");
const ONLY = (argOf("only", "") || "").split(",").filter(Boolean);

const key = readFileSync(resolve(ROOT, ".env"), "utf8")
  .split("\n")
  .find((line) => line.startsWith("OIG_API_KEYS="))
  ?.split("=")[1]
  .split(",")[0]
  .trim();
if (!key) throw new Error("no OIG_API_KEYS in .env");

mkdirSync(OUT, { recursive: true });

const browser = await chromium.launch();
const context = await browser.newContext({ deviceScaleFactor: 2 });
// Authenticate the way the studio does, so the SPA opens past the gate.
await context.request.post(`${BASE}/v1/auth`, { data: { key } });

const page = await context.newPage();
const problems = [];
page.on("console", (m) => m.type() === "error" && problems.push(m.text()));
page.on("pageerror", (e) => problems.push(String(e)));

// Dark is the register this system was designed in; the light one is fully
// supported but the README shows the default.
await page.addInitScript(() => localStorage.setItem("oig-theme", "dark"));
const VIEWPORT = { width: 1600, height: 1000 };
await page.setViewportSize(VIEWPORT);

const settle = async (ms = 1200) => {
  await page.waitForTimeout(ms);
  await page.evaluate(() => document.fonts.ready);
};

const shot = async (name, options = {}) => {
  if (ONLY.length && !ONLY.includes(name)) return;
  // Each screen has a height at which it reads as one composition rather than
  // as a page that happens to have been cut somewhere.
  if (options.viewport) await page.setViewportSize(options.viewport);
  await settle(options.wait ?? 1200);
  await page.screenshot({ path: `${OUT}/${name}.png`, ...options.screenshot });
  if (options.viewport) await page.setViewportSize(VIEWPORT);
  console.log(`  ${name}.png`);
};

const go = async (path) => {
  await page.goto(`${BASE}${path}`, { waitUntil: "networkidle" });
};

console.log(`capturing ${BASE} -> docs/screens/`);

// ------------------------------------------------------------------- studio
if (!ONLY.length || ONLY.includes("studio")) {
  await go("/");
  // Cut after the first band of the archive: the point of the shot is the
  // form, the running job and that finished work sits under both, not a wall
  // of history.
  await shot("studio", { viewport: { width: 1500, height: 830 } });
}

// -------------------------------------------------------------------- models
if (!ONLY.length || ONLY.includes("models")) {
  await go("/models");
  await shot("models", { viewport: { width: 1500, height: 1000 } });
}

// ---------------------------------------------------------------- web models
if (!ONLY.length || ONLY.includes("web-models")) {
  await go("/models");
  await page.getByRole("button", { name: /web models/i }).click();
  await page.getByRole("button", { name: /^runware$/i }).click();
  await shot("web-models", { viewport: { width: 1500, height: 1000 }, wait: 3000 });
}

// -------------------------------------------------------------------- picker
if (!ONLY.length || ONLY.includes("picker")) {
  await go("/");
  await page.getByRole("button", { name: /other model/i }).click();
  const dialog = page.getByRole("dialog");
  await dialog.waitFor();
  // The key check is a real request to each provider and the mark it leaves is
  // half the point of the shot, so wait for the verdict rather than for a
  // number of seconds.
  await dialog
    .getByText(/checking/i)
    .first()
    .waitFor({ state: "hidden", timeout: 30_000 })
    .catch(() => console.log("  (a key check did not settle in time)"));
  // Short enough that the archive behind the dimmed ground is the band this
  // README already shows, and not whatever else is in the operator's history.
  await shot("picker", { viewport: { width: 1500, height: 720 }, wait: 600 });
  await page.keyboard.press("Escape");
}

// ---------------------------------------------------------------- job detail
if (!ONLY.length || ONLY.includes("job")) {
  await go("/");
  const cell = page.locator('a[href^="/j/"]').first();
  if (await cell.count()) {
    await cell.click();
    await shot("job", { viewport: { width: 1500, height: 950 }, wait: 2500 });
  } else {
    console.log("  (no finished job in the archive; skipping job.png)");
  }
}

await browser.close();
if (problems.length) {
  console.error("\nconsole errors while capturing:");
  for (const problem of problems.slice(0, 10)) console.error("  " + problem);
  process.exitCode = 1;
}
