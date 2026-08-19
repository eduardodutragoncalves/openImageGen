import { expect, test } from "@playwright/test";

const KEY = "e2e-key";

async function unlock(page: import("@playwright/test").Page) {
  await page.goto("/");
  await page.getByLabel(/api key/i).fill(KEY);
  await page.getByRole("button", { name: /unlock/i }).click();
  await expect(page.getByRole("link", { name: /^studio$/i })).toBeVisible();
}

test.describe("the critical path", () => {
  test("a key is required before anything is reachable", async ({ page }) => {
    await page.goto("/");
    await expect(page.getByLabel(/api key/i)).toBeVisible();
    await expect(page.getByRole("link", { name: /^studio$/i })).toHaveCount(0);

    await page.getByLabel(/api key/i).fill("wrong-key");
    await page.getByRole("button", { name: /unlock/i }).click();
    await expect(page.getByRole("alert")).toContainText("not recognised");
  });

  test("submit, track and retrieve", async ({ page }) => {
    await unlock(page);

    const prompt = `a heron at dawn ${Date.now()}`;
    await page.getByLabel(/^prompt$/i).fill(prompt);
    await page.getByRole("button", { name: /^generate$/i }).click();

    // Running, with real per-step progress rather than a spinner.
    await expect(page.getByText(/^step$/i).first()).toBeVisible();
    await expect(page.getByText(/^complete$/i).first()).toBeVisible();

    // The form stays usable while the GPU is busy: queueing more work is the
    // normal case, not an edge one.
    await expect(page.getByLabel(/^prompt$/i)).toBeEditable();

    // It lands, and it lands in the archive with its settings intact.
    const cell = page.getByRole("link", { name: new RegExp(prompt.slice(0, 20)) }).first();
    await expect(cell).toBeVisible({ timeout: 40_000 });

    await cell.click();
    await expect(page.getByText(/^seed$/i).first()).toBeVisible();
    await expect(page.getByText(prompt)).toBeVisible();
    await expect(page.getByRole("link", { name: /download/i })).toBeVisible();

    // The permalink survives a reload: that is the answer to "I lost the id".
    const permalink = page.url();
    expect(permalink).toContain("/j/");
    await page.reload();
    await expect(page.getByText(prompt)).toBeVisible();
  });

  test("the archive filters and searches", async ({ page }) => {
    await unlock(page);
    const unique = `violin interior ${Date.now()}`;
    await page.getByLabel(/^prompt$/i).fill(unique);
    await page.getByRole("button", { name: /^generate$/i }).click();
    await expect(
      page.getByRole("link", { name: new RegExp(unique.slice(0, 18)) }).first(),
    ).toBeVisible({ timeout: 40_000 });

    await page.getByLabel(/search prompts/i).fill("violin interior");
    await page.getByLabel(/search prompts/i).blur();
    await expect(page.getByRole("link", { name: /violin interior/ }).first()).toBeVisible();

    await page.getByLabel(/search prompts/i).fill("nothing matches this at all");
    await page.getByLabel(/search prompts/i).blur();
    await expect(page.getByText(/nothing matches those filters/i)).toBeVisible();
  });

  test("the model catalog shows what will not fit, with the reason", async ({ page }) => {
    await unlock(page);
    await page.getByRole("link", { name: /^models$/i }).click();

    await expect(page.getByRole("heading", { name: /^models$/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /^flux\.2$/i })).toBeVisible();
    await expect(page.getByRole("heading", { name: /^flux\.1$/i })).toBeVisible();

    // The 113GB checkpoint is listed rather than hidden, and says why.
    const oversized = page.getByRole("listitem").filter({ hasText: "FLUX.2 [dev] bf16" });
    await expect(oversized.getByRole("button", { name: /will not fit/i })).toBeDisabled();
    await expect(oversized).toContainText("the transformer alone needs");

    // Exactly one model is marked loaded.
    await expect(page.getByRole("button", { name: /^loaded$/i })).toHaveCount(1);
  });

  test("switching model is a state, not a silent reload", async ({ page }) => {
    await unlock(page);
    await page.goto("/models");

    const target = page.getByRole("listitem").filter({ hasText: "FLUX.1 [schnell]" });
    await target.getByRole("button", { name: /load this model/i }).click();

    // The rail reports the new model once the swap settles.
    await expect(page.getByText("flux1-schnell").first()).toBeVisible({ timeout: 30_000 });
  });

  test("history is scoped to the key that made it", async ({ page, context }) => {
    await unlock(page);
    const mine = `private prompt ${Date.now()}`;
    await page.getByLabel(/^prompt$/i).fill(mine);
    await page.getByRole("button", { name: /^generate$/i }).click();
    await expect(page.getByRole("link", { name: new RegExp(mine.slice(0, 15)) }).first()).toBeVisible({
      timeout: 40_000,
    });

    await context.clearCookies();
    await page.goto("/");
    await page.getByLabel(/api key/i).fill("other-key");
    await page.getByRole("button", { name: /unlock/i }).click();
    await expect(page.getByRole("link", { name: /^studio$/i })).toBeVisible();
    await expect(page.getByText(mine)).toHaveCount(0);
  });
});
