import { expect, test } from "@playwright/test";

const KEY = "e2e-key";

async function unlock(page: import("@playwright/test").Page) {
  await page.goto("/");
  await page.getByLabel(/api key/i).fill(KEY);
  await page.getByRole("button", { name: /unlock/i }).click();
  await expect(page.getByRole("link", { name: /^studio$/i })).toBeVisible();
}

test.describe("web models", () => {
  test("the catalog is filtered to models that can actually make an image", async ({ page }) => {
    await unlock(page);
    await page.getByRole("link", { name: /^models$/i }).click();
    await page.getByRole("button", { name: /web models/i }).click();

    // The filter is the substance of the tab: hundreds of models, a handful of
    // generators, and the page says so in as many words.
    await expect(page.getByText(/of \d+ models on OpenRouter can output an image/i)).toBeVisible({
      timeout: 30_000,
    });

    const rows = page.getByRole("listitem").filter({ hasText: "makes images" });
    expect(await rows.count()).toBeGreaterThan(0);

    // Switching to text models must widen the set, not narrow it.
    const imageTotal = await page.getByRole("listitem").count();
    await page.getByRole("button", { name: /text models/i }).click();
    await expect
      .poll(async () => page.getByRole("listitem").count(), { timeout: 30_000 })
      .toBeGreaterThan(imageTotal);
  });

  test("pinning a model makes it a target in the compose form", async ({ page }) => {
    await unlock(page);
    await page.goto("/models");
    await page.getByRole("button", { name: /web models/i }).click();
    await expect(page.getByText(/can output an image/i)).toBeVisible({ timeout: 30_000 });

    const first = page.getByRole("listitem").first();
    const name = (await first.locator("span").first().textContent()) ?? "";
    await first.getByRole("button", { name: /^pin$/i }).click();
    await expect(first.getByRole("button", { name: /^unpin$/i })).toBeVisible();

    await page.getByRole("link", { name: /^studio$/i }).click();
    await expect(page.getByText(/generate with/i)).toBeVisible();
    await expect(page.getByRole("button", { name: new RegExp(name.slice(0, 12), "i") })).toBeVisible();

    // And unpinning takes it away again.
    await page.goto("/models");
    await page.getByRole("button", { name: /web models/i }).click();
    await expect(page.getByText(/can output an image/i)).toBeVisible({ timeout: 30_000 });
    await page.getByRole("button", { name: /^unpin$/i }).first().click();
    await expect(page.getByRole("button", { name: /^unpin$/i })).toHaveCount(0);
  });

  test("a provider key is stored, reported, and never handed back", async ({ page }) => {
    await unlock(page);
    await page.goto("/models");
    await page.getByRole("button", { name: /web models/i }).click();

    await expect(page.getByText(/^not set$/i)).toBeVisible();
    await page.getByLabel(/openrouter api key/i).fill("sk-or-e2e-secret");
    await page.getByRole("button", { name: /^save$/i }).click();

    await expect(page.getByText(/stored on this server/i)).toBeVisible();
    // The page must never render the credential back.
    expect(await page.content()).not.toContain("sk-or-e2e-secret");

    await page.getByRole("button", { name: /^remove$/i }).click();
    await expect(page.getByText(/^not set$/i)).toBeVisible();
  });

  test("choosing openrouter upsampling opens the model picker", async ({ page }) => {
    await unlock(page);
    await page.getByRole("button", { name: /^openrouter$/i }).click();

    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    await expect(dialog.getByText(/rewrite the prompt with/i)).toBeVisible();
    // No key configured yet, so the dialog says the rewrite would fail.
    await expect(dialog.getByText(/no API key yet/i)).toBeVisible();

    await expect(dialog.getByRole("listitem").first()).toBeVisible({ timeout: 30_000 });
    await dialog.getByRole("listitem").first().click();
    await expect(dialog).toHaveCount(0);
    // The chosen model is shown on the form, not buried in state.
    await expect(page.getByRole("button", { name: /change/i })).toBeVisible();
  });
});

test.describe("reuse", () => {
  test("reuse restores every setting, not just the prompt", async ({ page }) => {
    await unlock(page);

    const prompt = `a lighthouse in fog ${Date.now()}`;
    await page.getByLabel(/^prompt$/i).fill(prompt);
    await page.getByRole("textbox", { name: /^seed$/i }).fill("4242");
    await page.getByRole("button", { name: /increase steps/i }).click();
    await page.getByRole("button", { name: /^3$/ }).click(); // three images
    await page.getByRole("button", { name: /^generate$/i }).click();

    const cell = page.getByRole("link", { name: new RegExp(prompt.slice(0, 18)) }).first();
    await expect(cell).toBeVisible({ timeout: 40_000 });
    await cell.click();

    // The detail page shows the request, and reuse puts all of it back.
    await expect(page.getByText(/^steps$/i)).toBeVisible();
    await page.getByRole("button", { name: /reuse these settings/i }).click();

    await expect(page.getByLabel(/^prompt$/i)).toHaveValue(prompt);
    await expect(page.getByRole("textbox", { name: /^seed$/i })).toHaveValue("4242");
    await expect(page.getByRole("button", { name: /^3$/ })).toHaveAttribute(
      "aria-pressed",
      "true",
    );
  });
});
