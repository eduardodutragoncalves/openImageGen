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

  test("the curated catalog browses without a key", async ({ page }) => {
    await unlock(page);
    await page.goto("/models");
    await page.getByRole("button", { name: /web models/i }).click();
    await page.getByRole("button", { name: /^runware$/i }).click();

    // No key in this environment. The curated catalog is public, so there is
    // something to look at, and no alarm.
    await expect(page.getByRole("listitem").first()).toBeVisible({ timeout: 30_000 });
    await expect(page.getByRole("alert")).toHaveCount(0);
    await expect(page.getByLabel(/runware api key/i)).toBeVisible();

    // It offers only the filters Runware can honour.
    await expect(page.getByRole("button", { name: /image generators/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /community models/i })).toBeVisible();
    await expect(page.getByRole("button", { name: /text models/i })).toHaveCount(0);
  });

  test("the community mirror waits for a query rather than guessing", async ({ page }) => {
    await unlock(page);
    await page.goto("/models");
    await page.getByRole("button", { name: /web models/i }).click();
    await page.getByRole("button", { name: /^runware$/i }).click();
    await page.getByRole("button", { name: /community models/i }).click();

    // Hundreds of thousands of checkpoints: there is no default view, and the
    // page says what to do instead of firing a request that cannot be useful.
    await expect(page.getByText(/search it by name or AIR id/i)).toBeVisible();
    await expect(page.getByRole("alert")).toHaveCount(0);
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

test.describe("the model picker", () => {
  test("one dialog holds every way to answer 'what makes this picture'", async ({ page }) => {
    await unlock(page);
    await page.getByRole("button", { name: /other model/i }).click();

    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();

    // The three kinds of answer sit side by side rather than in three places.
    const sources = dialog.getByRole("navigation", { name: /model sources/i });
    await expect(sources.getByRole("button", { name: /on this machine/i })).toBeVisible();
    await expect(sources.getByRole("button", { name: /hugging face/i })).toBeVisible();
    await expect(sources.getByRole("button", { name: /openrouter/i })).toBeVisible();
    await expect(sources.getByRole("button", { name: /runware/i })).toBeVisible();

    // It opens on what is already here, and says which one is loaded.
    await expect(dialog.getByText(/FLUX/).first()).toBeVisible();
    await expect(dialog.getByText(/^loaded$/i)).toBeVisible();
  });

  test("a model that will not fit is listed with the reason, not hidden", async ({ page }) => {
    await unlock(page);
    await page.getByRole("button", { name: /other model/i }).click();
    const dialog = page.getByRole("dialog");

    await dialog.getByRole("textbox", { name: /search models/i }).fill("bf16");
    // The entry stays, because "why can't I pick that?" deserves an answer.
    await expect(dialog.getByText(/bf16/i).first()).toBeVisible();
  });

  test("the hub is searched, not browsed", async ({ page }) => {
    await unlock(page);
    await page.getByRole("button", { name: /other model/i }).click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: /hugging face/i }).click();

    // Nothing is fetched, and nothing is downloaded, until you say what you want.
    await expect(dialog.getByText(/nothing is downloaded until you load it/i)).toBeVisible();

    await dialog.getByRole("textbox", { name: /search models/i }).fill("flux");
    await expect(dialog.getByText(/black-forest-labs/).first()).toBeVisible({
      timeout: 30_000,
    });
    await expect(dialog.getByText(/downloads/).first()).toBeVisible();
  });

  test("a provider with no key says so before you pick from it", async ({ page }) => {
    await unlock(page);
    await page.getByRole("button", { name: /other model/i }).click();
    const dialog = page.getByRole("dialog");

    // The claim is about the credential, and it is made where the choice is.
    await expect(
      dialog.getByRole("navigation", { name: /model sources/i }).getByText(/no key/i).first(),
    ).toBeVisible();
  });

  test("picking a provider model puts it on the form", async ({ page }) => {
    await unlock(page);
    await page.getByRole("button", { name: /other model/i }).click();
    const dialog = page.getByRole("dialog");
    await dialog.getByRole("button", { name: /openrouter/i }).click();

    const first = dialog.getByRole("listitem").first();
    await expect(first).toBeVisible({ timeout: 30_000 });
    const name = (await first.locator("span").first().textContent()) ?? "";
    await first.getByRole("button", { name: /use it/i }).click();

    // The dialog closes onto the choice it was opened to make.
    await expect(dialog).toHaveCount(0);
    await expect(
      page.getByRole("button", { name: new RegExp(name.slice(0, 12), "i") }),
    ).toBeVisible();
  });
});
