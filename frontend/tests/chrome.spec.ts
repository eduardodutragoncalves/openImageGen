import { expect, test } from "@playwright/test";

const KEY = "e2e-key";

async function unlock(page: import("@playwright/test").Page) {
  await page.goto("/");
  await page.getByLabel(/api key/i).fill(KEY);
  await page.getByRole("button", { name: /unlock/i }).click();
  await expect(page.getByRole("link", { name: /^studio$/i })).toBeVisible();
}

/** The compose column is the first section on the studio. */
function composeColumn(page: import("@playwright/test").Page) {
  return page.locator("section").first();
}

async function composeWidth(page: import("@playwright/test").Page) {
  const box = await composeColumn(page).boundingBox();
  return Math.round(box!.width);
}

test.describe("the ground", () => {
  test("the four grounds are offered, and the picked one is marked", async ({ page }) => {
    await unlock(page);
    const swatches = page.getByRole("radiogroup", { name: /ground/i }).getByRole("radio");
    await expect(swatches).toHaveCount(4);
    // Dark with the grid is what an operator gets before choosing anything.
    await expect(swatches.nth(0)).toHaveAttribute("aria-checked", "true");
  });

  test("black and white drop the grid; the grid ones keep it", async ({ page }) => {
    await unlock(page);
    const swatches = page.getByRole("radiogroup", { name: /ground/i }).getByRole("radio");
    const ground = () =>
      page.evaluate(() => {
        const style = getComputedStyle(document.body);
        return {
          colour: style.backgroundColor,
          gridded: style.backgroundImage !== "none",
          theme: document.documentElement.dataset.theme,
        };
      });

    await swatches.nth(2).click(); // Black
    expect(await ground()).toEqual({ colour: "rgb(0, 0, 0)", gridded: false, theme: "dark" });

    await swatches.nth(3).click(); // White
    expect(await ground()).toEqual({
      colour: "rgb(255, 255, 255)",
      gridded: false,
      theme: "light",
    });

    // The grid is still there for whoever wants the armature back.
    await swatches.nth(1).click(); // Sheet
    const sheet = await ground();
    expect(sheet.gridded).toBe(true);
    expect(sheet.theme).toBe("light");
  });

  test("the choice outlives the tab", async ({ page }) => {
    await unlock(page);
    await page.getByRole("radiogroup", { name: /ground/i }).getByRole("radio").nth(2).click();
    await page.reload();
    await expect(page.getByRole("link", { name: /^studio$/i })).toBeVisible();
    await expect(page.locator("html")).toHaveAttribute("data-ground", "flat");
    await expect(page.locator("html")).toHaveAttribute("data-theme", "dark");
  });
});

test.describe("the compose split", () => {
  test("the column is dragged wider and narrower, and stops before it is unusable", async ({
    page,
  }) => {
    await unlock(page);
    expect(await composeWidth(page)).toBe(380);

    const handle = page.getByRole("separator", { name: /compose width/i });
    const grip = (await handle.boundingBox())!;
    await page.mouse.move(grip.x + grip.width / 2, grip.y + 300);
    await page.mouse.down();
    await page.mouse.move(grip.x + grip.width / 2 + 220, grip.y + 300, { steps: 10 });
    await page.mouse.up();
    expect(await composeWidth(page)).toBe(600);

    // Dragged far past the left edge it clamps rather than collapsing to
    // nothing: a form you cannot read is not a smaller form.
    const moved = (await handle.boundingBox())!;
    await page.mouse.move(moved.x + moved.width / 2, moved.y + 300);
    await page.mouse.down();
    await page.mouse.move(moved.x - 900, moved.y + 300, { steps: 10 });
    await page.mouse.up();
    expect(await composeWidth(page)).toBe(288);
  });

  test("it is reachable from the keyboard", async ({ page }) => {
    await unlock(page);
    const handle = page.getByRole("separator", { name: /compose width/i });
    await handle.focus();

    await page.keyboard.press("ArrowRight");
    await page.keyboard.press("ArrowRight");
    expect(await composeWidth(page)).toBe(380 + 32);

    await page.keyboard.press("Shift+ArrowRight");
    expect(await composeWidth(page)).toBe(380 + 32 + 64);

    await page.keyboard.press("ArrowLeft");
    expect(await composeWidth(page)).toBe(380 + 32 + 64 - 16);
  });

  test("the width survives a reload, and a double-click puts it back", async ({ page }) => {
    await unlock(page);
    const handle = page.getByRole("separator", { name: /compose width/i });
    await handle.focus();
    await page.keyboard.press("Shift+ArrowRight");
    expect(await composeWidth(page)).toBe(444);

    await page.reload();
    await expect(page.getByRole("link", { name: /^studio$/i })).toBeVisible();
    expect(await composeWidth(page)).toBe(444);

    // The only way back for someone who dragged it somewhere they did not mean to.
    await page.getByRole("separator", { name: /compose width/i }).dblclick();
    expect(await composeWidth(page)).toBe(380);
  });
});

test.describe("clearing a GPU", () => {
  /** The e2e server shares one model across the whole run, so these exercise
   *  the UI contract and stop short of confirming. What actually gets unloaded
   *  is pinned in the Python suite, where a clear cannot strand later tests. */
  async function firstCard(page: import("@playwright/test").Page) {
    const card = page.getByRole("button", { name: /^gpu0/i });
    if ((await card.count()) === 0) test.skip(true, "this machine has no CUDA device");
    return card;
  }

  test("hovering a card says what is loaded on it", async ({ page }) => {
    await unlock(page);
    const card = await firstCard(page);

    await expect(page.getByRole("tooltip")).toHaveCount(0);
    await card.hover();

    const tip = page.getByRole("tooltip");
    await expect(tip).toBeVisible();
    await expect(tip).toContainText(/loaded here/i);
    // Either the model that is on it, or the plain statement that none is.
    await expect(tip).toContainText(/flux|no part of the model/i);
    await expect(tip).toContainText(/GB in use/i);
  });

  test("the panel is reachable without a pointer", async ({ page }) => {
    await unlock(page);
    const card = await firstCard(page);
    await card.focus();
    await expect(page.getByRole("tooltip")).toBeVisible();
    await card.blur();
    await expect(page.getByRole("tooltip")).toHaveCount(0);
  });

  test("clicking asks first, and says what clearing would cost", async ({ page }) => {
    await unlock(page);
    const card = await firstCard(page);
    await card.click();

    const dialog = page.getByRole("dialog");
    await expect(dialog).toBeVisible();
    await expect(dialog).toContainText(/clear gpu0/i);

    // A card carrying the model must say the unload is not card-local, since
    // that is the one thing an operator must not learn afterwards.
    if (await dialog.getByText(/this card is carrying/i).isVisible()) {
      await expect(dialog).toContainText(/unloads the model from/i);
      await expect(dialog).toContainText(/every/i);
      await expect(dialog).toContainText(/nothing is unloaded out from under a running job/i);
      await expect(
        dialog.getByRole("button", { name: /unload the model and clear/i }),
      ).toBeVisible();
    } else {
      await expect(dialog).toContainText(/nothing gets unloaded/i);
      await expect(dialog.getByRole("button", { name: /clear cached memory/i })).toBeVisible();
    }
  });

  test("keeping it changes nothing", async ({ page }) => {
    await unlock(page);
    const card = await firstCard(page);
    const before = await page.getByText(/^model$/i).locator("..").innerText();

    await card.click();
    await page.getByRole("button", { name: /keep it/i }).click();

    await expect(page.getByRole("dialog")).toHaveCount(0);
    expect(await page.getByText(/^model$/i).locator("..").innerText()).toBe(before);
  });
});

test.describe("the wait, made visible", () => {
  test("the generate button carries the shine, and drops it when it cannot be pressed", async ({
    page,
  }) => {
    await unlock(page);
    const generate = page.getByRole("button", { name: /^generate$/i });
    await expect(generate).toHaveClass(/btn-shiny/);

    // Empty prompt: the control is disabled, and a control that cannot be
    // used has no business drawing the eye.
    await expect(generate).toBeDisabled();
    const flat = await generate.evaluate((el) => getComputedStyle(el).backgroundImage);
    expect(flat).not.toContain("conic-gradient");

    await page.getByLabel(/^prompt$/i).fill("a heron");
    await expect(generate).toBeEnabled();
    const lit = await generate.evaluate((el) => getComputedStyle(el).backgroundImage);
    expect(lit).toContain("conic-gradient");
  });

  test("the cell of an image being made fills as the steps land", async ({ page }) => {
    await unlock(page);
    // A long run, so the reveal has a window to be observed in.
    await page.getByLabel(/^steps$/i).fill("100");
    await page.getByLabel(/^prompt$/i).fill(`a slow heron ${Date.now()}`);
    await page.getByRole("button", { name: /^generate$/i }).click();

    const field = page.locator("canvas").first();
    await expect(field).toBeVisible({ timeout: 20_000 });

    const reveal = () =>
      field.evaluate((el) => getComputedStyle(el.parentElement!).clipPath);
    // It starts closed and opens; the exact fractions belong to the job, so
    // this pins the direction rather than a number.
    await expect
      .poll(async () => {
        const inset = (await reveal()).match(/([\d.]+)%/);
        return inset ? Number(inset[1]) : 100;
      }, { timeout: 25_000, message: "the field never claimed any of the cell" })
      .toBeLessThan(100);
  });

  test("the panel beside a result names the model and the price", async ({ page }) => {
    await unlock(page);
    const prompt = `a priced heron ${Date.now()}`;
    await page.getByLabel(/^prompt$/i).fill(prompt);
    await page.getByRole("button", { name: /^generate$/i }).click();

    const cell = page.getByRole("link", { name: new RegExp(prompt.slice(0, 16)) }).first();
    await cell.click({ timeout: 40_000 });

    const panel = page.locator("dl").first();
    await expect(panel).toContainText(/model/i);
    await expect(panel).toContainText(/cost/i);
    // These GPUs bill nothing, and "—" is the honest answer: "$0.00" would
    // claim the run was free rather than unpriced.
    await expect(panel).toContainText("—");
  });
});
