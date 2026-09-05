import { useEffect, useState } from "react";

/**
 * The look of the world: a palette and a ground, chosen together.
 *
 * These were two ideas until now — the palette was the operator's and the grid
 * was not negotiable. But the grid is a second image behind the first, and a
 * photograph judged against a graticule is judged against the graticule too.
 * So the ground flattens, and because "dark" and "black" are different claims
 * the flat ones are pure rather than merely dim.
 *
 * They are one control rather than two because only four of the combinations
 * mean anything, and a matrix of switches for four answers is a puzzle.
 */
export type Visual = "grid-dark" | "grid-light" | "black" | "white";

export interface VisualEntry {
  id: Visual;
  label: string;
  /** What the switch does, said in full for the screen reader and the tooltip. */
  hint: string;
  theme: "dark" | "light";
  ground: "grid" | "flat";
  /** The swatch, in fixed colours: it has to show the ground it *would* set,
   *  not the one currently in force, so it cannot use the palette tokens. */
  swatch: { fill: string; line: string | null };
}

export const VISUALS: readonly VisualEntry[] = [
  {
    id: "grid-dark",
    label: "Grid",
    hint: "Construction grid on the dark ground",
    theme: "dark",
    ground: "grid",
    swatch: { fill: "#08090b", line: "rgb(242 245 250 / 0.34)" },
  },
  {
    id: "grid-light",
    label: "Sheet",
    hint: "Construction grid on the light sheet",
    theme: "light",
    ground: "grid",
    swatch: { fill: "#f5f7fa", line: "rgb(17 17 17 / 0.26)" },
  },
  {
    id: "black",
    label: "Black",
    hint: "Flat black, no grid",
    theme: "dark",
    ground: "flat",
    swatch: { fill: "#000000", line: null },
  },
  {
    id: "white",
    label: "White",
    hint: "Flat white, no grid",
    theme: "light",
    ground: "flat",
    swatch: { fill: "#ffffff", line: null },
  },
];

const KEY = "oig-visual";
const LEGACY_KEY = "oig-theme";

function stored(): Visual {
  try {
    const saved = localStorage.getItem(KEY);
    if (VISUALS.some((entry) => entry.id === saved)) return saved as Visual;
    // Before the ground was a choice, only the palette was one. An operator
    // who already picked light keeps it rather than being reset to the default.
    return localStorage.getItem(LEGACY_KEY) === "light" ? "grid-light" : "grid-dark";
  } catch {
    // A browser with site data blocked still gets a studio, just not a memory.
    return "grid-dark";
  }
}

export function entryFor(visual: Visual): VisualEntry {
  return VISUALS.find((entry) => entry.id === visual) ?? VISUALS[0];
}

/** Dark with the grid is the default because of the room this runs in, not the
 *  category: a workstation beside the rig, judging photographs for hours. */
export function useVisual() {
  const [visual, setVisual] = useState<Visual>(stored);

  useEffect(() => {
    const entry = entryFor(visual);
    document.documentElement.dataset.theme = entry.theme;
    document.documentElement.dataset.ground = entry.ground;
    try {
      localStorage.setItem(KEY, visual);
    } catch {
      /* the choice still holds for this session */
    }
  }, [visual]);

  return [visual, setVisual] as const;
}
