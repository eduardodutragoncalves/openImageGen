/**
 * The server's pixel budget, mirrored so the form can show the real output
 * size before submitting instead of surprising the operator with a downscale
 * after a five-minute wait. Kept identical to `fit_to_budget` in
 * app/images.py; the cap itself always comes from /v1/models.
 */
export function fitToBudget(width: number, height: number, maxPixels: number) {
  let w = Math.max(16, 16 * Math.floor(width / 16));
  let h = Math.max(16, 16 * Math.floor(height / 16));
  if (w * h <= maxPixels) return { width: w, height: h, capped: false };

  const scale = Math.sqrt(maxPixels / (w * h));
  w = Math.max(16, 16 * Math.floor((w * scale) / 16));
  h = Math.max(16, 16 * Math.floor((h * scale) / 16));
  return { width: w, height: h, capped: true };
}

export interface AspectPreset {
  label: string;
  ratio: [number, number];
}

export const ASPECTS: AspectPreset[] = [
  { label: "1:1", ratio: [1, 1] },
  { label: "4:3", ratio: [4, 3] },
  { label: "3:4", ratio: [3, 4] },
  { label: "16:9", ratio: [16, 9] },
  { label: "9:16", ratio: [9, 16] },
];

/** The largest 16-aligned size at this aspect that fits the budget. */
export function sizeForAspect(ratio: [number, number], maxPixels: number) {
  const [rw, rh] = ratio;
  const unit = Math.sqrt(maxPixels / (rw * rh));
  return fitToBudget(Math.round(rw * unit), Math.round(rh * unit), maxPixels);
}
