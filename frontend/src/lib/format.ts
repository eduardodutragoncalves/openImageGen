/** Formatting for measurements. Every one of these is read while it changes,
 *  so nothing here may vary in width from one tick to the next. */

export function bytes(value: number): string {
  if (value < 1024) return `${value} B`;
  const units = ["KB", "MB", "GB", "TB"];
  let scaled = value / 1024;
  let unit = 0;
  while (scaled >= 1024 && unit < units.length - 1) {
    scaled /= 1024;
    unit += 1;
  }
  return `${scaled.toFixed(scaled >= 100 ? 0 : 1)} ${units[unit]}`;
}

export function gigabytes(mb: number): string {
  return `${(mb / 1024).toFixed(1)}`;
}

/** Compact and monotonic: 4s, 1:04, 12:04, 1:02:04. */
export function duration(seconds: number | null | undefined): string {
  if (seconds == null || !Number.isFinite(seconds)) return "—";
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total}s`;
  const h = Math.floor(total / 3600);
  const m = Math.floor((total % 3600) / 60);
  const s = total % 60;
  const pad = (n: number) => String(n).padStart(2, "0");
  return h > 0 ? `${h}:${pad(m)}:${pad(s)}` : `${m}:${pad(s)}`;
}

export function clockTime(epochSeconds: number): string {
  return new Date(epochSeconds * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
  });
}

export function shortDate(epochSeconds: number): string {
  const date = new Date(epochSeconds * 1000);
  const today = new Date();
  const sameDay = date.toDateString() === today.toDateString();
  return sameDay
    ? date.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" })
    : date.toLocaleDateString([], { day: "2-digit", month: "short" });
}

export function percent(value: number | null | undefined): string {
  if (value == null) return "—";
  return `${Math.round(value * 100)}`;
}

export function megapixels(pixels: number): string {
  return `${(pixels / 1_000_000).toFixed(1)}`;
}

/**
 * A provider's price for one image, in USD.
 *
 * Four decimals because a single image routinely bills fractions of a cent,
 * and a price rounded to "$0.00" reads as free rather than as cheap. Returns
 * null when nothing was quoted: an absent price and a zero price are
 * different claims, and only the caller can decide how to show the absence.
 */
export function usd(value: number | null | undefined): string | null {
  if (value == null || !Number.isFinite(value) || value <= 0) return null;
  return value < 0.01 ? `$${value.toFixed(4)}` : `$${value.toFixed(2)}`;
}
