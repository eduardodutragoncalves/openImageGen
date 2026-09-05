import { useCallback, useEffect, useState } from "react";

/**
 * The width of a resizable column, remembered.
 *
 * 380px was a guess that suited one screen. A prompt worth writing wants a
 * wider field; a session spent reading the archive wants a narrower one. The
 * guess stays the default, and the operator's correction to it outlives the
 * tab, because a size you have to set again every morning is not a setting.
 */
export const SPLIT = { min: 288, max: 760, initial: 380 } as const;

function clamp(value: number) {
  return Math.min(SPLIT.max, Math.max(SPLIT.min, Math.round(value)));
}

export function useSplit(key: string) {
  const [width, setWidth] = useState(() => {
    try {
      const saved = Number(localStorage.getItem(key));
      return saved ? clamp(saved) : SPLIT.initial;
    } catch {
      return SPLIT.initial;
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(key, String(width));
    } catch {
      /* the width still holds for this session */
    }
  }, [key, width]);

  // Clamped here rather than at every call site, so no caller can store a
  // width that leaves the panel unusable.
  const set = useCallback((next: number) => setWidth(clamp(next)), []);
  const reset = useCallback(() => setWidth(SPLIT.initial), []);

  return { width, set, reset } as const;
}
