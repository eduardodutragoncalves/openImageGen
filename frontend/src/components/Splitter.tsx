import { useRef } from "react";
import { SPLIT } from "../hooks/useSplit";

const STEP = 16;
const COARSE_STEP = 64; // one major cell of the construction grid

/**
 * The rule between two regions, made draggable.
 *
 * It sits *on* the rule rather than beside it: the hairline is already the
 * boundary, and putting a visible gutter next to it would add a second one.
 * The target is 7px wide even though the line is 1px, because a hairline is
 * something to look at, not something to hit.
 *
 * Dragged with the pointer, nudged with the arrows, and returned to the
 * default with a double-click — which is the only way back for someone who
 * dragged it somewhere they did not mean to.
 */
export function Splitter({
  width,
  onWidth,
  onReset,
  label,
}: {
  width: number;
  onWidth: (width: number) => void;
  onReset: () => void;
  label: string;
}) {
  // The drag is relative to where it started, not to the container's left
  // edge: no measuring, and it stays correct if anything to the left of the
  // panel changes size mid-drag.
  const from = useRef<{ x: number; width: number } | null>(null);

  const end = (event: React.PointerEvent<HTMLDivElement>) => {
    if (!from.current) return;
    from.current = null;
    delete document.documentElement.dataset.resizing;
    event.currentTarget.releasePointerCapture(event.pointerId);
  };

  return (
    <div
      role="separator"
      aria-orientation="vertical"
      aria-label={label}
      aria-valuenow={width}
      aria-valuemin={SPLIT.min}
      aria-valuemax={SPLIT.max}
      tabIndex={0}
      onPointerDown={(event) => {
        if (event.button !== 0) return;
        from.current = { x: event.clientX, width };
        document.documentElement.dataset.resizing = "col";
        event.currentTarget.setPointerCapture(event.pointerId);
      }}
      onPointerMove={(event) => {
        if (!from.current) return;
        onWidth(from.current.width + (event.clientX - from.current.x));
      }}
      onPointerUp={end}
      onPointerCancel={end}
      onDoubleClick={onReset}
      onKeyDown={(event) => {
        const step = event.shiftKey ? COARSE_STEP : STEP;
        if (event.key === "ArrowLeft") onWidth(width - step);
        else if (event.key === "ArrowRight") onWidth(width + step);
        else if (event.key === "Home") onWidth(SPLIT.min);
        else if (event.key === "End") onWidth(SPLIT.max);
        else if (event.key === "Enter") onReset();
        else return;
        event.preventDefault();
      }}
      title={`${label} — drag to resize, double-click to reset`}
      className="group absolute -right-[3px] top-0 z-20 hidden h-full w-[7px] cursor-col-resize touch-none xl:block"
    >
      {/* The line only announces itself under the pointer: at rest this is the
          same hairline every other region is separated by. */}
      <span
        aria-hidden
        className="absolute inset-y-0 left-[3px] w-px bg-transparent transition-colors group-hover:bg-[var(--accent)] group-focus-visible:bg-[var(--accent)]"
      />
    </div>
  );
}
