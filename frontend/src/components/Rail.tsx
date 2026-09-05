import { NavLink } from "react-router-dom";
import type { Health } from "../lib/api";
import { Diagonal } from "./primitives";
import { GpuCard } from "./GpuCard";
import { VISUALS } from "../hooks/useVisual";
import type { Visual } from "../hooks/useVisual";

/**
 * The status rail. Fixed height, fixed positions: the operator learns where
 * each value lives and reads it without looking for it, which is the whole
 * argument for not letting this reflow.
 */
export function Rail({
  health,
  precision,
  visual,
  onVisual,
}: {
  health?: Health;
  precision?: string;
  visual: Visual;
  onVisual: (visual: Visual) => void;
}) {
  const model = health?.model;
  const gpus = health?.gpus ?? [];

  return (
    <header className="flex h-[var(--rail)] shrink-0 items-stretch border-b border-[var(--rule)]">
      <div className="flex items-center gap-2 border-r border-[var(--rule)] px-4">
        <span className="text-[15px] font-bold tracking-[-0.01em]" style={{ fontStretch: "112%" }}>
          openImageGen
        </span>
        <Diagonal size={14} className="text-[var(--accent-ink)]" />
      </div>

      <ModelPlate health={health} precision={precision} />

      <div className="hidden items-center gap-3 border-r border-[var(--rule)] px-3 lg:flex">
        {gpus.length === 0 ? (
          <span className="text-xs text-[var(--ink-faint)]">no CUDA device</span>
        ) : (
          gpus.map((gpu) => (
            <GpuCard key={gpu.index} gpu={gpu} modelId={model?.model_id} />
          ))
        )}
      </div>

      <div className="hidden items-center gap-3 border-r border-[var(--rule)] px-4 md:flex">
        <span className="label">queue</span>
        <span className="font-mono text-sm tabular">
          {health ? health.queue_depth : "—"}
        </span>
        {health?.queue_paused ? (
          <span className="label label-accent">held</span>
        ) : null}
      </div>

      <nav className="ml-auto flex items-stretch" aria-label="Sections">
        <RailLink to="/">Studio</RailLink>
        <RailLink to="/models">Models</RailLink>
      </nav>

      <GroundPicker visual={visual} onVisual={onVisual} />

      {model?.state === "ready" ? null : <RailState state={model?.state} phase={model?.phase} />}
    </header>
  );
}

/**
 * The ground, picked rather than cycled. Four swatches showing the field each
 * one lays down: a cycling button would hide three of the four answers behind
 * a guess about how many presses it takes, and this is a choice about how the
 * work *looks*, which is the one thing you should be able to see before you
 * commit to it.
 */
function GroundPicker({
  visual,
  onVisual,
}: {
  visual: Visual;
  onVisual: (visual: Visual) => void;
}) {
  return (
    <div
      role="radiogroup"
      aria-label="Ground"
      className="flex items-stretch border-l border-[var(--rule)]"
    >
      {VISUALS.map((entry) => {
        const active = entry.id === visual;
        return (
          <button
            key={entry.id}
            type="button"
            role="radio"
            aria-checked={active}
            aria-label={entry.hint}
            title={`${entry.label} — ${entry.hint}`}
            onClick={() => onVisual(entry.id)}
            className="flex w-7 items-center justify-center transition-colors hover:bg-[var(--accent-wash)]"
          >
            <span
              aria-hidden
              className="h-4 w-4 border"
              style={{
                backgroundColor: entry.swatch.fill,
                // The 4px cell is the construction grid at swatch scale, so
                // the sample is the ground itself rather than a picture of it.
                backgroundImage: entry.swatch.line
                  ? `linear-gradient(to right, ${entry.swatch.line} 1px, transparent 1px),` +
                    `linear-gradient(to bottom, ${entry.swatch.line} 1px, transparent 1px)`
                  : undefined,
                backgroundSize: "4px 4px",
                borderColor: active ? "var(--accent)" : "var(--rule-strong)",
                outline: active ? "1px solid var(--accent)" : undefined,
                outlineOffset: "1px",
              }}
            />
          </button>
        );
      })}
    </div>
  );
}

function ModelPlate({ health, precision }: { health?: Health; precision?: string }) {
  const model = health?.model;
  const label = model?.model_id ?? "—";
  return (
    <div className="hidden min-w-[200px] flex-col justify-center gap-[3px] border-r border-[var(--rule)] px-4 sm:flex">
      <span className="label">model</span>
      <span className="flex items-baseline gap-2 font-mono text-xs tracking-tight text-[var(--ink)]">
        {label}
        {precision && precision !== "bf16" ? (
          <span className="text-[10px] text-[var(--ink-faint)]">{precision}</span>
        ) : null}
      </span>
    </div>
  );
}

function RailState({ state, phase }: { state?: string; phase?: string }) {
  const tone =
    state === "error" ? "text-[var(--alarm-ink)]" : "text-[var(--accent-ink)]";
  return (
    <div className="flex items-center gap-2 border-l border-[var(--rule)] px-4">
      <span className={`label ${tone}`}>{state ?? "starting"}</span>
      <span className="text-xs text-[var(--ink-muted)]">{phase}</span>
    </div>
  );
}

/**
 * The active section, marked by the specimen's slanted plane rather than by an
 * underline. The diagonal is this world's way of saying "here".
 */
function RailLink({ to, children }: { to: string; children: string }) {
  return (
    <NavLink
      to={to}
      end={to === "/"}
      className={({ isActive }) =>
        [
          "relative flex items-center border-l border-[var(--rule)] px-5 text-[11px] font-semibold uppercase tracking-[0.14em] transition-colors",
          isActive
            ? "text-[var(--ink-on-accent)]"
            : "text-[var(--ink-muted)] hover:text-[var(--ink)]",
        ].join(" ")
      }
    >
      {({ isActive }) => (
        <>
          {isActive ? (
            <span
              aria-hidden
              className="absolute inset-0 bg-[var(--accent)]"
              style={{ clipPath: "polygon(14px 0, 100% 0, calc(100% - 14px) 100%, 0 100%)" }}
            />
          ) : null}
          <span className="relative">{children}</span>
        </>
      )}
    </NavLink>
  );
}
