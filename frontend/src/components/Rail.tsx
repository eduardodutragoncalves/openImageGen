import { NavLink } from "react-router-dom";
import type { Health } from "../lib/api";
import { gigabytes } from "../lib/format";
import { SegmentBar, Diagonal } from "./primitives";
import { IconMoon, IconSun } from "./Icons";

/**
 * The status rail. Fixed height, fixed positions: the operator learns where
 * each value lives and reads it without looking for it, which is the whole
 * argument for not letting this reflow.
 */
export function Rail({
  health,
  precision,
  theme,
  onToggleTheme,
}: {
  health?: Health;
  precision?: string;
  theme: "dark" | "light";
  onToggleTheme: () => void;
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

      <div className="hidden items-center gap-5 border-r border-[var(--rule)] px-4 lg:flex">
        {gpus.length === 0 ? (
          <span className="text-xs text-[var(--ink-faint)]">no CUDA device</span>
        ) : (
          gpus.map((gpu) => (
            <GpuTape
              key={gpu.index}
              index={gpu.index}
              name={gpu.name}
              usedMb={gpu.memory_used_mb}
              totalMb={gpu.memory_total_mb}
              role={gpu.role ?? undefined}
            />
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

      <button
        type="button"
        onClick={onToggleTheme}
        className="flex w-[var(--rail)] items-center justify-center border-l border-[var(--rule)] text-[var(--ink-muted)] transition-colors hover:text-[var(--accent-ink)]"
        aria-label={theme === "dark" ? "Switch to the light sheet" : "Switch to the dark ground"}
        title={theme === "dark" ? "Light" : "Dark"}
      >
        {theme === "dark" ? <IconSun /> : <IconMoon />}
      </button>

      {model?.state === "ready" ? null : <RailState state={model?.state} phase={model?.phase} />}
    </header>
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

function GpuTape({
  index,
  name,
  usedMb,
  totalMb,
  role,
}: {
  index: number;
  name: string;
  usedMb: number;
  totalMb: number;
  role?: string;
}) {
  const ratio = totalMb > 0 ? usedMb / totalMb : 0;
  // Pressure is legible before an OOM, not after it.
  const tone = ratio > 0.94 ? "alarm" : ratio > 0.82 ? "caution" : "accent";
  return (
    <div className="flex flex-col justify-center gap-[3px]" title={`${name}${role ? ` — ${role}` : ""}`}>
      <div className="flex items-baseline gap-2">
        <span className="label">gpu{index}</span>
        <span className="font-mono text-[10px] tabular text-[var(--ink-muted)]">
          {gigabytes(usedMb)}/{gigabytes(totalMb)} GB
        </span>
      </div>
      <SegmentBar ratio={ratio} segments={10} tone={tone} height={6} />
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
