import type { ReactNode } from "react";

/**
 * A region of the console. Regions are separated by rules, never by boxes:
 * no radius, no shadow, no nested container. The label *is* the heading.
 */
export function Region({
  label,
  aside,
  children,
  className = "",
  bodyClassName = "",
}: {
  label: string;
  aside?: ReactNode;
  children: ReactNode;
  className?: string;
  bodyClassName?: string;
}) {
  return (
    <section className={`flex min-h-0 flex-col ${className}`}>
      <header className="flex h-8 shrink-0 items-center justify-between gap-4 border-b border-[var(--rule)] px-3">
        <h2 className="label">{label}</h2>
        {aside}
      </header>
      <div className={`min-h-0 flex-1 ${bodyClassName}`}>{children}</div>
    </section>
  );
}

/**
 * A measurement in its cell: monumental number, small label beneath. Used for
 * the values the operator reads while they change.
 */
export function Readout({
  label,
  value,
  unit,
  tone = "ink",
  size = "md",
}: {
  label: string;
  value: ReactNode;
  unit?: string;
  tone?: "ink" | "accent" | "caution" | "alarm" | "muted";
  size?: "sm" | "md" | "lg" | "xl";
}) {
  const tones = {
    ink: "text-[var(--ink)]",
    accent: "text-[var(--accent-ink)]",
    caution: "text-[var(--caution-ink)]",
    alarm: "text-[var(--alarm-ink)]",
    muted: "text-[var(--ink-faint)]",
  };
  const sizes = { sm: "text-base", md: "text-2xl", lg: "text-4xl", xl: "text-6xl" };
  return (
    <div className="flex flex-col gap-1">
      <div className={`numeral ${sizes[size]} ${tones[tone]}`}>
        {value}
        {unit ? (
          <span className="ml-1 align-baseline text-[0.4em] font-semibold tracking-[0.1em] text-[var(--ink-faint)]">
            {unit}
          </span>
        ) : null}
      </div>
      <div className="label">{label}</div>
    </div>
  );
}

/**
 * A quantity as filled cells on the construction grid, which is what this
 * world has instead of a rounded progress bar.
 */
export function SegmentBar({
  ratio,
  segments = 12,
  tone = "accent",
  height = 8,
  title,
}: {
  ratio: number;
  segments?: number;
  tone?: "accent" | "caution" | "alarm" | "ink";
  height?: number;
  title?: string;
}) {
  const filled = Math.round(Math.max(0, Math.min(1, ratio)) * segments);
  const colours = {
    accent: "var(--accent)",
    caution: "var(--caution)",
    alarm: "var(--alarm)",
    ink: "var(--ink)",
  };
  return (
    <div className="flex gap-[2px]" title={title} aria-hidden>
      {Array.from({ length: segments }, (_, index) => (
        <span
          key={index}
          style={{
            height,
            background: index < filled ? colours[tone] : "transparent",
            borderColor: index < filled ? colours[tone] : "var(--rule)",
          }}
          className="w-[6px] border"
        />
      ))}
    </div>
  );
}

const STATE_TONE: Record<string, { label: string; className: string }> = {
  queued: { label: "queued", className: "text-[var(--ink-muted)] border-[var(--rule-strong)]" },
  running: {
    label: "running",
    className: "bg-[var(--accent)] text-[var(--ink-on-accent)] border-[var(--accent)]",
  },
  succeeded: { label: "done", className: "text-[var(--ink-muted)] border-[var(--rule)]" },
  rejected: {
    label: "refused",
    className: "text-[var(--caution-ink)] border-[var(--caution)]",
  },
  failed: { label: "failed", className: "text-[var(--alarm-ink)] border-[var(--alarm)]" },
};

/** State as a plate. `rejected` is amber and reads as a refusal, not an error:
 *  the filter did its job, and the copy elsewhere says which one fired. */
export function StateMark({ state }: { state: string }) {
  const tone = STATE_TONE[state] ?? STATE_TONE.queued;
  return (
    <span
      className={`inline-flex h-5 items-center border px-[6px] text-[10px] font-semibold uppercase tracking-[0.12em] ${tone.className}`}
    >
      {tone.label}
    </span>
  );
}

/** The 45° mark, sized to a cell. */
export function Diagonal({ size = 16, className = "" }: { size?: number; className?: string }) {
  return (
    <span
      aria-hidden
      className={`diagonal shrink-0 ${className}`}
      style={{ width: size, height: size }}
    />
  );
}

export function Field({
  label,
  hint,
  children,
  htmlFor,
}: {
  label: string;
  hint?: ReactNode;
  children: ReactNode;
  htmlFor?: string;
}) {
  return (
    <div className="flex flex-col gap-[6px]">
      <div className="flex items-baseline justify-between gap-2">
        <label className="label" htmlFor={htmlFor}>
          {label}
        </label>
        {hint ? <span className="text-[10px] text-[var(--ink-faint)]">{hint}</span> : null}
      </div>
      {children}
    </div>
  );
}

/** An area with nothing in it yet, carrying the specimen's dotted texture
 *  rather than an illustration or an apology. */
export function EmptyField({
  title,
  detail,
  action,
}: {
  title: string;
  detail: string;
  action?: ReactNode;
}) {
  return (
    <div className="dotted flex h-full min-h-[160px] flex-col items-center justify-center gap-2 border border-[var(--rule)] p-6 text-center">
      <p className="text-sm font-semibold text-[var(--ink)]">{title}</p>
      <p className="max-w-[46ch] text-xs text-[var(--ink-muted)]">{detail}</p>
      {action}
    </div>
  );
}
