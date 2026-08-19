import { Link } from "react-router-dom";
import type { JobSummary } from "../lib/api";
import { duration, percent } from "../lib/format";
import { useEta } from "../hooks/useApi";
import { Readout, SegmentBar, StateMark } from "./primitives";
import { IconArrowRight } from "./Icons";

/**
 * What is happening right now.
 *
 * The wait is the product: this panel exists so five minutes of silence never
 * happens. Progress is per-step and truthful, the estimate is measured from
 * this run rather than assumed, and a queued job says what it is waiting
 * behind. Nothing here blocks composing the next one.
 */
export function ActiveJob({ jobs }: { jobs: JobSummary[] }) {
  const running = jobs.find((job) => job.status === "running");
  const queued = jobs.filter((job) => job.status === "queued");

  if (!running && queued.length === 0) {
    return (
      <div className="flex h-full items-center gap-5 px-4 py-4">
        <div className="numeral text-4xl text-[var(--ink-faint)]">IDLE</div>
        <p className="max-w-[52ch] text-xs leading-relaxed text-[var(--ink-muted)]">
          The GPUs are free. Anything you submit starts immediately.
        </p>
      </div>
    );
  }

  return (
    <div className="flex h-full min-h-0 flex-col">
      {running ? <RunningJob job={running} /> : null}
      {queued.length > 0 ? (
        <ol className="shrink-0 border-t border-[var(--rule)]">
          {queued.map((job, index) => (
            <li
              key={job.id}
              className="flex items-center gap-3 border-b border-[var(--rule)] px-4 py-2 last:border-b-0"
            >
              <span className="font-mono text-[11px] tabular text-[var(--ink-faint)]">
                {String(index + 1).padStart(2, "0")}
              </span>
              <StateMark state="queued" />
              <span className="min-w-0 flex-1 truncate text-xs text-[var(--ink-muted)]">
                {job.prompt}
              </span>
              <span className="text-[10px] uppercase tracking-[0.1em] text-[var(--ink-faint)]">
                {index === 0 && running ? "next" : `${index + (running ? 1 : 0)} ahead`}
              </span>
            </li>
          ))}
        </ol>
      ) : null}
    </div>
  );
}

function RunningJob({ job }: { job: JobSummary }) {
  const eta = useEta(job.id, job.progress);
  const progress = job.progress ?? 0;
  const totalSteps = (job.num_steps ?? 0) * (job.num_images ?? 1);
  const currentStep = Math.max(1, Math.round(progress * totalSteps));
  const elapsed = job.started ? Math.max(0, Date.now() / 1000 - job.started) : null;

  return (
    <div className="flex min-h-0 flex-1 flex-col justify-center gap-4 px-4 py-5">
      <div className="flex items-start justify-between gap-6">
        <div className="flex items-baseline gap-6">
          <Readout label="complete" value={percent(progress)} unit="%" size="xl" tone="accent" />
          {totalSteps > 0 ? (
            <Readout
              label="step"
              value={
                <>
                  {currentStep}
                  <span className="text-[var(--ink-faint)]">/{totalSteps}</span>
                </>
              }
              size="lg"
            />
          ) : null}
          <Readout label="elapsed" value={duration(elapsed)} size="md" tone="muted" />
          <Readout
            label="remaining"
            value={eta == null ? "measuring" : duration(eta)}
            size="md"
            tone={eta == null ? "muted" : "ink"}
          />
        </div>
        <Link
          to={`/j/${job.id}`}
          className="btn h-8 no-underline"
          aria-label="Open this job"
        >
          <span>Open</span>
          <IconArrowRight size={14} />
        </Link>
      </div>

      <SegmentBar ratio={progress} segments={48} height={14} tone="accent" />

      <div className="flex items-baseline gap-4">
        <StateMark state="running" />
        <p className="min-w-0 flex-1 truncate text-xs text-[var(--ink-muted)]" title={job.prompt}>
          {job.prompt}
        </p>
        <span className="font-mono text-[10px] tabular text-[var(--ink-faint)]">
          {job.width}×{job.height} · {job.model_label ?? job.model_id}
        </span>
      </div>
    </div>
  );
}
