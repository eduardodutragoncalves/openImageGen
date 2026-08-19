import { useState } from "react";
import { Link } from "react-router-dom";
import type { CatalogEntry, JobSummary } from "../lib/api";
import type { ArchiveFilters } from "../hooks/useApi";
import { shortDate } from "../lib/format";
import { EmptyField, StateMark } from "./primitives";
import { IconCaution, IconSearch } from "./Icons";

const STATES = ["succeeded", "running", "queued", "rejected", "failed"] as const;

/**
 * The archive. Hundreds of rows a week, so it is a dense grid rather than a
 * list, and every cell carries its own designed label strip: prompt, seed,
 * size and model read at grid density without hovering anything.
 */
export function Archive({
  jobs,
  total,
  filters,
  onFilters,
  models,
  loading,
  onLoadMore,
  canLoadMore,
}: {
  jobs: JobSummary[];
  total: number;
  filters: ArchiveFilters;
  onFilters: (next: ArchiveFilters) => void;
  models: CatalogEntry[];
  loading: boolean;
  onLoadMore: () => void;
  canLoadMore: boolean;
}) {
  const [draft, setDraft] = useState(filters.search ?? "");
  const usedModels = models.filter((model) =>
    jobs.some((job) => job.model_id === model.id),
  );

  return (
    <div className="flex h-full min-h-0 flex-col">
      <div className="flex shrink-0 flex-wrap items-center gap-2 border-b border-[var(--rule)] px-3 py-2">
        <form
          className="relative flex-1 min-w-[180px]"
          onSubmit={(event) => {
            event.preventDefault();
            onFilters({ ...filters, search: draft.trim() || undefined });
          }}
        >
          <IconSearch
            size={14}
            className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-[var(--ink-faint)]"
          />
          <input
            value={draft}
            onChange={(event) => setDraft(event.target.value)}
            onBlur={() => onFilters({ ...filters, search: draft.trim() || undefined })}
            placeholder="Search prompts"
            aria-label="Search prompts"
            className="field h-8 pl-7 text-xs"
          />
        </form>

        <FilterGroup
          value={filters.status}
          options={STATES.map((state) => ({ value: state, label: state }))}
          onChange={(status) => onFilters({ ...filters, status })}
        />

        {usedModels.length > 1 ? (
          <FilterGroup
            value={filters.model_id}
            options={usedModels.map((model) => ({ value: model.id, label: model.label }))}
            onChange={(model_id) => onFilters({ ...filters, model_id })}
          />
        ) : null}

        <span className="ml-auto pr-1 font-mono text-[10px] tabular text-[var(--ink-faint)]">
          {jobs.length} of {total}
        </span>
      </div>

      <div className="min-h-0 flex-1 overflow-y-auto p-3">
        {jobs.length === 0 && !loading ? (
          <EmptyField
            title={
              filters.search || filters.status || filters.model_id
                ? "Nothing matches those filters"
                : "Nothing here yet"
            }
            detail={
              filters.search || filters.status || filters.model_id
                ? "Clear the filters to see the whole archive."
                : "Every image you generate lands here, with the prompt, seed and settings that made it. It survives restarts."
            }
          />
        ) : (
          <>
            <ul className="grid grid-cols-[repeat(auto-fill,minmax(196px,1fr))] border-l border-t border-[var(--rule)]">
              {jobs.map((job) => (
                <ArchiveCell key={job.id} job={job} />
              ))}
            </ul>
            {canLoadMore ? (
              <div className="flex justify-center pt-3">
                <button type="button" onClick={onLoadMore} className="btn h-8" disabled={loading}>
                  <span>{loading ? "Loading" : `Load more (${total - jobs.length} left)`}</span>
                </button>
              </div>
            ) : null}
          </>
        )}
      </div>
    </div>
  );
}

function ArchiveCell({ job }: { job: JobSummary }) {
  const images = job.images ?? [];
  const first = images[0];
  const missing = first && !first.available;

  return (
    <li className="border-b border-r border-[var(--rule)]">
      <Link
        to={`/j/${job.id}`}
        className="group flex h-full flex-col text-inherit no-underline outline-offset-[-2px]"
      >
        <div className="relative aspect-square overflow-hidden">
          {first?.url && first.available ? (
            <img
              src={first.url}
              alt={job.prompt}
              loading="lazy"
              decoding="async"
              className="h-full w-full object-cover transition-[filter] duration-200 group-hover:brightness-110"
            />
          ) : (
            <CellPlaceholder job={job} missing={Boolean(missing)} />
          )}
          {images.length > 1 ? (
            <span className="absolute right-0 top-0 bg-[var(--ground)] px-[5px] py-[2px] font-mono text-[10px] tabular text-[var(--ink-muted)]">
              ×{images.length}
            </span>
          ) : null}
        </div>

        {/* The label strip: readable at grid density, so a wall of these can be
            scanned without opening anything. */}
        <div className="flex flex-1 flex-col gap-[6px] border-t border-[var(--rule)] px-2 py-[6px]">
          <p className="line-clamp-2 text-[11px] leading-snug text-[var(--ink)]">{job.prompt}</p>
          <div className="mt-auto flex items-center justify-between gap-2">
            <span className="font-mono text-[9px] tabular text-[var(--ink-faint)]">
              {first ? first.seed : "—"}
            </span>
            <span className="font-mono text-[9px] tabular text-[var(--ink-faint)]">
              {job.width && job.height ? `${job.width}×${job.height}` : "—"}
            </span>
            <span className="font-mono text-[9px] tabular text-[var(--ink-faint)]">
              {shortDate(job.created)}
            </span>
          </div>
        </div>
      </Link>
    </li>
  );
}

function CellPlaceholder({ job, missing }: { job: JobSummary; missing: boolean }) {
  if (job.status === "rejected") {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 border-b border-[var(--caution)] px-3 text-center">
        <IconCaution className="text-[var(--caution-ink)]" />
        <span className="label text-[var(--caution-ink)]">refused</span>
        <p className="line-clamp-3 text-[10px] leading-snug text-[var(--ink-muted)]">
          {job.error}
        </p>
      </div>
    );
  }
  if (job.status === "failed") {
    return (
      <div className="flex h-full flex-col items-center justify-center gap-2 px-3 text-center">
        <span className="label text-[var(--alarm-ink)]">failed</span>
        <p className="line-clamp-3 text-[10px] leading-snug text-[var(--ink-muted)]">
          {job.error}
        </p>
      </div>
    );
  }
  if (missing) {
    return (
      <div className="dotted flex h-full flex-col items-center justify-center gap-1 px-3 text-center">
        <span className="label">file removed</span>
        <p className="text-[10px] leading-snug text-[var(--ink-muted)]">
          Retention reclaimed the disk. The settings are still here.
        </p>
      </div>
    );
  }
  return (
    <div className="dotted flex h-full items-center justify-center">
      <StateMark state={job.status} />
    </div>
  );
}

function FilterGroup({
  value,
  options,
  onChange,
}: {
  value?: string;
  options: { value: string; label: string }[];
  onChange: (value: string | undefined) => void;
}) {
  return (
    <div className="flex gap-[2px]">
      {options.map((option) => {
        const active = value === option.value;
        return (
          <button
            key={option.value}
            type="button"
            aria-pressed={active}
            onClick={() => onChange(active ? undefined : option.value)}
            className={`h-8 border px-2 text-[10px] font-semibold uppercase tracking-[0.1em] transition-colors ${
              active
                ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--ink-on-accent)]"
                : "border-[var(--rule)] text-[var(--ink-muted)] hover:border-[var(--rule-strong)] hover:text-[var(--ink)]"
            }`}
          >
            {option.label}
          </button>
        );
      })}
    </div>
  );
}
