import { useState } from "react";
import { useLocation } from "react-router-dom";
import type { Health } from "../lib/api";
import {
  useArchive,
  useCatalog,
  useCompletionNotice,
  useLiveJobs,
  useModel,
  useTitleCount,
} from "../hooks/useApi";
import type { ArchiveFilters } from "../hooks/useApi";
import { Compose } from "../components/Compose";
import type { ComposePreset } from "../components/Compose";
import { ActiveJob } from "../components/ActiveJob";
import { Archive } from "../components/Archive";
import { Region } from "../components/primitives";

const PAGE = 60;

/**
 * The studio: compose at left, what is running and what has been made at
 * right. Both are live at once, because queueing more work while a job runs is
 * the normal case, not an edge one.
 */
export function Studio({ health }: { health?: Health }) {
  const location = useLocation();
  const preset = (location.state as { preset?: ComposePreset } | null)?.preset;
  const model = useModel();
  const live = useLiveJobs();
  const catalog = useCatalog();
  const [filters, setFilters] = useState<ArchiveFilters>({});
  const [limit, setLimit] = useState(PAGE);
  const archive = useArchive(filters, limit);

  const liveJobs = live.live;
  const running = liveJobs.filter((job) => job.status === "running");
  useCompletionNotice(live.data?.jobs ?? []);
  useTitleCount(running.length, running[0]?.progress ?? null);

  const jobs = archive.data?.jobs ?? [];
  const total = archive.data?.total ?? 0;

  return (
    <div className="grid min-h-0 flex-1 grid-cols-1 xl:grid-cols-[380px_minmax(0,1fr)]">
      <Region
        label="compose"
        className="min-h-0 border-b border-[var(--rule)] xl:border-b-0 xl:border-r"
      >
        <Compose model={model.data} health={health} preset={preset} />
      </Region>

      <div className="grid min-h-0 grid-rows-[auto_minmax(0,1fr)]">
        <Region
          label="running"
          className="border-b border-[var(--rule)]"
          aside={
            <span className="font-mono text-[10px] tabular text-[var(--ink-faint)]">
              {liveJobs.length} in flight
            </span>
          }
        >
          <ActiveJob jobs={liveJobs} />
        </Region>

        <Region label="archive" className="min-h-0">
          <Archive
            jobs={jobs}
            total={total}
            filters={filters}
            onFilters={(next) => {
              setFilters(next);
              setLimit(PAGE);
            }}
            models={catalog.data ?? []}
            loading={archive.isFetching}
            canLoadMore={jobs.length < total}
            onLoadMore={() => setLimit((current) => current + PAGE)}
          />
        </Region>
      </div>
    </div>
  );
}
