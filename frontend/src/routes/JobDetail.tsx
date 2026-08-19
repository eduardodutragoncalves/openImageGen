import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useDeleteJob, useJob } from "../hooks/useApi";
import { duration, shortDate } from "../lib/format";
import { Readout, StateMark } from "../components/primitives";
import { IconArrowRight, IconCaution, IconDownload, IconTrash } from "../components/Icons";

/**
 * One job, addressable. This route is the structural answer to "I lost the
 * id": every result is a URL that survives the tab, the restart and the TTL.
 */
export function JobDetail() {
  const { jobId } = useParams();
  const navigate = useNavigate();
  const job = useJob(jobId);
  const remove = useDeleteJob();
  const [selected, setSelected] = useState(0);

  if (job.isLoading) {
    return <Centered>Reading the archive…</Centered>;
  }
  if (job.isError || !job.data) {
    return (
      <Centered>
        <p className="mb-3">No job with that id, or it belongs to another key.</p>
        <Link to="/" className="btn h-8 no-underline">
          <span>Back to the studio</span>
          <IconArrowRight size={14} />
        </Link>
      </Centered>
    );
  }

  const data = job.data;
  const images = data.result?.images ?? [];
  const image = images[selected];
  const params = data.result;

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto grid max-w-[1500px] grid-cols-1 gap-6 px-4 py-6 lg:grid-cols-[minmax(0,1fr)_320px]">
        <div className="min-w-0">
          <div className="mb-3 flex items-center gap-3">
            <StateMark state={data.status} />
            <span className="font-mono text-[11px] tabular text-[var(--ink-faint)]">
              {data.id}
            </span>
          </div>

          {image?.url ? (
            <figure className="flex max-h-[calc(100dvh-var(--rail)-96px)] justify-center border border-[var(--rule)]">
              <img
                src={image.url}
                alt={params?.prompt ?? "Generated image"}
                className="max-h-full w-auto max-w-full object-contain"
              />
            </figure>
          ) : data.status === "rejected" ? (
            <div className="border border-[var(--caution)] p-6">
              <div className="mb-2 flex items-center gap-2">
                <IconCaution className="text-[var(--caution-ink)]" />
                <h2 className="label text-[var(--caution-ink)]">refused by a content filter</h2>
              </div>
              <p className="max-w-[64ch] text-sm leading-relaxed text-[var(--ink)]">
                {data.error}
              </p>
              <p className="mt-3 max-w-[64ch] text-xs leading-relaxed text-[var(--ink-muted)]">
                This is the filter working, not a failure. Rewording the prompt away from
                named people, brands and protected characters is usually enough.
              </p>
            </div>
          ) : data.status === "failed" ? (
            <div className="border border-[var(--alarm)] p-6">
              <h2 className="label mb-2 text-[var(--alarm-ink)]">failed</h2>
              <p className="max-w-[64ch] font-mono text-xs leading-relaxed text-[var(--ink)]">
                {data.error}
              </p>
            </div>
          ) : (
            <div className="dotted flex min-h-[320px] items-center justify-center border border-[var(--rule)]">
              <StateMark state={data.status} />
            </div>
          )}

          {images.length > 1 ? (
            <ul className="mt-[1px] flex gap-[1px] bg-[var(--rule)]">
              {images.map((candidate, index) => (
                <li key={index}>
                  <button
                    type="button"
                    onClick={() => setSelected(index)}
                    aria-pressed={index === selected}
                    className={`block h-16 w-16 border-2 ${
                      index === selected ? "border-[var(--accent)]" : "border-transparent"
                    }`}
                  >
                    {candidate.url ? (
                      <img
                        src={candidate.url}
                        alt={`Variation ${index + 1}, seed ${candidate.seed}`}
                        className="h-full w-full object-cover"
                      />
                    ) : null}
                  </button>
                </li>
              ))}
            </ul>
          ) : null}
        </div>

        <aside className="flex flex-col gap-4">
          <div>
            <h1 className="label mb-2">prompt</h1>
            <p className="text-sm leading-relaxed text-[var(--ink)]">{params?.prompt}</p>
            {params?.revised_prompt ? (
              <>
                <h2 className="label mb-1 mt-3">upsampled to</h2>
                <p className="text-xs leading-relaxed text-[var(--ink-muted)]">
                  {params.revised_prompt}
                </p>
              </>
            ) : null}
          </div>

          <dl className="grid grid-cols-2 gap-4 border-t border-[var(--rule)] pt-3">
            <Readout label="seed" value={image?.seed ?? "—"} size="sm" />
            <Readout
              label="size"
              value={image ? `${image.width}×${image.height}` : "—"}
              size="sm"
            />
            <Readout label="took" value={duration(params?.timings?.total_s)} size="sm" />
            <Readout label="made" value={shortDate(data.created)} size="sm" />
          </dl>

          <div className="flex flex-col gap-[2px] border-t border-[var(--rule)] pt-3">
            {image?.url ? (
              <a href={image.url} download className="btn no-underline">
                <span>Download</span>
                <IconDownload size={14} />
              </a>
            ) : null}
            <button
              type="button"
              className="btn"
              onClick={() =>
                navigate("/", {
                  state: {
                    preset: {
                      prompt: params?.prompt ?? "",
                      seed: image?.seed,
                      width: image?.width,
                      height: image?.height,
                    },
                  },
                })
              }
            >
              <span>Reuse these settings</span>
              <IconArrowRight size={14} />
            </button>
            <button
              type="button"
              className="btn hover:!border-[var(--alarm)] hover:!text-[var(--alarm-ink)]"
              disabled={remove.isPending}
              onClick={() => {
                if (!jobId) return;
                remove.mutate(jobId, { onSuccess: () => navigate("/") });
              }}
            >
              <span>Delete job and files</span>
              <IconTrash size={14} />
            </button>
          </div>
        </aside>
      </div>
    </div>
  );
}

function Centered({ children }: { children: React.ReactNode }) {
  return (
    <div className="flex min-h-0 flex-1 flex-col items-center justify-center p-8 text-center text-sm text-[var(--ink-muted)]">
      {children}
    </div>
  );
}
