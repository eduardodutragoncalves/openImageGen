import { useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import { useDeleteJob, useJob } from "../hooks/useApi";
import type { ComposePreset } from "../components/Compose";
import { duration, shortDate, usd } from "../lib/format";
import { Readout, StateMark } from "../components/primitives";
import { IconArrowRight, IconCaution, IconDownload, IconImage, IconTrash } from "../components/Icons";

/**
 * One job, addressable. This route is the structural answer to "I lost the
 * id": every result is a URL that survives the tab, the restart and the TTL.
 *
 * Everything here reads from the *request* rather than from the result, so a
 * job that was refused or failed shows the prompt and settings that produced
 * it. That is precisely when the operator needs them back.
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
  const request = data.request;
  const images = data.result?.images ?? [];
  const image = images[selected];
  const references = request?.references ?? [];

  const preset: ComposePreset = {
    prompt: request?.prompt ?? "",
    seed: image?.seed ?? request?.seed ?? undefined,
    width: image?.width ?? request?.width ?? undefined,
    height: image?.height ?? request?.height ?? undefined,
    numSteps: request?.num_steps ?? undefined,
    guidance: request?.guidance ?? undefined,
    numImages: request?.num_images,
    upsampleMode: request?.upsample_mode ?? undefined,
    referenceUrls: references.filter((r) => r.available && r.url).map((r) => r.url as string),
    stamp: `${data.id}:${Date.now()}`,
  };

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      <div className="mx-auto grid max-w-[1500px] grid-cols-1 gap-6 px-4 py-6 lg:grid-cols-[minmax(0,1fr)_340px]">
        <div className="min-w-0">
          <div className="mb-3 flex items-center gap-3">
            <StateMark state={data.status} />
            <span className="font-mono text-[11px] tabular text-[var(--ink-faint)]">{data.id}</span>
            {request?.model_label ? (
              <span className="text-[10px] uppercase tracking-[0.1em] text-[var(--ink-faint)]">
                {request.model_label}
              </span>
            ) : null}
          </div>

          {image?.url ? (
            <figure className="flex max-h-[calc(100dvh-var(--rail)-96px)] justify-center border border-[var(--rule)]">
              <img
                src={image.url}
                alt={request?.prompt ?? "Generated image"}
                className="max-h-full w-auto max-w-full object-contain"
              />
            </figure>
          ) : data.status === "rejected" ? (
            <div className="border border-[var(--caution)] p-6">
              <div className="mb-2 flex items-center gap-2">
                <IconCaution className="text-[var(--caution-ink)]" />
                <h2 className="label text-[var(--caution-ink)]">refused by a content filter</h2>
              </div>
              <p className="max-w-[64ch] text-sm leading-relaxed text-[var(--ink)]">{data.error}</p>
              <p className="mt-3 max-w-[64ch] text-xs leading-relaxed text-[var(--ink-muted)]">
                This is the filter working, not a failure. Reuse the settings below, reword the
                prompt away from named people, brands and protected characters, and run it again.
              </p>
            </div>
          ) : data.status === "failed" ? (
            <div className="border border-[var(--alarm)] p-6">
              <h2 className="label mb-2 text-[var(--alarm-ink)]">failed</h2>
              <p className="max-w-[64ch] font-mono text-xs leading-relaxed text-[var(--ink)]">
                {data.error}
              </p>
              <p className="mt-3 max-w-[64ch] text-xs leading-relaxed text-[var(--ink-muted)]">
                Nothing was lost: the prompt and every setting are below, ready to send again.
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

          {references.length > 0 ? (
            <section className="mt-4">
              <h2 className="label mb-2">references it was given</h2>
              <ul className="flex flex-wrap gap-[1px] bg-[var(--rule)]">
                {references.map((reference, index) => (
                  <li key={index} className="h-20 w-20 bg-[var(--ground)]">
                    {reference.available && reference.url ? (
                      <img
                        src={reference.url}
                        alt={`Reference ${index + 1}`}
                        className="h-full w-full object-cover"
                      />
                    ) : (
                      <div className="dotted flex h-full items-center justify-center">
                        <IconImage size={14} className="text-[var(--ink-faint)]" />
                      </div>
                    )}
                  </li>
                ))}
              </ul>
            </section>
          ) : null}
        </div>

        <aside className="flex flex-col gap-4">
          <div>
            <h1 className="label mb-2">prompt</h1>
            <p className="text-sm leading-relaxed text-[var(--ink)]">
              {request?.prompt || <span className="text-[var(--ink-faint)]">—</span>}
            </p>
            {data.result?.revised_prompt ? (
              <>
                <h2 className="label mb-1 mt-3">upsampled to</h2>
                <p className="text-xs leading-relaxed text-[var(--ink-muted)]">
                  {data.result.revised_prompt}
                </p>
              </>
            ) : null}
          </div>

          {/* The settings, shown for every outcome. A refused job that hid its
              own parameters would be the one job you cannot learn from. */}
          <dl className="grid grid-cols-2 gap-4 border-t border-[var(--rule)] pt-3">
            <Readout label="seed" value={image?.seed ?? request?.seed ?? "random"} size="sm" />
            <Readout
              label="size"
              value={
                image
                  ? `${image.width}×${image.height}`
                  : request?.width && request?.height
                    ? `${request.width}×${request.height}`
                    : "—"
              }
              size="sm"
            />
            <Readout label="steps" value={request?.num_steps ?? "—"} size="sm" />
            <Readout
              label="guidance"
              value={request?.guidance != null ? request.guidance.toFixed(1) : "—"}
              size="sm"
            />
            <Readout label="images" value={request?.num_images ?? 1} size="sm" />
            <Readout
              label="upsampling"
              value={request?.upsample_mode && request.upsample_mode !== "none" ? request.upsample_mode : "none"}
              size="sm"
            />
            <Readout label="took" value={duration(data.result?.timings?.total_s)} size="sm" />
            <Readout label="made" value={shortDate(data.created)} size="sm" />
            {/* What made it and what it cost, in the panel rather than only in
                the file: the header names the model in passing, but this is
                the grid an operator actually reads a result off. A local run
                bills nothing and says "—" rather than "$0.00", which would
                claim it was free instead of unpriced. */}
            <Readout
              label="model"
              value={request?.model_label ?? request?.model_id ?? "—"}
              size="sm"
              tone={request?.remote ? "accent" : "ink"}
            />
            <Readout label="cost" value={usd(image?.cost) ?? "—"} size="sm" />
          </dl>

          <div className="flex flex-col gap-[2px] border-t border-[var(--rule)] pt-3">
            <button
              type="button"
              className="btn btn-primary"
              onClick={() => navigate("/", { state: { preset } })}
            >
              <span>Reuse these settings</span>
              <IconArrowRight size={14} />
            </button>
            {image?.url ? (
              <a href={image.url} download className="btn no-underline">
                <span>Download</span>
                <IconDownload size={14} />
              </a>
            ) : null}
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
