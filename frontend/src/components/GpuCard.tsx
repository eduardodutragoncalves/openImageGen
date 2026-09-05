import { useRef, useState } from "react";
import type { GpuInfo, GpuRelease } from "../lib/api";
import { gigabytes } from "../lib/format";
import { SegmentBar } from "./primitives";
import { Dialog } from "./Dialog";
import { useReleaseGpu } from "../hooks/useApi";

/**
 * One card in the rail: how full it is, what is on it, and the way to empty it.
 *
 * The tape is the reading — fixed position, fixed width, legible without being
 * looked for. What is *on* the card is a second question, asked far less often,
 * so it waits under the pointer rather than taking rail space it would hold all
 * day.
 */
export function GpuCard({
  gpu,
  modelId,
  onCleared,
}: {
  gpu: GpuInfo;
  /** The loaded model, which is what a role on this card belongs to. */
  modelId?: string | null;
  onCleared?: (result: GpuRelease) => void;
}) {
  const [panel, setPanel] = useState<{ left: number } | null>(null);
  const [asking, setAsking] = useState(false);
  const button = useRef<HTMLButtonElement>(null);

  const ratio = gpu.memory_total_mb > 0 ? gpu.memory_used_mb / gpu.memory_total_mb : 0;
  // Pressure is legible before an OOM, not after it.
  const tone = ratio > 0.94 ? "alarm" : ratio > 0.82 ? "caution" : "accent";
  const carriesModel = Boolean(gpu.role);

  const show = () => {
    const rect = button.current?.getBoundingClientRect();
    // Fixed rather than absolute: the shell clips its overflow, and a panel
    // that hangs below the rail would be cut off at the rule.
    if (rect) setPanel({ left: rect.left });
  };

  return (
    <>
      <button
        ref={button}
        type="button"
        aria-haspopup="dialog"
        aria-label={`gpu${gpu.index}, ${gpu.name} — ${
          carriesModel ? `carrying ${gpu.role}` : "no model loaded on it"
        }. Clear its memory`}
        onPointerEnter={show}
        onPointerLeave={() => setPanel(null)}
        onFocus={show}
        onBlur={() => setPanel(null)}
        onClick={() => {
          setPanel(null);
          setAsking(true);
        }}
        className="flex flex-col justify-center gap-[3px] px-1 text-left transition-colors hover:bg-[var(--accent-wash)]"
      >
        <div className="flex items-baseline gap-2">
          <span className="label">gpu{gpu.index}</span>
          <span className="font-mono text-[10px] tabular text-[var(--ink-muted)]">
            {gigabytes(gpu.memory_used_mb)}/{gigabytes(gpu.memory_total_mb)} GB
          </span>
          {/* A dot, not a word: the rail has no room for a sentence, and this
              only has to say "there is something here to look at". */}
          {carriesModel ? (
            <span aria-hidden className="h-[5px] w-[5px] bg-[var(--accent)]" />
          ) : null}
        </div>
        <SegmentBar ratio={ratio} segments={10} tone={tone} height={6} />
      </button>

      {panel ? (
        <div
          role="tooltip"
          style={{ left: Math.max(8, Math.min(panel.left, window.innerWidth - 300)) }}
          className="fixed top-[var(--rail)] z-40 w-[292px] border border-[var(--rule-strong)] bg-[var(--ground)] p-3"
        >
          <p className="font-mono text-[11px] leading-tight text-[var(--ink)]">{gpu.name}</p>
          <p className="mt-[2px] font-mono text-[10px] tabular text-[var(--ink-faint)]">
            {gigabytes(gpu.memory_used_mb)} of {gigabytes(gpu.memory_total_mb)} GB in use
          </p>

          <div className="mt-3 border-t border-[var(--rule)] pt-2">
            <span className="label">loaded here</span>
            {carriesModel ? (
              <>
                <p className="mt-1 font-mono text-[11px] leading-tight text-[var(--ink)]">
                  {modelId ?? "the loaded model"}
                </p>
                <p className="mt-[2px] text-[10px] uppercase tracking-[0.1em] text-[var(--accent-ink)]">
                  {gpu.role}
                </p>
              </>
            ) : (
              <p className="mt-1 text-[11px] leading-relaxed text-[var(--ink-muted)]">
                No part of the model is on this card. Anything in use is cached
                memory, or belongs to another process.
              </p>
            )}
          </div>

          <p className="mt-3 border-t border-[var(--rule)] pt-2 text-[10px] text-[var(--ink-faint)]">
            Click to clear this card's memory.
          </p>
        </div>
      ) : null}

      {asking ? (
        <ClearDialog
          gpu={gpu}
          modelId={modelId}
          onClose={() => setAsking(false)}
          onCleared={onCleared}
        />
      ) : null}
    </>
  );
}

/**
 * The confirmation, and then the receipt.
 *
 * It stays open after the call to show what actually came back. "Cleared" with
 * the dialog already gone would be a claim; a measured number is an answer,
 * and on a card holding another process's memory the honest answer is often
 * "nothing came back".
 */
function ClearDialog({
  gpu,
  modelId,
  onClose,
  onCleared,
}: {
  gpu: GpuInfo;
  modelId?: string | null;
  onClose: () => void;
  onCleared?: (result: GpuRelease) => void;
}) {
  const release = useReleaseGpu();
  const carriesModel = Boolean(gpu.role);
  const result = release.data;

  return (
    <Dialog
      title={`Clear gpu${gpu.index}`}
      description={`${gpu.name} — ${gigabytes(gpu.memory_used_mb)} of ${gigabytes(
        gpu.memory_total_mb,
      )} GB in use`}
      onClose={onClose}
    >
      <div className="flex flex-col gap-3 px-4 py-4">
        {result ? (
          <p className="text-xs leading-relaxed text-[var(--ink)]">{result.detail}</p>
        ) : (
          <>
            {carriesModel ? (
              <>
                <p className="text-xs leading-relaxed text-[var(--ink)]">
                  This card is carrying{" "}
                  <span className="font-mono text-[var(--accent-ink)]">{gpu.role}</span> of{" "}
                  <span className="font-mono">{modelId ?? "the loaded model"}</span>.
                </p>
                {/* The one thing an operator must not learn afterwards. */}
                <p className="border-l-2 border-[var(--caution)] pl-3 text-xs leading-relaxed text-[var(--ink-muted)]">
                  A model is placed <em>across</em> cards, so it cannot be dropped from
                  one and kept on the others — a pipeline missing its text encoder is
                  not a smaller model. Clearing this card unloads the model from{" "}
                  <strong className="text-[var(--ink)]">every</strong> card. Generations
                  will fail until you load one again.
                </p>
                <p className="text-xs leading-relaxed text-[var(--ink-muted)]">
                  Anything queued is allowed to finish first; nothing is unloaded out
                  from under a running job.
                </p>
              </>
            ) : (
              <p className="text-xs leading-relaxed text-[var(--ink-muted)]">
                No part of the model is on this card, so nothing gets unloaded. This
                hands back the memory this process has cached and finished with. Memory
                belonging to another process cannot be released from here.
              </p>
            )}
          </>
        )}

        {release.isError ? (
          <p role="alert" className="text-xs leading-relaxed text-[var(--alarm-ink)]">
            {(release.error as Error).message}
          </p>
        ) : null}

        {/* "Done" rather than "Close" on the receipt: the dialog's own dismiss
            is already named Close, and two buttons with one name is a guess —
            for a screen reader as much as for a test. */}
        <div className="mt-1 flex flex-col gap-[2px]">
          {result ? (
            <button type="button" className="btn" onClick={onClose}>
              Done
            </button>
          ) : (
            <>
              <button
                type="button"
                className="btn btn-primary"
                disabled={release.isPending}
                onClick={() =>
                  release.mutate(gpu.index, {
                    onSuccess: (data) => onCleared?.(data),
                  })
                }
              >
                {release.isPending
                  ? carriesModel
                    ? "Draining the queue and unloading…"
                    : "Clearing…"
                  : carriesModel
                    ? "Unload the model and clear"
                    : "Clear cached memory"}
              </button>
              <button
                type="button"
                className="btn"
                onClick={onClose}
                disabled={release.isPending}
              >
                Keep it
              </button>
            </>
          )}
        </div>
      </div>
    </Dialog>
  );
}
