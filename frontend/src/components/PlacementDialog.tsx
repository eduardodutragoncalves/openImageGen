import { useState } from "react"
import type { CatalogEntry, GpuInfo, Health } from "../lib/api"
import { gigabytes } from "../lib/format"
import { Dialog } from "./Dialog"
import { SegmentBar } from "./primitives"

/** What the load endpoint accepts, plus the card for a pinned placement. */
export interface Placement {
  mode: "auto" | "single" | "split"
  device?: number
}

interface Option {
  key: string
  title: string
  detail: string
  placement: Placement
  /** False when the hardware cannot hold it this way. Listed anyway, with the
   *  reason: an option that is missing teaches nothing. */
  possible: boolean
}

const MARGIN_GB = 1.5 // activations and the allocator's own overhead

/**
 * Where a checkpoint should go, asked before tens of gigabytes start moving.
 *
 * One thing here is easy to get wrong and worth stating plainly in the UI: a
 * switch is not a per-card operation. The manager drains the queue, unloads
 * whatever is resident from *every* card, and only then places the new model.
 * So the question is never "which card do I empty" — both are emptied either
 * way — it is "where does the new one go", and the dialog says so rather than
 * offering a choice the server does not actually have.
 */
export function PlacementDialog({
  entry,
  health,
  pending,
  error,
  onLoad,
  onClose,
}: {
  entry: CatalogEntry
  health?: Health
  pending: boolean
  error?: string
  onLoad: (placement: Placement) => void
  onClose: () => void
}) {
  const gpus = health?.gpus ?? []
  const options = buildOptions(entry, gpus)
  const [chosen, setChosen] = useState<string>(
    () => options.find((option) => option.possible)?.key ?? options[0]?.key ?? "auto",
  )
  const selected = options.find((option) => option.key === chosen)
  const resident = health?.model?.state === "ready" ? health?.model?.model_id : null

  return (
    <Dialog
      title={`Load ${entry.label}`}
      description={`${entry.total_vram_gb.toFixed(1)} GB at ${entry.precision} — transformer ${entry.transformer_vram_gb.toFixed(1)} GB, text encoder ${entry.text_encoder_vram_gb.toFixed(1)} GB`}
      onClose={onClose}
    >
      <div className="flex flex-col gap-4 px-4 py-4">
        {/* What is on the cards right now, so the cost of the swap is visible
            before it is paid rather than after. */}
        <div className="flex flex-col gap-2">
          <span className="label">the cards now</span>
          {gpus.length === 0 ? (
            <p className="text-xs text-[var(--ink-muted)]">
              No CUDA device is visible, so this would run on the CPU if it runs at all.
            </p>
          ) : (
            gpus.map((gpu) => (
              <div key={gpu.index} className="flex items-center gap-3">
                <span className="label w-10 shrink-0">gpu{gpu.index}</span>
                <SegmentBar
                  ratio={gpu.memory_total_mb ? gpu.memory_used_mb / gpu.memory_total_mb : 0}
                  segments={16}
                  height={8}
                  tone={gpu.role ? "accent" : "ink"}
                />
                <span className="w-28 shrink-0 text-right font-mono text-[10px] tabular text-[var(--ink-faint)]">
                  {gigabytes(gpu.memory_used_mb)}/{gigabytes(gpu.memory_total_mb)} GB
                </span>
                <span className="min-w-0 flex-1 truncate text-[10px] uppercase tracking-[0.1em] text-[var(--ink-muted)]">
                  {gpu.role ?? "free"}
                </span>
              </div>
            ))
          )}
        </div>

        {resident ? (
          <p className="border-l-2 border-[var(--caution)] pl-3 text-xs leading-relaxed text-[var(--ink-muted)]">
            Loading this unloads <span className="font-mono">{resident}</span> from{" "}
            <strong className="text-[var(--ink)]">every</strong> card first — a switch
            frees the whole placement, not one card of it. Anything queued finishes
            before the weights move.
          </p>
        ) : null}

        <fieldset className="flex flex-col gap-[2px]">
          <legend className="label mb-2">where it goes</legend>
          {options.map((option) => (
            <label
              key={option.key}
              className={`flex cursor-pointer items-start gap-3 border px-3 py-2 transition-colors ${
                chosen === option.key
                  ? "border-[var(--accent)] bg-[var(--accent-wash)]"
                  : "border-[var(--rule)] hover:border-[var(--rule-strong)]"
              } ${option.possible ? "" : "cursor-not-allowed opacity-55"}`}
            >
              <input
                type="radio"
                name="placement"
                className="mt-[3px] accent-[var(--accent)]"
                checked={chosen === option.key}
                disabled={!option.possible}
                onChange={() => setChosen(option.key)}
              />
              <span className="min-w-0">
                <span className="block text-xs font-semibold text-[var(--ink)]">
                  {option.title}
                </span>
                <span className="mt-[2px] block text-[11px] leading-relaxed text-[var(--ink-muted)]">
                  {option.detail}
                </span>
              </span>
            </label>
          ))}
        </fieldset>

        {error ? (
          <p role="alert" className="text-xs leading-relaxed text-[var(--alarm-ink)]">
            {error}
          </p>
        ) : null}

        <div className="flex flex-col gap-[2px]">
          <button
            type="button"
            className="btn btn-primary"
            disabled={pending || !selected?.possible}
            onClick={() => selected && onLoad(selected.placement)}
          >
            {pending ? "Moving the weights…" : "Load it"}
          </button>
          <button type="button" className="btn" onClick={onClose} disabled={pending}>
            Keep what is loaded
          </button>
        </div>
      </div>
    </Dialog>
  )
}

/**
 * The placements this machine can actually offer for this checkpoint.
 *
 * Sized against each card's *total* rather than its free memory, because the
 * switch empties the cards first: what is resident now is about to be gone, and
 * refusing an option because of memory that is on its way out would be wrong.
 */
function buildOptions(entry: CatalogEntry, gpus: GpuInfo[]): Option[] {
  const options: Option[] = [
    {
      key: "auto",
      title: "Let the planner choose",
      detail:
        entry.placement_reason ||
        "Fits the checkpoint the way this machine has room for it.",
      placement: { mode: "auto" },
      possible: entry.runnable,
    },
  ]

  const need = entry.total_vram_gb + MARGIN_GB
  for (const gpu of gpus) {
    const capacity = gpu.memory_total_mb / 1024
    const fits = capacity >= need
    options.push({
      key: `single-${gpu.index}`,
      title: `Whole on GPU ${gpu.index}`,
      detail: fits
        ? `${entry.total_vram_gb.toFixed(1)} GB into ${capacity.toFixed(1)} GB${
            gpus.length > 1 ? ", leaving the other card free for anything else" : ""
          }.`
        : `Will not fit: ${entry.total_vram_gb.toFixed(1)} GB of weights into ${capacity.toFixed(
            1,
          )} GB, with no room left for activations.`,
      placement: { mode: "single", device: gpu.index },
      possible: fits,
    })
  }

  if (gpus.length > 1) {
    const combined = gpus.reduce((sum, gpu) => sum + gpu.memory_total_mb / 1024, 0)
    const largest = Math.max(...gpus.map((gpu) => gpu.memory_total_mb / 1024))
    // Split puts the transformer on one card and the text encoder on another,
    // so it is the largest single component that has to fit, not the total.
    const fits =
      largest >= entry.transformer_vram_gb + MARGIN_GB &&
      largest >= entry.text_encoder_vram_gb + MARGIN_GB
    // Whether splitting is the only way this runs, or merely one way, changes
    // what the option means — and calling it "the only way" for a checkpoint
    // that fits on either card would be a plain untruth.
    const needsBoth = !gpus.some(
      (gpu) => gpu.memory_total_mb / 1024 >= entry.total_vram_gb + MARGIN_GB,
    )
    options.push({
      key: "split",
      title: "Split across both cards",
      detail: !fits
        ? `Will not fit even split: the transformer alone needs ${entry.transformer_vram_gb.toFixed(
            1,
          )} GB and the largest card has ${largest.toFixed(1)} GB of ${combined.toFixed(
            1,
          )} GB total.`
        : needsBoth
          ? `Transformer on one card, text encoder on the other — the only way a checkpoint this size runs here, since no single card holds all ${entry.total_vram_gb.toFixed(
              1,
            )} GB. Both cards are given to it.`
          : "Transformer on one card, text encoder on the other. Not needed for this one — it fits whole on either card — but it lowers the peak on each, at the cost of leaving neither free.",
      placement: { mode: "split" },
      possible: fits,
    })
  }

  return options
}
