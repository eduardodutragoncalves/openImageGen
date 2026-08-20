import { useMemo, useState } from "react";
import { WebModels } from "./WebModels";
import type { CatalogEntry, Health } from "../lib/api";
import { ApiError } from "../lib/api";
import { useCatalog, useLoadModel } from "../hooks/useApi";
import { Diagonal, Readout, SegmentBar } from "../components/primitives";
import { IconCaution, IconCheck, IconChip, IconKey, IconLayers } from "../components/Icons";

const FAMILY_LABEL: Record<string, string> = { flux2: "FLUX.2", flux1: "FLUX.1" };

/**
 * The configuration surface: which model this server runs.
 *
 * Every model gets the same specification block, and the ones this hardware
 * cannot hold are shown demoted with the reason rather than filtered away —
 * "why can't I pick that?" is a question the page should answer, not dodge.
 */
type Tab = "local" | "web";

export function Models({ health }: { health?: Health }) {
  const catalog = useCatalog();
  const load = useLoadModel();
  const [custom, setCustom] = useState("");
  const [tab, setTab] = useState<Tab>("local");

  const busy = health?.model.state === "switching" || health?.model.state === "loading";
  const entries = catalog.data ?? [];

  const families = useMemo(() => {
    const grouped = new Map<string, CatalogEntry[]>();
    for (const entry of entries) {
      const list = grouped.get(entry.family) ?? [];
      list.push(entry);
      grouped.set(entry.family, list);
    }
    // Runnable first inside each family; unavailable ones sink but stay.
    for (const list of grouped.values()) {
      list.sort((a, b) => Number(b.runnable) - Number(a.runnable));
    }
    return [...grouped.entries()];
  }, [entries]);

  return (
    <div className="min-h-0 flex-1 overflow-y-auto">
      {busy ? <SwitchBanner health={health!} /> : null}

      <nav
        className="flex items-stretch border-b border-[var(--rule)]"
        aria-label="Model sources"
      >
        {(
          [
            ["local", "On this machine"],
            ["web", "Web models"],
          ] as const
        ).map(([value, label]) => (
          <button
            key={value}
            type="button"
            onClick={() => setTab(value)}
            aria-pressed={tab === value}
            className={`relative flex h-11 items-center border-r border-[var(--rule)] px-5 text-[11px] font-semibold uppercase tracking-[0.14em] transition-colors ${
              tab === value
                ? "text-[var(--ink-on-accent)]"
                : "text-[var(--ink-muted)] hover:text-[var(--ink)]"
            }`}
          >
            {tab === value ? (
              <span
                aria-hidden
                className="absolute inset-0 bg-[var(--accent)]"
                style={{ clipPath: "polygon(12px 0, 100% 0, calc(100% - 12px) 100%, 0 100%)" }}
              />
            ) : null}
            <span className="relative">{label}</span>
          </button>
        ))}
      </nav>

      {tab === "web" ? <WebModels /> : (
      <div className="mx-auto max-w-[1400px] px-4 py-6">
        <div className="mb-6 flex items-end justify-between gap-8 border-b border-[var(--rule)] pb-4">
          <div>
            <h1
              className="numeral text-5xl"
              style={{ fontStretch: "125%" }}
            >
              MODELS
            </h1>
            <p className="mt-2 max-w-[62ch] text-xs leading-relaxed text-[var(--ink-muted)]">
              Loading a model unloads the current one, replans where each component
              sits on your GPUs, and takes minutes. Queued work waits; it is not lost.
            </p>
          </div>
          <div className="flex gap-8">
            <Readout label="known" value={entries.length} size="md" />
            <Readout
              label="runnable here"
              value={entries.filter((entry) => entry.runnable).length}
              size="md"
              tone="accent"
            />
          </div>
        </div>

        {load.isError ? (
          <p
            role="alert"
            className="mb-4 border border-[var(--alarm)] px-3 py-2 text-xs text-[var(--alarm-ink)]"
          >
            {load.error instanceof ApiError ? load.error.message : "That model would not load."}
          </p>
        ) : null}

        {catalog.isLoading ? (
          <p className="text-xs text-[var(--ink-muted)]">Reading the catalog…</p>
        ) : null}

        {families.map(([family, list]) => (
          <section key={family} className="mb-8">
            <header className="mb-3 flex items-baseline gap-3">
              <h2 className="numeral text-2xl">{FAMILY_LABEL[family] ?? family}</h2>
              <span className="label">{list.length} checkpoints</span>
            </header>
            <ul className="grid grid-cols-[repeat(auto-fill,minmax(340px,1fr))] border-l border-t border-[var(--rule)]">
              {list.map((entry) => (
                <ModelRow
                  key={entry.id}
                  entry={entry}
                  busy={busy || load.isPending}
                  onLoad={() => load.mutate({ model: entry.id })}
                />
              ))}
            </ul>
          </section>
        ))}

        <section className="border-t border-[var(--rule)] pt-4">
          <h2 className="label mb-2">Any other checkpoint</h2>
          <p className="mb-3 max-w-[62ch] text-xs leading-relaxed text-[var(--ink-muted)]">
            A hugging face repo id outside the catalog is accepted. Its architecture is
            guessed from the name and its memory footprint is assumed, so placement may
            be wrong — set OIG_TRANSFORMER_VRAM_GB and OIG_TEXT_ENCODER_VRAM_GB if it
            misplaces.
          </p>
          <form
            className="flex max-w-[560px] gap-[2px]"
            onSubmit={(event) => {
              event.preventDefault();
              if (custom.trim()) load.mutate({ model: custom.trim() });
            }}
          >
            <input
              value={custom}
              onChange={(event) => setCustom(event.target.value)}
              placeholder="black-forest-labs/FLUX.1-schnell"
              aria-label="Hugging Face repo id"
              className="field h-10 font-mono text-xs"
            />
            <button type="submit" className="btn shrink-0" disabled={busy || !custom.trim()}>
              <span>Load</span>
              <Diagonal size={14} />
            </button>
          </form>
        </section>
      </div>
      )}
    </div>
  );
}

function ModelRow({
  entry,
  busy,
  onLoad,
}: {
  entry: CatalogEntry;
  busy: boolean;
  onLoad: () => void;
}) {
  const unavailable = !entry.runnable;
  return (
    <li
      className={`flex flex-col gap-3 border-b border-r border-[var(--rule)] p-3 ${
        unavailable ? "opacity-60" : ""
      }`}
    >
      <div className="flex items-start justify-between gap-3">
        <div className="min-w-0">
          <h3 className="text-sm font-semibold tracking-tight text-[var(--ink)]">
            {entry.label}
          </h3>
          <p className="truncate font-mono text-[10px] text-[var(--ink-faint)]">
            {entry.repo_id}
          </p>
        </div>
        {entry.loaded ? (
          <span className="flex h-5 shrink-0 items-center gap-1 bg-[var(--accent)] px-[6px] text-[10px] font-semibold uppercase tracking-[0.12em] text-[var(--ink-on-accent)]">
            <IconCheck size={11} />
            loaded
          </span>
        ) : null}
      </div>

      <p className="text-[11px] leading-relaxed text-[var(--ink-muted)]">{entry.summary}</p>

      {/* The specification block. Identical on every entry, so two models can
          be compared by reading down the same line. */}
      <dl className="grid grid-cols-2 gap-x-4 gap-y-[6px] border-t border-[var(--rule)] pt-2 text-[10px]">
        <Spec label="licence">
          <a
            href={entry.licence_url || undefined}
            target="_blank"
            rel="noreferrer"
            className={entry.licence_url ? "" : "pointer-events-none no-underline"}
          >
            {entry.licence}
          </a>
          {entry.commercial_use ? (
            <span className="ml-1 text-[var(--ink-faint)]">· commercial ok</span>
          ) : (
            <span className="ml-1 text-[var(--caution-ink)]">· non-commercial</span>
          )}
        </Spec>
        <Spec label="steps">
          <span className="font-mono tabular">{entry.default_steps}</span>
          <span className="text-[var(--ink-faint)]">
            {" "}
            ({entry.step_range[0]}–{entry.step_range[1]})
          </span>
        </Spec>
        <Spec label={entry.precision === "nf4" ? "weights (4-bit)" : "weights"}>
          <span className="font-mono tabular">{entry.total_vram_gb} GB</span>
          <span className="text-[var(--ink-faint)]">
            {" "}
            = {entry.transformer_vram_gb} + {entry.text_encoder_vram_gb}
          </span>
        </Spec>
        <Spec label="guidance">
          {entry.guidance_range[0] === entry.guidance_range[1] ? (
            <span className="text-[var(--ink-faint)]">not used</span>
          ) : (
            <span className="font-mono tabular">{entry.default_guidance}</span>
          )}
        </Spec>
      </dl>

      <div className="flex flex-wrap gap-1">
        {entry.capabilities.map((capability) => (
          <span
            key={capability}
            className="flex h-5 items-center gap-1 border border-[var(--rule)] px-[5px] text-[10px] uppercase tracking-[0.1em] text-[var(--ink-muted)]"
          >
            {capability === "multi-reference" ? <IconLayers size={10} /> : null}
            {capability}
          </span>
        ))}
        {entry.gated ? (
          <span className="flex h-5 items-center gap-1 border border-[var(--caution)] px-[5px] text-[10px] uppercase tracking-[0.1em] text-[var(--caution-ink)]">
            <IconKey size={10} />
            gated
          </span>
        ) : null}
      </div>

      <div className="mt-auto flex items-center gap-2 border-t border-[var(--rule)] pt-2">
        <IconChip
          size={13}
          className={unavailable ? "text-[var(--caution-ink)]" : "text-[var(--ink-faint)]"}
        />
        <p
          className={`min-w-0 flex-1 text-[10px] leading-snug ${
            unavailable ? "text-[var(--caution-ink)]" : "text-[var(--ink-faint)]"
          }`}
        >
          {unavailable ? (
            <>
              <IconCaution size={10} className="mr-1 inline align-[-1px]" />
              {entry.placement_reason}
            </>
          ) : (
            <>
              <span className="uppercase tracking-[0.1em]">{entry.placement}</span> ·{" "}
              {entry.placement_reason}
            </>
          )}
        </p>
      </div>

      {entry.precision === "nf4" && entry.runnable ? (
        <p className="text-[10px] leading-snug text-[var(--ink-faint)]">
          Quantized to NF4 on load: the bf16 weights are{" "}
          <span className="font-mono tabular">
            {entry.family === "flux1" ? "33.6" : ""} GB
          </span>{" "}
          and would not fit a card here.
        </p>
      ) : null}
      {entry.notes ? (
        <p className="text-[10px] leading-snug text-[var(--ink-faint)]">{entry.notes}</p>
      ) : null}

      <button
        type="button"
        className={`btn w-full ${entry.loaded ? "" : "btn-primary"}`}
        disabled={busy || entry.loaded || unavailable}
        onClick={onLoad}
      >
        <span>
          {entry.loaded ? "Loaded" : unavailable ? "Will not fit" : "Load this model"}
        </span>
        {entry.loaded || unavailable ? null : <Diagonal size={14} />}
      </button>
    </li>
  );
}

function Spec({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex flex-col gap-[2px]">
      <dt className="label">{label}</dt>
      <dd className="text-[var(--ink)]">{children}</dd>
    </div>
  );
}

/**
 * Switching, made legible. The phases are real — draining, unloading,
 * transformer, encoders — so a two-minute silence never reads as a hang.
 */
function SwitchBanner({ health }: { health: Health }) {
  const model = health.model;
  return (
    <div className="sticky top-0 z-10 border-b border-[var(--accent)] bg-[var(--ground)]/95 px-4 py-3 backdrop-blur-[2px]">
      <div className="mx-auto flex max-w-[1400px] items-center gap-6">
        <Readout
          label={model.state === "switching" ? "switching" : "loading"}
          value={Math.round(model.progress * 100)}
          unit="%"
          size="lg"
          tone="accent"
        />
        <div className="min-w-0 flex-1">
          <p className="mb-2 text-xs text-[var(--ink)]">
            {model.phase}
            {model.target_id ? (
              <span className="text-[var(--ink-muted)]"> → {model.target_id}</span>
            ) : null}
          </p>
          <SegmentBar ratio={model.progress} segments={40} height={10} />
        </div>
      </div>
      {model.detail ? (
        <p className="mx-auto mt-2 max-w-[1400px] text-[11px] text-[var(--caution-ink)]">
          {model.detail}
        </p>
      ) : null}
    </div>
  );
}
