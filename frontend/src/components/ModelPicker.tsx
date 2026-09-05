import { useState } from "react";
import type { CatalogEntry, GpuInfo, HubModel, Placement, ProviderInfo } from "../lib/api";
import {
  useCatalog,
  useGpus,
  useHealth,
  useHubSearch,
  useLoadModel,
  usePin,
  usePinned,
  useProviderCheck,
  useProviderModels,
  useProviders,
  useUnpin,
} from "../hooks/useApi";
import { Dialog } from "./Dialog";
import { RemoteModelList } from "./RemoteModelList";
import { PlacementDialog } from "./PlacementDialog";
import { IconCheck, IconClose, IconImage, IconSearch } from "./Icons";

/**
 * One place to answer "what should make this picture".
 *
 * The three answers used to live in three places: the checkpoint on the GPUs
 * was the Models page, a repo id was a text field at the bottom of it, and a
 * provider's model was a different tab entirely. They are the same decision,
 * so they are now the same dialog, and the sources sit side by side with what
 * separates them stated rather than implied — what runs on your cards, what
 * would have to be downloaded first, and what is billed by someone else.
 */

type Source = { id: string; label: string; kind: "local" | "hub" | "provider" };

export function ModelPicker({
  onClose,
  onPicked,
}: {
  onClose: () => void;
  /** A provider model was pinned and is ready to generate with. */
  onPicked?: (key: string) => void;
}) {
  const providers = useProviders();
  const [source, setSource] = useState<Source>({ id: "local", label: "On this machine", kind: "local" });
  const [query, setQuery] = useState("");

  const sources: Source[] = [
    { id: "local", label: "On this machine", kind: "local" },
    { id: "hub", label: "Hugging Face", kind: "hub" },
    ...(providers.data ?? []).map((entry) => ({
      id: entry.id,
      label: entry.label,
      kind: "provider" as const,
    })),
  ];

  return (
    <Dialog
      title="Choose a model"
      description="What makes the picture: a checkpoint on your GPUs, one to download, or a model reached over an API."
      onClose={onClose}
      wide
    >
      <div className="flex min-h-[420px] flex-col sm:flex-row">
        <nav
          aria-label="Model sources"
          className="flex shrink-0 gap-[2px] overflow-x-auto border-b border-[var(--rule)] p-2 sm:w-[180px] sm:flex-col sm:overflow-visible sm:border-b-0 sm:border-r"
        >
          {sources.map((entry) => (
            <SourceButton
              key={entry.id}
              entry={entry}
              active={entry.id === source.id}
              provider={(providers.data ?? []).find((p) => p.id === entry.id)}
              onSelect={() => {
                setSource(entry);
                setQuery("");
              }}
            />
          ))}
        </nav>

        <div className="flex min-w-0 flex-1 flex-col">
          <div className="relative border-b border-[var(--rule)] p-2">
            <IconSearch
              size={14}
              className="pointer-events-none absolute left-4 top-1/2 -translate-y-1/2 text-[var(--ink-faint)]"
            />
            <input
              value={query}
              onChange={(event) => setQuery(event.target.value)}
              placeholder={
                source.kind === "hub"
                  ? "Search Hugging Face — a name or an author"
                  : `Search ${source.label}`
              }
              aria-label="Search models"
              className="field h-9 w-full pl-8 text-xs"
              autoFocus
            />
          </div>

          <div className="min-h-0 flex-1 overflow-y-auto p-2">
            {source.kind === "local" ? (
              <LocalModels query={query} onDone={onClose} />
            ) : source.kind === "hub" ? (
              <HubModels query={query} onDone={onClose} />
            ) : (
              <ProviderModels providerId={source.id} query={query} onPicked={onPicked} />
            )}
          </div>
        </div>
      </div>
    </Dialog>
  );
}

/** A source, and for a provider whether its credential actually works. */
function SourceButton({
  entry,
  active,
  provider,
  onSelect,
}: {
  entry: Source;
  active: boolean;
  provider?: ProviderInfo;
  onSelect: () => void;
}) {
  // Asked only for a provider that claims to have one: there is nothing to
  // check otherwise, and every check costs a request.
  const check = useProviderCheck(entry.id, entry.kind === "provider" && Boolean(provider?.configured));

  return (
    <button
      type="button"
      onClick={onSelect}
      aria-pressed={active}
      className={`flex h-9 shrink-0 items-center justify-between gap-2 border px-3 text-left text-[11px] font-semibold uppercase tracking-[0.1em] transition-colors ${
        active
          ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--ink-on-accent)]"
          : "border-transparent text-[var(--ink-muted)] hover:text-[var(--ink)]"
      }`}
    >
      <span className="truncate">{entry.label}</span>
      {entry.kind === "provider" ? <KeyDot provider={provider} state={check} /> : null}
    </button>
  );
}

function KeyDot({
  provider,
  state,
}: {
  provider?: ProviderInfo;
  state: ReturnType<typeof useProviderCheck>;
}) {
  if (!provider?.configured) {
    return (
      <span title="No key set" aria-label="no key set" className="text-[10px] tracking-normal opacity-70">
        no key
      </span>
    );
  }
  if (state.isLoading) {
    return <span className="text-[10px] tracking-normal opacity-70">checking</span>;
  }
  const ok = state.data?.ok;
  return (
    <span
      title={state.data?.detail ?? "key not checked"}
      aria-label={ok ? "key valid" : "key rejected"}
      className={`h-2 w-2 shrink-0 border ${
        ok
          ? "border-[var(--accent-ink)] bg-[var(--accent-ink)]"
          : "border-[var(--alarm)] bg-transparent"
      }`}
    />
  );
}

/** The checkpoints this build ships a tested profile for. */
function LocalModels({ query, onDone }: { query: string; onDone: () => void }) {
  const catalog = useCatalog();
  const needle = query.trim().toLowerCase();
  const entries = (catalog.data ?? []).filter(
    (entry) =>
      !needle ||
      entry.label.toLowerCase().includes(needle) ||
      entry.repo_id.toLowerCase().includes(needle),
  );

  if (catalog.isLoading) {
    return <p className="p-4 text-xs text-[var(--ink-muted)]">Reading the catalog…</p>;
  }
  if (entries.length === 0) {
    return (
      <p className="dotted border border-[var(--rule)] p-6 text-center text-xs text-[var(--ink-muted)]">
        Nothing here matches “{query}”.
      </p>
    );
  }
  return (
    <ul className="border-t border-[var(--rule)]">
      {entries.map((entry) => (
        <LocalRow key={entry.id} entry={entry} onDone={onDone} />
      ))}
    </ul>
  );
}

function LocalRow({ entry, onDone }: { entry: CatalogEntry; onDone: () => void }) {
  const [open, setOpen] = useState(false);
  const load = useLoadModel();
  const health = useHealth();
  return (
    <li className="border-b border-[var(--rule)]">
      <div className="flex items-start gap-3 px-2 py-2">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="text-[11px] font-semibold text-[var(--ink)]">{entry.label}</span>
            <span className="font-mono text-[10px] text-[var(--ink-faint)]">{entry.repo_id}</span>
            {entry.loaded ? (
              <span className="flex h-4 items-center gap-1 border border-[var(--accent)] px-1 text-[10px] uppercase tracking-[0.1em] text-[var(--accent-ink)]">
                <IconCheck size={10} />
                loaded
              </span>
            ) : null}
          </div>
          <p className="mt-1 text-[10px] text-[var(--ink-faint)]">
            {/* When it will not run, the reason is the useful part — that is
                why the entry is listed at all rather than hidden. */}
            {entry.runnable
              ? `${entry.precision} · ~${Math.round(entry.total_vram_gb)}GB · ${entry.placement}`
              : entry.placement_reason}
          </p>
        </div>
        {entry.runnable && !entry.loaded ? (
          <button type="button" className="btn h-8 shrink-0" onClick={() => setOpen((was) => !was)}>
            <span>{open ? "Cancel" : "Load"}</span>
          </button>
        ) : null}
      </div>
      {open ? (
        <PlacementDialog
          entry={entry}
          health={health.data}
          pending={load.isPending}
          error={load.isError ? (load.error as Error).message : undefined}
          onLoad={(placement) =>
            load.mutate(
              { model: entry.id, placement: placement.mode, device: placement.device },
              {
                onSuccess: () => {
                  setOpen(false);
                  onDone();
                },
              },
            )
          }
          onClose={() => setOpen(false)}
        />
      ) : null}
    </li>
  );
}

/**
 * Where the weights go, for a checkpoint off the hub.
 *
 * A catalog entry gets the fuller PlacementDialog, which can say what fits
 * and what a swap unloads because the registry knows this model's real
 * footprints. A hub model's size is estimated from its files, so the same
 * dialog would be stating arithmetic it cannot stand behind — this stays a
 * plain choice of where to try.
 *
 * The planner is good at fitting the largest checkpoint, and that is exactly
 * why it is not always what you want: splitting a model across both cards is
 * what makes FLUX.2 [dev] runnable at all, and it is also what stops you doing
 * anything else with the second one. On a machine with more than one GPU that
 * is a decision, so it is offered as one.
 */
function PlacementChoice({ model, onDone }: { model: string; onDone: () => void }) {
  const gpus = useGpus();
  const load = useLoadModel();
  const cards = gpus.data ?? [];

  const start = (placement: Placement, device?: number) =>
    load.mutate({ model, placement, device }, { onSuccess: onDone });

  if (cards.length < 2) {
    // One card: there is nothing to choose, so nothing is asked.
    return (
      <div className="border-t border-[var(--rule)] bg-[var(--ground-sunk)] px-2 py-2">
        <button
          type="button"
          className="btn btn-primary h-8"
          disabled={load.isPending}
          onClick={() => start("auto")}
        >
          <span>{load.isPending ? "Starting…" : "Load it"}</span>
        </button>
        {load.isError ? <Failure error={load.error} /> : null}
      </div>
    );
  }

  return (
    <div className="border-t border-[var(--rule)] bg-[var(--ground-sunk)] px-2 py-2">
      <p className="label mb-2 text-[var(--ink-muted)]">where</p>
      <div className="flex flex-wrap gap-[2px]">
        <button
          type="button"
          className="btn h-8"
          disabled={load.isPending}
          onClick={() => start("split")}
        >
          <span>Split across both</span>
        </button>
        {cards.map((card: GpuInfo) => (
          <button
            key={card.index}
            type="button"
            className="btn h-8"
            disabled={load.isPending}
            onClick={() => start("single", card.index)}
            title={`${card.name} — ${Math.round(
              (card.memory_total_mb - card.memory_used_mb) / 1024,
            )}GB free`}
          >
            <span>Whole on GPU {card.index}</span>
          </button>
        ))}
      </div>
      <p className="mt-2 text-[10px] leading-relaxed text-[var(--ink-faint)]">
        Split fits the largest checkpoints and uses both cards. Pinning one whole to a
        single GPU leaves the other free — only for a model small enough to sit there.
      </p>
      {load.isError ? <Failure error={load.error} /> : null}
    </div>
  );
}

function Failure({ error }: { error: unknown }) {
  return (
    <p role="alert" className="mt-2 border border-[var(--alarm)] px-2 py-1 text-[10px] text-[var(--alarm-ink)]">
      {error instanceof Error ? error.message : "That load was refused."}
    </p>
  );
}

/** Anything on the hub, whether or not this build has a profile for it. */
function HubModels({ query, onDone }: { query: string; onDone: () => void }) {
  const results = useHubSearch(query);

  if (query.trim().length < 2) {
    return (
      <p className="border border-[var(--rule)] p-6 text-center text-xs text-[var(--ink-muted)]">
        Search the hub by name or author. Nothing is downloaded until you load it.
      </p>
    );
  }
  if (results.isLoading) {
    return <p className="p-4 text-xs text-[var(--ink-muted)]">Searching the hub…</p>;
  }
  if (results.isError) {
    return <Failure error={results.error} />;
  }
  const models = results.data ?? [];
  if (models.length === 0) {
    return (
      <p className="dotted border border-[var(--rule)] p-6 text-center text-xs text-[var(--ink-muted)]">
        The hub has no diffusion checkpoint matching “{query}”.
      </p>
    );
  }
  return (
    <ul className="border-t border-[var(--rule)]">
      {models.map((model: HubModel) => (
        <HubRow key={model.repo_id} model={model} onDone={onDone} />
      ))}
    </ul>
  );
}

function HubRow({ model, onDone }: { model: HubModel; onDone: () => void }) {
  const [open, setOpen] = useState(false);
  return (
    <li className="border-b border-[var(--rule)]">
      <div className="flex items-start gap-3 px-2 py-2">
        <div className="min-w-0 flex-1">
          <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
            <span className="font-mono text-[11px] text-[var(--ink)]">{model.repo_id}</span>
            {model.cached ? (
              <span className="flex h-4 items-center border border-[var(--rule-strong)] px-1 text-[10px] uppercase tracking-[0.1em] text-[var(--ink-faint)]">
                on disk
              </span>
            ) : null}
            {model.in_catalog ? (
              <span className="flex h-4 items-center border border-[var(--accent)] px-1 text-[10px] uppercase tracking-[0.1em] text-[var(--accent-ink)]">
                profiled
              </span>
            ) : null}
          </div>
          <p className="mt-1 flex flex-wrap gap-3 text-[10px] text-[var(--ink-faint)]">
            <span className="font-mono tabular">
              {Intl.NumberFormat(undefined, { notation: "compact" }).format(model.downloads)}{" "}
              downloads
            </span>
            {model.pipeline_tag ? (
              <span className="flex items-center gap-1">
                <IconImage size={10} /> {model.pipeline_tag}
              </span>
            ) : null}
            <span>{model.family}</span>
            {/* Outside the catalog the footprints are guesses, and a load that
                misplaces is the operator's to fix with the VRAM overrides. */}
            {!model.in_catalog ? <span>size estimated</span> : null}
          </p>
        </div>
        <button type="button" className="btn h-8 shrink-0" onClick={() => setOpen((was) => !was)}>
          <span>{open ? "Cancel" : model.cached ? "Load" : "Download & load"}</span>
        </button>
      </div>
      {open ? <PlacementChoice model={model.repo_id} onDone={onDone} /> : null}
    </li>
  );
}

/** A provider's catalog, where picking one means pinning it. */
function ProviderModels({
  providerId,
  query,
  onPicked,
}: {
  providerId: string;
  query: string;
  onPicked?: (key: string) => void;
}) {
  const models = useProviderModels(providerId, { q: query, kind: "image", limit: 40 });
  const pinned = usePinned();
  const pin = usePin();
  const unpin = useUnpin();

  if (models.isLoading) {
    return <p className="p-4 text-xs text-[var(--ink-muted)]">Reading the catalog…</p>;
  }
  if (models.isError) {
    return <Failure error={models.error} />;
  }
  return (
    <RemoteModelList
      models={models.data?.models ?? []}
      emptyLabel={query ? `Nothing matches “${query}”.` : "This provider lists nothing usable here."}
      action={(model) =>
        model.pinned ? (
          <button
            type="button"
            className="btn h-8 hover:!border-[var(--alarm)] hover:!text-[var(--alarm-ink)]"
            disabled={unpin.isPending}
            onClick={() => unpin.mutate(`${providerId}:${model.id}`)}
          >
            <span>Unpin</span>
            <IconClose size={12} />
          </button>
        ) : (
          <button
            type="button"
            className="btn btn-primary h-8"
            disabled={pin.isPending || !model.makes_images}
            onClick={() =>
              pin.mutate(
                { provider: providerId, modelId: model.id },
                { onSuccess: () => onPicked?.(`${providerId}:${model.id}`) },
              )
            }
          >
            <span>Use it</span>
            <IconCheck size={12} />
          </button>
        )
      }
      selectedId={(pinned.data ?? []).find((entry) => entry.provider === providerId)?.model_id}
    />
  );
}
