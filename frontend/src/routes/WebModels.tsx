import { useState } from "react";
import { ApiError } from "../lib/api";
import type { ProviderInfo } from "../lib/api";
import {
  useClearProviderKey,
  usePin,
  usePinned,
  useProviderModels,
  useProviders,
  useSetProviderKey,
  useUnpin,
} from "../hooks/useApi";
import { Readout } from "../components/primitives";
import { RemoteModelList } from "../components/RemoteModelList";
import { IconCheck, IconClose, IconKey, IconSearch } from "../components/Icons";

/**
 * Models this server does not host.
 *
 * The filter is the substance of the page: a provider lists hundreds of models
 * and only a handful of them output an image, so the default view is the
 * handful. Pinning one makes it a real generation target in the compose form,
 * alongside the checkpoint on the GPUs.
 */
export function WebModels() {
  const providers = useProviders();
  const [providerId, setProviderId] = useState<string>("openrouter");
  const [query, setQuery] = useState("");
  const [kind, setKind] = useState<"image" | "text" | "all">("image");

  const provider = (providers.data ?? []).find((entry) => entry.id === providerId);
  const models = useProviderModels(providerId, { q: query, kind, limit: 80 });
  const pinned = usePinned();
  const pin = usePin();
  const unpin = useUnpin();

  const pinnedHere = (pinned.data ?? []).filter((entry) => entry.provider === providerId);

  return (
    <div className="mx-auto max-w-[1400px] px-4 py-6">
      <div className="mb-5 flex flex-wrap items-end justify-between gap-6 border-b border-[var(--rule)] pb-4">
        <div>
          <h2 className="numeral text-3xl">WEB MODELS</h2>
          <p className="mt-2 max-w-[68ch] text-xs leading-relaxed text-[var(--ink-muted)]">
            Models reached over an API instead of loaded onto your GPUs. Pin the ones you
            want and they become choices in the compose form — no VRAM, no swap, and
            billed by the provider rather than by your electricity.
          </p>
        </div>
        <div className="flex gap-8">
          <Readout label="found" value={models.data?.total ?? "—"} size="md" />
          <Readout label="pinned" value={pinnedHere.length} size="md" tone="accent" />
        </div>
      </div>

      <nav className="mb-4 flex flex-wrap gap-[2px]" aria-label="Providers">
        {(providers.data ?? []).map((entry) => (
          <button
            key={entry.id}
            type="button"
            onClick={() => setProviderId(entry.id)}
            aria-pressed={entry.id === providerId}
            className={`flex h-9 items-center gap-2 border px-3 text-[11px] font-semibold uppercase tracking-[0.1em] transition-colors ${
              entry.id === providerId
                ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--ink-on-accent)]"
                : "border-[var(--rule)] text-[var(--ink-muted)] hover:border-[var(--rule-strong)] hover:text-[var(--ink)]"
            }`}
          >
            {entry.label}
            {entry.configured ? <IconCheck size={11} /> : null}
          </button>
        ))}
      </nav>

      {provider ? <ProviderKey provider={provider} /> : null}

      <div className="mb-3 flex flex-wrap items-center gap-2">
        <div className="relative min-w-[220px] flex-1">
          <IconSearch
            size={14}
            className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-[var(--ink-faint)]"
          />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder={`Search ${provider?.label ?? "the provider"}`}
            aria-label="Search models"
            className="field h-9 pl-7 text-xs"
          />
        </div>
        <div className="flex gap-[2px]">
          {(
            [
              ["image", "image generators"],
              ["text", "text models"],
              ["all", "everything"],
            ] as const
          ).map(([value, label]) => (
            <button
              key={value}
              type="button"
              onClick={() => setKind(value)}
              aria-pressed={kind === value}
              className={`h-9 border px-3 text-[10px] font-semibold uppercase tracking-[0.1em] transition-colors ${
                kind === value
                  ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--ink-on-accent)]"
                  : "border-[var(--rule)] text-[var(--ink-muted)] hover:border-[var(--rule-strong)] hover:text-[var(--ink)]"
              }`}
            >
              {label}
            </button>
          ))}
        </div>
      </div>

      {models.data ? (
        <p className="mb-2 text-[11px] text-[var(--ink-faint)]">
          {kind === "image" ? (
            <>
              <span className="font-mono tabular text-[var(--ink-muted)]">
                {models.data.total}
              </span>{" "}
              of {models.data.catalog_total} models on {provider?.label} can output an
              image.
            </>
          ) : (
            <>
              <span className="font-mono tabular text-[var(--ink-muted)]">
                {models.data.total}
              </span>{" "}
              of {models.data.catalog_total} shown.
            </>
          )}
        </p>
      ) : null}

      {models.isError ? (
        <p role="alert" className="border border-[var(--alarm)] px-3 py-2 text-xs text-[var(--alarm-ink)]">
          {models.error instanceof ApiError
            ? models.error.message
            : "That provider's catalog could not be read."}
        </p>
      ) : null}

      {models.isLoading ? (
        <p className="text-xs text-[var(--ink-muted)]">Reading the catalog…</p>
      ) : (
        <RemoteModelList
          models={models.data?.models ?? []}
          emptyLabel={
            query
              ? `Nothing on ${provider?.label} matches “${query}”.`
              : "This provider lists nothing of that kind."
          }
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
                title={model.makes_images ? undefined : "This model does not output images"}
                onClick={() => pin.mutate({ provider: providerId, modelId: model.id })}
              >
                <span>Pin</span>
                <IconCheck size={12} />
              </button>
            )
          }
        />
      )}
    </div>
  );
}

/**
 * The credential. Stored on the server and never sent back, so the field shows
 * whether one exists rather than what it is.
 */
function ProviderKey({ provider }: { provider: ProviderInfo }) {
  const [value, setValue] = useState("");
  const save = useSetProviderKey();
  const clear = useClearProviderKey();

  return (
    <section className="mb-4 border border-[var(--rule)] p-3">
      <div className="mb-2 flex flex-wrap items-center gap-2">
        <IconKey size={13} className="text-[var(--ink-muted)]" />
        <h3 className="label">api key</h3>
        {provider.configured ? (
          <span className="text-[10px] uppercase tracking-[0.1em] text-[var(--accent-ink)]">
            {provider.key_source === "env" ? "set in .env" : "stored on this server"}
          </span>
        ) : (
          <span className="text-[10px] uppercase tracking-[0.1em] text-[var(--caution-ink)]">
            not set
          </span>
        )}
      </div>

      <p className="mb-3 max-w-[68ch] text-[11px] leading-relaxed text-[var(--ink-muted)]">
        {provider.catalog_is_public
          ? "The catalog above is public, so you can browse without a key. Generating or rewriting a prompt needs one."
          : "This provider needs a key before its catalog can be read."}{" "}
        The key is kept on the server and never sent back to this page.
        {provider.key_url ? (
          <>
            {" "}
            <a href={provider.key_url} target="_blank" rel="noreferrer">
              Get one
            </a>
            .
          </>
        ) : null}
      </p>

      <form
        className="flex max-w-[560px] flex-wrap gap-[2px]"
        onSubmit={(event) => {
          event.preventDefault();
          if (value.trim()) {
            save.mutate(
              { provider: provider.id, key: value.trim() },
              { onSuccess: () => setValue("") },
            );
          }
        }}
      >
        <input
          type="password"
          value={value}
          onChange={(event) => setValue(event.target.value)}
          placeholder={provider.configured ? "replace the stored key" : "paste the key"}
          aria-label={`${provider.label} API key`}
          autoComplete="off"
          className="field h-9 min-w-[220px] flex-1 font-mono text-xs"
        />
        <button type="submit" className="btn h-9 shrink-0" disabled={!value.trim() || save.isPending}>
          <span>{save.isPending ? "Saving" : "Save"}</span>
        </button>
        {provider.configured && provider.key_source === "stored" ? (
          <button
            type="button"
            className="btn h-9 shrink-0 hover:!border-[var(--alarm)] hover:!text-[var(--alarm-ink)]"
            disabled={clear.isPending}
            onClick={() => clear.mutate(provider.id)}
          >
            <span>Remove</span>
          </button>
        ) : null}
      </form>

      {save.isError ? (
        <p role="alert" className="mt-2 text-[11px] text-[var(--alarm-ink)]">
          {save.error instanceof ApiError ? save.error.message : "That key was not accepted."}
        </p>
      ) : null}
    </section>
  );
}
