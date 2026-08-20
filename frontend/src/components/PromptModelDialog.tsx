import { useState } from "react";
import { ApiError } from "../lib/api";
import type { RemoteModel } from "../lib/api";
import { useProviderModels, useProviders } from "../hooks/useApi";
import { Dialog } from "./Dialog";
import { RemoteModelList } from "./RemoteModelList";
import { Diagonal } from "./primitives";
import { IconKey, IconSearch } from "./Icons";

/**
 * Choosing which model rewrites the prompt.
 *
 * Searched live against the provider rather than picked from a pinned list:
 * the model that improves a prompt is a text model, and which one suits a
 * given prompt changes often enough that a saved shortlist would go stale.
 */
export function PromptModelDialog({
  selected,
  onPick,
  onClose,
}: {
  selected?: string;
  onPick: (modelId: string | undefined) => void;
  onClose: () => void;
}) {
  const [query, setQuery] = useState("");
  const providers = useProviders();
  const openrouter = (providers.data ?? []).find((entry) => entry.id === "openrouter");
  const models = useProviderModels("openrouter", { q: query, kind: "text", limit: 40 });

  function choose(model: RemoteModel) {
    onPick(model.id);
    onClose();
  }

  return (
    <Dialog
      title="Rewrite the prompt with"
      description="OpenRouter runs the model you pick over your prompt before generation, and the studio keeps both the original and the rewrite."
      onClose={onClose}
      wide
    >
      <div className="border-b border-[var(--rule)] p-3">
        <div className="relative">
          <IconSearch
            size={14}
            className="pointer-events-none absolute left-2 top-1/2 -translate-y-1/2 text-[var(--ink-faint)]"
          />
          <input
            value={query}
            onChange={(event) => setQuery(event.target.value)}
            placeholder="Search OpenRouter — try 'claude', 'gemini', 'vision'"
            aria-label="Search models"
            className="field h-9 pl-7 text-xs"
          />
        </div>
        {!openrouter?.configured ? (
          <p className="mt-2 flex items-start gap-2 border border-[var(--caution)] px-2 py-[6px] text-[11px] leading-snug text-[var(--caution-ink)]">
            <IconKey size={12} className="mt-[2px] shrink-0" />
            <span>
              OpenRouter has no API key yet, so the rewrite will fail at run time. Add one
              under Models → Web models.
            </span>
          </p>
        ) : null}
      </div>

      <div className="max-h-[52vh] min-h-0 flex-1 overflow-y-auto">
        {models.isError ? (
          <p role="alert" className="p-4 text-xs text-[var(--alarm-ink)]">
            {models.error instanceof ApiError
              ? models.error.message
              : "OpenRouter's catalog could not be read."}
          </p>
        ) : models.isLoading ? (
          <p className="p-4 text-xs text-[var(--ink-muted)]">Reading the catalog…</p>
        ) : (
          <RemoteModelList
            models={models.data?.models ?? []}
            emptyLabel={`Nothing on OpenRouter matches “${query}”.`}
            onSelect={choose}
            selectedId={selected}
          />
        )}
      </div>

      <footer className="flex shrink-0 items-center justify-between gap-3 border-t border-[var(--rule)] px-3 py-2">
        <span className="text-[10px] text-[var(--ink-faint)]">
          {selected ? (
            <>
              using <span className="font-mono text-[var(--ink-muted)]">{selected}</span>
            </>
          ) : (
            "no model chosen — the server default will be used"
          )}
        </span>
        <div className="flex gap-[2px]">
          {selected ? (
            <button
              type="button"
              className="btn h-8"
              onClick={() => {
                onPick(undefined);
                onClose();
              }}
            >
              <span>Use the default</span>
            </button>
          ) : null}
          <button type="button" className="btn h-8" onClick={onClose}>
            <span>Close</span>
            <Diagonal size={12} />
          </button>
        </div>
      </footer>
    </Dialog>
  );
}
