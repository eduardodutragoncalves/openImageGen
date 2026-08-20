import type { RemoteModel } from "../lib/api";
import { IconCheck, IconImage, IconLayers } from "./Icons";

/**
 * A provider's models, as rows. Shared by the Web models tab and the prompt
 * dialog, because "pick one of these" is the same job in both places.
 */
export function RemoteModelList({
  models,
  emptyLabel,
  action,
  onSelect,
  selectedId,
}: {
  models: RemoteModel[];
  emptyLabel: string;
  /** Rendered at the right of each row. */
  action?: (model: RemoteModel) => React.ReactNode;
  onSelect?: (model: RemoteModel) => void;
  selectedId?: string;
}) {
  if (models.length === 0) {
    return (
      <p className="dotted border border-[var(--rule)] p-6 text-center text-xs text-[var(--ink-muted)]">
        {emptyLabel}
      </p>
    );
  }

  return (
    <ul className="border-t border-[var(--rule)]">
      {models.map((model) => {
        const selected = selectedId === model.id;
        const Row = onSelect ? "button" : "div";
        return (
          <li key={model.id} className="border-b border-[var(--rule)]">
            <Row
              {...(onSelect
                ? {
                    type: "button" as const,
                    onClick: () => onSelect(model),
                    "aria-pressed": selected,
                  }
                : {})}
              className={`flex w-full items-start gap-3 px-3 py-2 text-left transition-colors ${
                onSelect ? "hover:bg-[var(--accent-wash)]" : ""
              } ${selected ? "bg-[var(--accent-wash)]" : ""}`}
            >
              {model.cover_image ? (
                // A square on the grid, like every other thumbnail in the
                // system. It is the difference between a list of identifiers
                // and a catalog you can read at a glance.
                <img
                  src={model.cover_image}
                  alt=""
                  loading="lazy"
                  className="h-10 w-10 shrink-0 border border-[var(--rule)] object-cover"
                  onError={(event) => {
                    event.currentTarget.style.visibility = "hidden";
                  }}
                />
              ) : null}
              <div className="min-w-0 flex-1">
                <div className="flex flex-wrap items-baseline gap-x-2 gap-y-1">
                  <span className="text-[12px] font-semibold text-[var(--ink)]">{model.name}</span>
                  <span className="font-mono text-[10px] text-[var(--ink-faint)]">{model.id}</span>
                  {model.pinned ? (
                    <span className="flex h-4 items-center gap-1 border border-[var(--accent)] px-1 text-[10px] uppercase tracking-[0.1em] text-[var(--accent-ink)]">
                      <IconCheck size={10} />
                      pinned
                    </span>
                  ) : null}
                  {model.is_router ? (
                    <span className="flex h-4 items-center border border-[var(--rule-strong)] px-1 text-[10px] uppercase tracking-[0.1em] text-[var(--ink-faint)]">
                      router
                    </span>
                  ) : null}
                </div>
                {model.description ? (
                  <p className="mt-1 line-clamp-2 text-[11px] leading-snug text-[var(--ink-muted)]">
                    {model.description}
                  </p>
                ) : null}
                <div className="mt-1 flex flex-wrap items-center gap-3 text-[10px] text-[var(--ink-faint)]">
                  {model.makes_images ? (
                    <span className="flex items-center gap-1">
                      <IconImage size={10} /> makes images
                    </span>
                  ) : null}
                  {model.reads_images ? (
                    <span className="flex items-center gap-1">
                      <IconLayers size={10} /> reads images
                    </span>
                  ) : null}
                  {model.creator ? <span>{model.creator}</span> : null}
                  {model.price_image ? (
                    <span className="font-mono tabular">${model.price_image}/image</span>
                  ) : null}
                  {model.price_note ? (
                    // Quoted as the provider wrote it: a price rephrased is a
                    // price misquoted.
                    <span className="tabular">{model.price_note}</span>
                  ) : null}
                  {model.context_length ? (
                    <span className="font-mono tabular">
                      {Math.round(model.context_length / 1000)}k ctx
                    </span>
                  ) : null}
                </div>
              </div>
              {action ? <div className="shrink-0 pt-[2px]">{action(model)}</div> : null}
            </Row>
          </li>
        );
      })}
    </ul>
  );
}
