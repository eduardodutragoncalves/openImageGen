import { useEffect, useRef } from "react";
import type { ReactNode } from "react";
import { IconClose } from "./Icons";

/**
 * A modal, used only where the task genuinely needs protected focus: choosing
 * one model out of hundreds. Square, hairline-bordered, on the same grid as
 * everything else — the ground behind it dims rather than blurs, because this
 * world has no glass.
 */
export function Dialog({
  title,
  description,
  onClose,
  children,
  wide,
}: {
  title: string;
  description?: string;
  onClose: () => void;
  children: ReactNode;
  wide?: boolean;
}) {
  const panel = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (event.key === "Escape") onClose();
    };
    document.addEventListener("keydown", onKey);
    panel.current?.querySelector<HTMLElement>("input, button")?.focus();
    return () => document.removeEventListener("keydown", onKey);
  }, [onClose]);

  return (
    <div
      className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-[var(--ground-sunk)]/85 p-4 sm:p-10"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget) onClose();
      }}
    >
      <div
        ref={panel}
        role="dialog"
        aria-modal="true"
        aria-label={title}
        className={`flex w-full ${wide ? "max-w-[860px]" : "max-w-[520px]"} flex-col border border-[var(--rule-strong)] bg-[var(--ground)]`}
      >
        <header className="flex shrink-0 items-start justify-between gap-4 border-b border-[var(--rule)] px-4 py-3">
          <div>
            <h2 className="label">{title}</h2>
            {description ? (
              <p className="mt-1 max-w-[64ch] text-[11px] leading-relaxed text-[var(--ink-muted)]">
                {description}
              </p>
            ) : null}
          </div>
          <button
            type="button"
            onClick={onClose}
            aria-label="Close"
            className="flex h-6 w-6 shrink-0 items-center justify-center text-[var(--ink-muted)] transition-colors hover:text-[var(--ink)]"
          >
            <IconClose size={14} />
          </button>
        </header>
        {children}
      </div>
    </div>
  );
}
