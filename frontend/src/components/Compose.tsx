import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { Health, ModelInfo } from "../lib/api";
import { ASPECTS, fitToBudget, sizeForAspect } from "../lib/budget";
import { megapixels } from "../lib/format";
import { Diagonal, Field } from "./primitives";
import { PromptModelDialog } from "./PromptModelDialog";
import { ModelPicker } from "./ModelPicker";
import { usePinned } from "../hooks/useApi";
import { IconClose, IconImage, IconMinus, IconPlus, IconRefresh, IconUpload } from "./Icons";

const MAX_REFERENCE_BYTES = 32 * 1024 * 1024;

/**
 * Compose. One form for both acts: attaching reference images is what makes it
 * an edit, so the operator never picks a mode. Every limit here is read from
 * /v1/models, and a control the loaded model cannot honour is disabled with
 * the reason rather than hidden.
 */
export interface ComposePreset {
  prompt?: string;
  seed?: number;
  width?: number;
  height?: number;
  numSteps?: number;
  guidance?: number;
  numImages?: number;
  upsampleMode?: string;
  /** Reference images to re-attach, by URL. */
  referenceUrls?: string[];
  /** Changes on every "reuse", so pressing it twice loads it twice. */
  stamp?: string;
}

export function Compose({
  model,
  health,
  preset,
}: {
  model?: ModelInfo;
  health?: Health;
  preset?: ComposePreset;
}) {
  const client = useQueryClient();
  const [prompt, setPrompt] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [steps, setSteps] = useState<number | null>(null);
  const [guidance, setGuidance] = useState<number | null>(null);
  const [seed, setSeed] = useState<string>("");
  const [count, setCount] = useState(1);
  const [upsample, setUpsample] = useState<"none" | "local" | "openrouter">("none");
  const [pickerOpen, setPickerOpen] = useState(false);
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const [restoring, setRestoring] = useState(false);
  // Where the image is made: the loaded checkpoint, or a pinned remote model.
  const [target, setTarget] = useState<string>("local");
  const [upsampleModel, setUpsampleModel] = useState<string | undefined>(undefined);
  const [promptDialog, setPromptDialog] = useState(false);
  const pinned = usePinned();
  const fileInput = useRef<HTMLInputElement>(null);

  const remote = (pinned.data ?? []).find((entry) => entry.key === target);
  const isRemote = Boolean(remote);

  const maxPixels = model?.max_pixels ?? 1024 * 1024;
  const guidanceRange = model?.guidance_range ?? [0, 20];
  const stepRange = model?.step_range ?? [1, 100];
  const guidanceFixed = guidanceRange[0] === guidanceRange[1];
  // A remote model is not bound by this machine's VRAM or by the local
  // checkpoint's capabilities, and exposes no step or guidance control.
  const maxReferences = isRemote
    ? remote?.reads_images
      ? 4
      : 0
    : (model?.max_reference_images ?? 0);
  const canEdit = maxReferences > 0;

  // Defaults follow the loaded model: switching to schnell must not leave a
  // 50 in the steps box that the model would then have to honour.
  useEffect(() => {
    if (!model) return;
    setSteps(Number(model.defaults.num_steps));
    setGuidance(Number(model.defaults.guidance));
    setSize((current) =>
      current ?? fitToBudget(Number(model.defaults.width), Number(model.defaults.height), model.max_pixels),
    );
  }, [model?.id]);

  useEffect(() => {
    if (!canEdit && files.length > 0) setFiles([]);
  }, [canEdit, files.length]);

  // "Reuse these settings" from a job in the archive. This has to restore the
  // whole request, not just the prompt: the run worth repeating is usually the
  // one that failed, and the operator is coming back to adjust one number.
  useEffect(() => {
    if (!preset) return;
    if (preset.prompt != null) setPrompt(preset.prompt);
    setSeed(preset.seed != null ? String(preset.seed) : "");
    if (preset.width != null && preset.height != null) {
      setSize({ width: preset.width, height: preset.height });
    }
    if (preset.numSteps != null) setSteps(preset.numSteps);
    if (preset.guidance != null) setGuidance(preset.guidance);
    if (preset.numImages != null) setCount(preset.numImages);
    if (preset.upsampleMode === "local" || preset.upsampleMode === "openrouter") {
      setUpsample(preset.upsampleMode);
    } else if (preset.upsampleMode === "none") {
      setUpsample("none");
    }

    const urls = preset.referenceUrls ?? [];
    if (urls.length === 0) {
      setFiles([]);
      return;
    }
    // The references were saved on the server when the job ran, so an edit
    // that was refused can be retried without hunting for the originals.
    let cancelled = false;
    setRestoring(true);
    Promise.all(
      urls.map(async (url, index) => {
        const response = await fetch(url, { credentials: "same-origin" });
        if (!response.ok) throw new Error(`reference ${index + 1} is gone`);
        const blob = await response.blob();
        const name = url.split("/").pop() ?? `reference-${index + 1}.png`;
        return new File([blob], name, { type: blob.type || "image/png" });
      }),
    )
      .then((restored) => {
        if (!cancelled) setFiles(restored);
      })
      .catch(() => {
        if (!cancelled) {
          setFiles([]);
          setFileError(
            "The reference images for that job are no longer on disk; attach them again.",
          );
        }
      })
      .finally(() => {
        if (!cancelled) setRestoring(false);
      });
    return () => {
      cancelled = true;
    };
  }, [preset?.stamp]);

  const effective = useMemo(
    () => fitToBudget(size?.width ?? 1024, size?.height ?? 1024, maxPixels),
    [size, maxPixels],
  );

  const warming = health?.model.state === "loading";
  const switching = health?.model.state === "switching";
  const broken = health?.model.state === "error";
  // "Loaded" means resident in VRAM right now, not "the model this build
  // prefers": it is the one thing about a local target worth a badge.
  const modelReady = health?.model.state === "ready";

  const submit = useMutation({
    mutationFn: async () => {
      const shared = {
        prompt: prompt.trim(),
        model: isRemote ? target : undefined,
        width: effective.width,
        height: effective.height,
        num_steps: isRemote ? undefined : (steps ?? undefined),
        guidance: isRemote || guidanceFixed ? undefined : (guidance ?? undefined),
        seed: seed.trim() === "" ? null : Number(seed),
        num_images: count,
        upsample_prompt: upsample,
        upsample_model: upsample === "openrouter" ? upsampleModel : undefined,
      };
      return files.length > 0 ? api.edit({ ...shared, files }) : api.generate(shared);
    },
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["jobs"] });
      // The prompt stays: the next run is nearly always a variation of it.
      setFiles([]);
    },
  });

  function addFiles(incoming: FileList | File[]) {
    setFileError(null);
    const accepted: File[] = [];
    for (const file of Array.from(incoming)) {
      if (!file.type.startsWith("image/")) {
        setFileError(`${file.name} is not an image.`);
        continue;
      }
      if (file.size > MAX_REFERENCE_BYTES) {
        setFileError(`${file.name} is over 32MB.`);
        continue;
      }
      accepted.push(file);
    }
    setFiles((current) => {
      const next = [...current, ...accepted];
      if (next.length > maxReferences) {
        setFileError(
          `${model?.label ?? "This model"} takes ${maxReferences} reference ${
            maxReferences === 1 ? "image" : "images"
          }.`,
        );
      }
      return next.slice(0, maxReferences);
    });
  }

  const ready = prompt.trim().length > 0 && !submit.isPending;
  const action = files.length > 0 ? "Edit" : "Generate";

  return (
    <form
      className="flex h-full min-h-0 flex-col"
      onSubmit={(event) => {
        event.preventDefault();
        if (ready) submit.mutate();
      }}
    >
      <div className="min-h-0 flex-1 space-y-5 overflow-y-auto p-3">
        <Field
          label="generate with"
          hint={isRemote ? "billed by the provider" : "your GPUs"}
        >
          {/* Grouped by where the work happens, because that is the difference
              that matters: one of these spends your VRAM and the rest spend
              your money. A flat wall of chips made a local checkpoint look
              like just another vendor's model. */}
          <div className="flex flex-col gap-3">
            <TargetGroup label="local" hint="on your GPUs">
              <TargetChip
                active={!isRemote}
                label={model?.label ?? "local model"}
                onClick={() => setTarget("local")}
                badge={modelReady ? "loaded" : undefined}
              />
            </TargetGroup>

            {PROVIDER_ORDER.map((provider) => {
              const entries = (pinned.data ?? []).filter(
                (entry) => entry.provider === provider,
              );
              if (entries.length === 0) return null;
              return (
                <TargetGroup
                  key={provider}
                  label={PROVIDER_LABEL[provider] ?? provider}
                  hint="billed per image"
                >
                  {entries.map((entry) => (
                    <TargetChip
                      key={entry.key}
                      active={target === entry.key}
                      label={entry.label}
                      onClick={() => setTarget(entry.key)}
                    />
                  ))}
                </TargetGroup>
              );
            })}

            {/* Outside the groups on purpose: this one dialog holds every way
                to answer "what makes this picture" — a checkpoint to load, one
                to download, a provider's catalog — so filing it under LOCAL
                would promise less than it opens. */}
            <button
              type="button"
              onClick={() => setPickerOpen(true)}
              className="flex h-8 items-center gap-1 self-start border border-dashed border-[var(--rule-strong)] px-3 text-[11px] text-[var(--ink-muted)] transition-colors hover:border-[var(--accent)] hover:text-[var(--ink)]"
            >
              <IconPlus size={12} />
              <span>Other model</span>
            </button>
          </div>
        </Field>

        <Field
          label="prompt"
          htmlFor="prompt"
          hint={`${prompt.length}/8000`}
        >
          <textarea
            id="prompt"
            value={prompt}
            onChange={(event) => setPrompt(event.target.value.slice(0, 8000))}
            onKeyDown={(event) => {
              if ((event.metaKey || event.ctrlKey) && event.key === "Enter" && ready) {
                event.preventDefault();
                submit.mutate();
              }
            }}
            rows={6}
            placeholder="Describe the image. Cmd/Ctrl + Enter to run."
            className="field resize-y font-sans text-sm leading-relaxed"
          />
        </Field>

        <ReferenceField
          files={files}
          maxReferences={maxReferences}
          canEdit={canEdit}
          modelLabel={model?.label}
          dragging={dragging}
          restoring={restoring}
          error={fileError}
          onPick={() => fileInput.current?.click()}
          onRemove={(index) => setFiles((current) => current.filter((_, i) => i !== index))}
          onDragState={setDragging}
          onDrop={addFiles}
        />
        <input
          ref={fileInput}
          type="file"
          accept="image/*"
          multiple
          className="sr-only"
          onChange={(event) => {
            if (event.target.files) addFiles(event.target.files);
            event.target.value = "";
          }}
        />

        <Field
          label="size"
          hint={
            effective.capped
              ? `capped to ${megapixels(maxPixels)} MP by this GPU`
              : `${megapixels(effective.width * effective.height)} MP`
          }
        >
          <div className="flex flex-wrap gap-[2px]">
            {ASPECTS.map((aspect) => {
              const candidate = sizeForAspect(aspect.ratio, maxPixels);
              const active =
                effective.width === candidate.width && effective.height === candidate.height;
              return (
                <button
                  key={aspect.label}
                  type="button"
                  aria-pressed={active}
                  onClick={() => setSize({ width: candidate.width, height: candidate.height })}
                  className={`h-8 border px-3 text-[11px] font-semibold tracking-[0.08em] transition-colors ${
                    active
                      ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--ink-on-accent)]"
                      : "border-[var(--rule)] text-[var(--ink-muted)] hover:border-[var(--rule-strong)] hover:text-[var(--ink)]"
                  }`}
                >
                  {aspect.label}
                </button>
              );
            })}
          </div>
          <div className="mt-2 flex items-center gap-2">
            <NumberBox
              label="w"
              value={effective.width}
              step={64}
              min={256}
              max={2048}
              onChange={(value) => setSize({ width: value, height: effective.height })}
            />
            <span className="text-[var(--ink-faint)]">×</span>
            <NumberBox
              label="h"
              value={effective.height}
              step={64}
              min={256}
              max={2048}
              onChange={(value) => setSize({ width: effective.width, height: value })}
            />
          </div>
        </Field>

        {isRemote ? (
          <p className="border border-[var(--rule)] px-3 py-2 text-[11px] leading-relaxed text-[var(--ink-muted)]">
            {remote?.label} runs on {remote?.provider}. Steps, guidance and the pixel cap
            are the provider's to decide, so they are not shown.
            {remote?.price_image ? (
              <>
                {" "}
                It bills about{" "}
                <span className="font-mono tabular">${remote.price_image}</span> per image.
              </>
            ) : null}
          </p>
        ) : (
        <div className="grid grid-cols-2 gap-4">
          <Field label="steps" hint={`${stepRange[0]}–${stepRange[1]}`}>
            <NumberBox
              label="steps"
              value={steps ?? stepRange[0]}
              min={stepRange[0]}
              max={stepRange[1]}
              step={1}
              onChange={setSteps}
              wide
            />
          </Field>

          <Field
            label="guidance"
            hint={guidanceFixed ? "not used by this model" : `${guidanceRange[0]}–${guidanceRange[1]}`}
          >
            <NumberBox
              label="guidance"
              value={guidance ?? 0}
              min={guidanceRange[0]}
              max={guidanceRange[1]}
              step={0.5}
              decimals={1}
              disabled={guidanceFixed}
              onChange={setGuidance}
              wide
            />
          </Field>
        </div>
        )}

        <div className="grid grid-cols-2 gap-4">
          <Field label="seed" hint="blank is random" htmlFor="seed">
            <div className="flex items-center gap-[2px]">
              <input
                id="seed"
                value={seed}
                onChange={(event) => setSeed(event.target.value.replace(/[^0-9]/g, ""))}
                placeholder="random"
                inputMode="numeric"
                className="field h-8 flex-1 font-mono text-xs tabular"
              />
              <button
                type="button"
                onClick={() => setSeed(String(Math.floor(Math.random() * 2 ** 31)))}
                className="flex h-8 w-8 shrink-0 items-center justify-center border border-[var(--rule)] text-[var(--ink-muted)] transition-colors hover:border-[var(--accent)] hover:text-[var(--accent-ink)]"
                aria-label="Pick a seed"
                title="Pick a seed"
              >
                <IconRefresh size={14} />
              </button>
            </div>
          </Field>

          <Field label="images" hint="per run">
            <div className="flex gap-[2px]">
              {[1, 2, 3, 4].map((value) => (
                <button
                  key={value}
                  type="button"
                  aria-pressed={count === value}
                  onClick={() => setCount(value)}
                  className={`h-8 flex-1 border text-[11px] font-semibold tabular transition-colors ${
                    count === value
                      ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--ink-on-accent)]"
                      : "border-[var(--rule)] text-[var(--ink-muted)] hover:border-[var(--rule-strong)] hover:text-[var(--ink)]"
                  }`}
                >
                  {value}
                </button>
              ))}
            </div>
          </Field>
        </div>

        <Field
          label="prompt upsampling"
          hint={
            isRemote
              ? "local runs on the loaded checkpoint"
              : model && !model.supports_local_upsample
                ? "local needs a vision-language encoder"
                : undefined
          }
        >
          <div className="flex gap-[2px]">
            {(["none", "local", "openrouter"] as const).map((mode) => {
              const unavailable =
                mode === "local" && (isRemote || (model && !model.supports_local_upsample));
              return (
                <button
                  key={mode}
                  type="button"
                  aria-pressed={upsample === mode}
                  disabled={Boolean(unavailable)}
                  // Choosing openrouter opens the picker: which model rewrites
                  // the prompt matters as much as the decision to rewrite it.
                  onClick={() => {
                    setUpsample(mode);
                    if (mode === "openrouter") setPromptDialog(true);
                  }}
                  title={
                    unavailable
                      ? isRemote
                        ? "Local upsampling uses the checkpoint on this machine"
                        : `${model?.label} has no vision-language text encoder`
                      : undefined
                  }
                  className={`h-8 flex-1 border text-[10px] font-semibold uppercase tracking-[0.1em] transition-colors ${
                    upsample === mode
                      ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--ink-on-accent)]"
                      : unavailable
                        ? "cursor-not-allowed border-[var(--rule)] text-[var(--ink-faint)]"
                        : "border-[var(--rule)] text-[var(--ink-muted)] hover:border-[var(--rule-strong)] hover:text-[var(--ink)]"
                  }`}
                >
                  {mode}
                </button>
              );
            })}
          </div>
          {upsample === "openrouter" ? (
            <button
              type="button"
              onClick={() => setPromptDialog(true)}
              className="mt-[6px] flex w-full items-center justify-between gap-2 border border-[var(--rule)] px-2 py-[6px] text-left text-[10px] transition-colors hover:border-[var(--accent)]"
            >
              <span className="truncate font-mono text-[var(--ink-muted)]">
                {upsampleModel ?? "server default"}
              </span>
              <span className="shrink-0 uppercase tracking-[0.1em] text-[var(--accent-ink)]">
                change
              </span>
            </button>
          ) : null}
        </Field>

        {submit.isError ? (
          <p
            role="alert"
            className="border border-[var(--alarm)] px-3 py-2 text-xs text-[var(--alarm-ink)]"
          >
            {submit.error instanceof ApiError
              ? submit.error.message
              : "The server did not accept that. Try again."}
          </p>
        ) : null}
      </div>

      {promptDialog ? (
        <PromptModelDialog
          selected={upsampleModel}
          onPick={setUpsampleModel}
          onClose={() => setPromptDialog(false)}
        />
      ) : null}

      {pickerOpen ? (
        <ModelPicker
          onClose={() => setPickerOpen(false)}
          // Pinning one is choosing it: the form switches to it rather than
          // making you find it again in the row of chips.
          onPicked={(key) => {
            setTarget(key);
            setPickerOpen(false);
          }}
        />
      ) : null}

      <div className="shrink-0 border-t border-[var(--rule)] p-3">
        {(warming || switching) && !isRemote ? (
          <p className="mb-2 text-[11px] leading-relaxed text-[var(--ink-muted)]">
            {switching
              ? "The model is being replaced. Queue this now — it runs the moment the new weights are resident."
              : "The weights are still loading. Queue this now — it runs as soon as they land."}
          </p>
        ) : null}
        {broken && !isRemote ? (
          <p className="mb-2 text-[11px] leading-relaxed text-[var(--alarm-ink)]">
            {health?.detail ?? "No model is loaded."}
          </p>
        ) : null}
        <button type="submit" disabled={!ready} className="btn btn-primary btn-shiny w-full">
          <span>{submit.isPending ? "Submitting" : action}</span>
          <Diagonal size={14} />
        </button>
      </div>
    </form>
  );
}

const PROVIDER_ORDER = ["openrouter", "runware"] as const;
const PROVIDER_LABEL: Record<string, string> = {
  openrouter: "OpenRouter",
  runware: "Runware",
};

/**
 * One row of targets under its own heading.
 *
 * The heading is what tells an OpenRouter model from a Runware one without
 * opening anything — the chips carry a vendor's product name, which says
 * nothing about who bills for it.
 */
function TargetGroup({
  label,
  hint,
  children,
}: {
  label: string;
  hint: string;
  children: React.ReactNode;
}) {
  return (
    <div className="flex flex-col gap-[6px]">
      <div className="flex items-baseline gap-2">
        <span className="label">{label}</span>
        <span className="text-[10px] text-[var(--ink-faint)]">{hint}</span>
      </div>
      <div className="flex flex-wrap gap-[2px]">{children}</div>
    </div>
  );
}

function TargetChip({
  active,
  label,
  onClick,
  badge,
}: {
  active: boolean;
  label: string;
  onClick: () => void;
  /** Marks the checkpoint actually resident in VRAM, which is the one fact
   *  about a local model an operator needs before pressing generate. */
  badge?: string;
}) {
  return (
    <button
      type="button"
      onClick={onClick}
      aria-pressed={active}
      className={`flex h-8 max-w-full items-center gap-2 border px-2 text-[10px] font-semibold uppercase tracking-[0.08em] transition-colors ${
        active
          ? "border-[var(--accent)] bg-[var(--accent)] text-[var(--ink-on-accent)]"
          : "border-[var(--rule)] text-[var(--ink-muted)] hover:border-[var(--rule-strong)] hover:text-[var(--ink)]"
      }`}
    >
      <span className="truncate">{label}</span>
      {badge ? (
        <span
          className={`shrink-0 px-1 py-[1px] text-[9px] tracking-[0.1em] ${
            active
              ? "bg-[var(--ink-on-accent)] text-[var(--accent)]"
              : "bg-[var(--accent)] text-[var(--ink-on-accent)]"
          }`}
        >
          {badge}
        </span>
      ) : null}
    </button>
  );
}

function NumberBox({
  label,
  value,
  min,
  max,
  step,
  decimals = 0,
  disabled,
  wide,
  onChange,
}: {
  label: string;
  value: number;
  min: number;
  max: number;
  step: number;
  decimals?: number;
  disabled?: boolean;
  wide?: boolean;
  onChange: (value: number) => void;
}) {
  const clamp = (next: number) => Math.min(max, Math.max(min, next));
  return (
    <div className={`flex items-center gap-[2px] ${wide ? "w-full" : ""}`}>
      <button
        type="button"
        disabled={disabled || value <= min}
        onClick={() => onChange(clamp(Number((value - step).toFixed(decimals))))}
        className="flex h-8 w-8 shrink-0 items-center justify-center border border-[var(--rule)] text-[var(--ink-muted)] transition-colors enabled:hover:border-[var(--accent)] enabled:hover:text-[var(--accent-ink)] disabled:text-[var(--ink-faint)]"
        aria-label={`Decrease ${label}`}
      >
        <IconMinus size={13} />
      </button>
      <input
        type="number"
        value={decimals ? value.toFixed(decimals) : value}
        min={min}
        max={max}
        step={step}
        disabled={disabled}
        onChange={(event) => onChange(clamp(Number(event.target.value)))}
        aria-label={label}
        className="field h-8 min-w-0 flex-1 text-center font-mono text-xs tabular disabled:text-[var(--ink-faint)] [appearance:textfield] [&::-webkit-inner-spin-button]:appearance-none [&::-webkit-outer-spin-button]:appearance-none"
      />
      <button
        type="button"
        disabled={disabled || value >= max}
        onClick={() => onChange(clamp(Number((value + step).toFixed(decimals))))}
        className="flex h-8 w-8 shrink-0 items-center justify-center border border-[var(--rule)] text-[var(--ink-muted)] transition-colors enabled:hover:border-[var(--accent)] enabled:hover:text-[var(--accent-ink)] disabled:text-[var(--ink-faint)]"
        aria-label={`Increase ${label}`}
      >
        <IconPlus size={13} />
      </button>
    </div>
  );
}

function ReferenceField({
  files,
  maxReferences,
  canEdit,
  modelLabel,
  dragging,
  restoring,
  error,
  onPick,
  onRemove,
  onDragState,
  onDrop,
}: {
  files: File[];
  maxReferences: number;
  canEdit: boolean;
  modelLabel?: string;
  dragging: boolean;
  restoring: boolean;
  error: string | null;
  onPick: () => void;
  onRemove: (index: number) => void;
  onDragState: (dragging: boolean) => void;
  onDrop: (files: FileList) => void;
}) {
  return (
    <Field
      label="references"
      hint={canEdit ? `${files.length}/${maxReferences}` : "text-to-image only"}
    >
      {canEdit ? (
        <div
          onDragOver={(event) => {
            event.preventDefault();
            onDragState(true);
          }}
          onDragLeave={() => onDragState(false)}
          onDrop={(event) => {
            event.preventDefault();
            onDragState(false);
            if (event.dataTransfer.files.length) onDrop(event.dataTransfer.files);
          }}
          className={`dotted flex min-h-[72px] flex-wrap items-center gap-2 border p-2 transition-colors ${
            dragging ? "border-[var(--accent)]" : "border-[var(--rule)]"
          }`}
        >
          {files.map((file, index) => (
            <ReferenceThumb
              key={`${file.name}-${index}`}
              file={file}
              onRemove={() => onRemove(index)}
            />
          ))}
          {files.length < maxReferences ? (
            <button
              type="button"
              onClick={onPick}
              className="flex h-14 w-14 flex-col items-center justify-center gap-1 border border-[var(--rule)] text-[var(--ink-faint)] transition-colors hover:border-[var(--accent)] hover:text-[var(--accent-ink)]"
              aria-label="Add a reference image"
            >
              <IconUpload size={14} />
              <span className="text-[10px] uppercase tracking-[0.1em]">add</span>
            </button>
          ) : null}
          {files.length === 0 ? (
            <p className="ml-1 text-[11px] leading-tight text-[var(--ink-muted)]">
              {restoring
                ? "Bringing the references back…"
                : `Drop up to ${maxReferences} image${maxReferences === 1 ? "" : "s"} to edit instead of generate.`}
            </p>
          ) : null}
        </div>
      ) : (
        <p className="border border-[var(--rule)] px-3 py-2 text-[11px] leading-relaxed text-[var(--ink-muted)]">
          {modelLabel ?? "This model"} cannot take reference images. Load a model with
          the image-edit capability to edit.
        </p>
      )}
      {error ? (
        <p role="alert" className="text-[11px] text-[var(--caution-ink)]">
          {error}
        </p>
      ) : null}
    </Field>
  );
}

function ReferenceThumb({ file, onRemove }: { file: File; onRemove: () => void }) {
  const [url, setUrl] = useState<string | null>(null);
  useEffect(() => {
    const objectUrl = URL.createObjectURL(file);
    setUrl(objectUrl);
    return () => URL.revokeObjectURL(objectUrl);
  }, [file]);

  return (
    <div className="group relative h-14 w-14 border border-[var(--rule)]">
      {url ? (
        <img src={url} alt={file.name} className="h-full w-full object-cover" />
      ) : (
        <IconImage className="m-auto text-[var(--ink-faint)]" />
      )}
      <button
        type="button"
        onClick={onRemove}
        className="absolute -right-px -top-px flex h-5 w-5 items-center justify-center bg-[var(--ground)] text-[var(--ink-muted)] opacity-0 transition-opacity hover:text-[var(--alarm-ink)] focus-visible:opacity-100 group-hover:opacity-100"
        aria-label={`Remove ${file.name}`}
      >
        <IconClose size={12} />
      </button>
    </div>
  );
}
