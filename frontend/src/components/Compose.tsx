import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQueryClient } from "@tanstack/react-query";
import { api, ApiError } from "../lib/api";
import type { Health, ModelInfo } from "../lib/api";
import { ASPECTS, fitToBudget, sizeForAspect } from "../lib/budget";
import { megapixels } from "../lib/format";
import { Diagonal, Field } from "./primitives";
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
  const [size, setSize] = useState<{ width: number; height: number } | null>(null);
  const [dragging, setDragging] = useState(false);
  const [fileError, setFileError] = useState<string | null>(null);
  const fileInput = useRef<HTMLInputElement>(null);

  const maxPixels = model?.max_pixels ?? 1024 * 1024;
  const maxReferences = model?.max_reference_images ?? 0;
  const guidanceRange = model?.guidance_range ?? [0, 20];
  const stepRange = model?.step_range ?? [1, 100];
  const guidanceFixed = guidanceRange[0] === guidanceRange[1];
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

  // "Reuse these settings" from a job in the archive: the whole point of the
  // archive is that a good run can be run again.
  useEffect(() => {
    if (!preset) return;
    if (preset.prompt != null) setPrompt(preset.prompt);
    if (preset.seed != null) setSeed(String(preset.seed));
    if (preset.width != null && preset.height != null) {
      setSize({ width: preset.width, height: preset.height });
    }
  }, [preset]);

  const effective = useMemo(
    () => fitToBudget(size?.width ?? 1024, size?.height ?? 1024, maxPixels),
    [size, maxPixels],
  );

  const warming = health?.model.state === "loading";
  const switching = health?.model.state === "switching";
  const broken = health?.model.state === "error";

  const submit = useMutation({
    mutationFn: async () => {
      const shared = {
        prompt: prompt.trim(),
        width: effective.width,
        height: effective.height,
        num_steps: steps ?? undefined,
        guidance: guidanceFixed ? undefined : (guidance ?? undefined),
        seed: seed.trim() === "" ? null : Number(seed),
        num_images: count,
        upsample_prompt: upsample,
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

        <div className="grid grid-cols-2 gap-4">
          <Field label="seed" hint="blank is random">
            <div className="flex items-center gap-[2px]">
              <input
                value={seed}
                onChange={(event) => setSeed(event.target.value.replace(/[^0-9]/g, ""))}
                placeholder="random"
                inputMode="numeric"
                className="field h-8 flex-1 font-mono text-xs tabular"
                aria-label="Seed"
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
            model && !model.supports_local_upsample
              ? "local needs a vision-language encoder"
              : undefined
          }
        >
          <div className="flex gap-[2px]">
            {(["none", "local", "openrouter"] as const).map((mode) => {
              const unavailable = mode === "local" && model && !model.supports_local_upsample;
              return (
                <button
                  key={mode}
                  type="button"
                  disabled={Boolean(unavailable)}
                  onClick={() => setUpsample(mode)}
                  title={
                    unavailable
                      ? `${model?.label} has no vision-language text encoder`
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

      <div className="shrink-0 border-t border-[var(--rule)] p-3">
        {warming || switching ? (
          <p className="mb-2 text-[11px] leading-relaxed text-[var(--ink-muted)]">
            {switching
              ? "The model is being replaced. Queue this now — it runs the moment the new weights are resident."
              : "The weights are still loading. Queue this now — it runs as soon as they land."}
          </p>
        ) : null}
        {broken ? (
          <p className="mb-2 text-[11px] leading-relaxed text-[var(--alarm-ink)]">
            {health?.detail ?? "No model is loaded."}
          </p>
        ) : null}
        <button type="submit" disabled={!ready} className="btn btn-primary w-full">
          <span>{submit.isPending ? "Submitting" : action}</span>
          <Diagonal size={14} />
        </button>
      </div>
    </form>
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
              <span className="text-[9px] uppercase tracking-[0.1em]">add</span>
            </button>
          ) : null}
          {files.length === 0 ? (
            <p className="ml-1 text-[11px] leading-tight text-[var(--ink-muted)]">
              Drop up to {maxReferences} image{maxReferences === 1 ? "" : "s"} to edit
              instead of generate.
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
