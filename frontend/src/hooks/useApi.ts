import { useEffect, useMemo, useRef, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { api, Unauthorized } from "../lib/api";
import type { Placement } from "../lib/api";
import type { JobSummary } from "../lib/api";

const noRetryOn401 = (count: number, error: unknown) =>
  !(error instanceof Unauthorized) && count < 2;

/** Health drives the whole shell, so it polls faster while the machine is
 *  doing something the operator is waiting on. */
export function useHealth() {
  return useQuery({
    queryKey: ["health"],
    queryFn: api.health,
    refetchInterval: (query) => {
      const state = query.state.data?.model.state;
      return state === "switching" || state === "loading" ? 1000 : 4000;
    },
    retry: noRetryOn401,
  });
}

export function useAuth() {
  return useQuery({ queryKey: ["auth"], queryFn: api.whoami, retry: false });
}

export function useSignIn() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (key: string) => api.signIn(key),
    onSuccess: () => client.invalidateQueries(),
  });
}

/** The loaded model's real limits. Every control in the compose form is
 *  driven by this rather than by constants, because steps, guidance range and
 *  the pixel cap all change with the checkpoint and the hardware. */
export function useModel(enabled = true) {
  return useQuery({
    queryKey: ["model"],
    queryFn: api.models,
    select: (models) => models[0],
    enabled,
    retry: (count, error) => !(error instanceof Unauthorized) && count < 5,
    refetchInterval: (query) => (query.state.data ? false : 3000),
  });
}

export function useCatalog(enabled = true) {
  return useQuery({
    queryKey: ["catalog"],
    queryFn: api.catalog,
    enabled,
    staleTime: 30_000,
    retry: noRetryOn401,
  });
}

/** GPUs, for the placement choice. They change slowly; the readings on them
 *  do not, but /healthz is what watches those. */
export function useGpus(enabled = true) {
  return useQuery({
    queryKey: ["gpus"],
    queryFn: api.gpus,
    enabled,
    staleTime: 30_000,
    retry: noRetryOn401,
  });
}

/** The hub, searched only once the operator has typed something: an empty
 *  query returns whatever is most downloaded, which is not an answer. */
/** Clearing a card changes both what is on it and whether a model is loaded,
 *  so both readings are refetched rather than one. */
export function useReleaseGpu() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (index: number) => api.releaseGpu(index),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["health"] });
      client.invalidateQueries({ queryKey: ["gpus"] });
    },
  });
}

export function useHubSearch(query: string, enabled = true) {
  return useQuery({
    queryKey: ["hub-search", query],
    queryFn: () => api.hubSearch(query),
    enabled: enabled && query.trim().length > 1,
    staleTime: 120_000,
    retry: noRetryOn401,
    placeholderData: (previous) => previous,
  });
}

/** Whether a provider key actually works. Costs a request to the provider, so
 *  it is asked once per provider per opening and cached on the server too. */
export function useProviderCheck(provider: string, enabled = true) {
  return useQuery({
    queryKey: ["provider-check", provider],
    queryFn: () => api.checkProviderKey(provider),
    enabled,
    staleTime: 120_000,
    retry: false,
  });
}

export function useLoadModel() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({
      model,
      placement = "auto",
      device,
    }: {
      model: string;
      placement?: Placement;
      device?: number;
    }) => api.loadModel(model, placement, device),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["health"] });
      client.invalidateQueries({ queryKey: ["catalog"] });
    },
  });
}

const LIVE_STATES = new Set(["queued", "running"]);

/** The jobs the operator is currently waiting on. Unfiltered by design: the
 *  archive's filters must never hide what is running right now. */
export function useLiveJobs(enabled = true) {
  const query = useQuery({
    queryKey: ["jobs", "live"],
    queryFn: () => api.jobs({ limit: 25 }),
    enabled,
    retry: noRetryOn401,
    refetchInterval: (q) => {
      const rows = q.state.data?.jobs ?? [];
      return rows.some((job) => LIVE_STATES.has(job.status)) ? 1500 : 8000;
    },
  });

  const live = useMemo(
    () => (query.data?.jobs ?? []).filter((job) => LIVE_STATES.has(job.status)),
    [query.data],
  );
  return { ...query, live };
}

// ------------------------------------------------------------------ providers
export function useProviders(enabled = true) {
  return useQuery({
    queryKey: ["providers"],
    queryFn: api.providers,
    enabled,
    retry: noRetryOn401,
    staleTime: 30_000,
  });
}

/** A provider's catalog. The image filter is the default because a model that
 *  cannot output an image cannot generate one here. */
export function useProviderModels(
  provider: string,
  params: { q?: string; kind?: "image" | "text" | "all" | "community"; limit?: number },
  enabled = true,
) {
  return useQuery({
    queryKey: ["provider-models", provider, params],
    queryFn: () => api.providerModels(provider, params),
    enabled: enabled && Boolean(provider),
    retry: noRetryOn401,
    staleTime: 60_000,
    placeholderData: (previous) => previous,
  });
}

export function usePinned(enabled = true) {
  return useQuery({
    queryKey: ["pinned"],
    queryFn: api.pinned,
    enabled,
    retry: noRetryOn401,
    staleTime: 30_000,
  });
}

export function useSetProviderKey() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ provider, key }: { provider: string; key: string }) =>
      api.setProviderKey(provider, key),
    onSuccess: () => client.invalidateQueries({ queryKey: ["providers"] }),
  });
}

export function useClearProviderKey() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (provider: string) => api.clearProviderKey(provider),
    onSuccess: () => client.invalidateQueries({ queryKey: ["providers"] }),
  });
}

export function usePin() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: ({ provider, modelId }: { provider: string; modelId: string }) =>
      api.pin(provider, modelId),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["pinned"] });
      client.invalidateQueries({ queryKey: ["provider-models"] });
    },
  });
}

export function useUnpin() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (key: string) => api.unpin(key),
    onSuccess: () => {
      client.invalidateQueries({ queryKey: ["pinned"] });
      client.invalidateQueries({ queryKey: ["provider-models"] });
    },
  });
}

export interface ArchiveFilters {
  search?: string;
  status?: string;
  kind?: string;
  model_id?: string;
}

export function useArchive(filters: ArchiveFilters, limit: number, enabled = true) {
  return useQuery({
    queryKey: ["jobs", "archive", filters, limit],
    queryFn: () => api.jobs({ ...filters, limit }),
    enabled,
    retry: noRetryOn401,
    placeholderData: (previous) => previous,
    refetchInterval: 10_000,
  });
}

export function useJob(id: string | undefined) {
  return useQuery({
    queryKey: ["job", id],
    queryFn: () => api.job(id!),
    enabled: Boolean(id),
    retry: noRetryOn401,
    refetchInterval: (query) =>
      query.state.data && LIVE_STATES.has(query.state.data.status) ? 1500 : false,
  });
}

export function useDeleteJob() {
  const client = useQueryClient();
  return useMutation({
    mutationFn: (id: string) => api.deleteJob(id),
    onSuccess: () => client.invalidateQueries({ queryKey: ["jobs"] }),
  });
}

/**
 * Time remaining, measured rather than guessed.
 *
 * The per-step cost is only knowable from this run: it depends on the model,
 * the placement and the size. Samples accumulate as progress ticks, and the
 * estimate is withheld until there are enough of them to mean anything —
 * showing "5 minutes left" three seconds in would be a number the product
 * invented.
 */
export function useEta(jobId: string | undefined, progress: number | null | undefined) {
  const samples = useRef<{ at: number; progress: number }[]>([]);
  const currentJob = useRef<string | undefined>(undefined);
  const [eta, setEta] = useState<number | null>(null);

  useEffect(() => {
    if (jobId !== currentJob.current) {
      currentJob.current = jobId;
      samples.current = [];
      setEta(null);
    }
    if (jobId == null || progress == null || progress <= 0 || progress >= 1) return;

    const now = Date.now();
    const last = samples.current[samples.current.length - 1];
    if (last && last.progress === progress) return;
    samples.current.push({ at: now, progress });
    if (samples.current.length > 12) samples.current.shift();

    const first = samples.current[0];
    const span = (now - first.at) / 1000;
    const advanced = progress - first.progress;
    // Three samples is roughly three steps: enough for the rate to be real.
    if (samples.current.length < 3 || span < 2 || advanced <= 0) return;
    setEta(((1 - progress) / advanced) * span);
  }, [jobId, progress]);

  return eta;
}

/**
 * Tell them when it lands. Permission is asked once, after a job the operator
 * actually waited through — never on page load, which is the version everyone
 * has learned to dismiss.
 */
export function useCompletionNotice(jobs: JobSummary[]) {
  const seen = useRef<Map<string, string>>(new Map());
  const asked = useRef(false);

  useEffect(() => {
    for (const job of jobs) {
      const previous = seen.current.get(job.id);
      seen.current.set(job.id, job.status);
      if (previous !== "running" || job.status === "running") continue;
      if (document.visibilityState === "visible") continue;

      if (!asked.current && "Notification" in window && Notification.permission === "default") {
        asked.current = true;
        void Notification.requestPermission();
        continue;
      }
      if ("Notification" in window && Notification.permission === "granted") {
        const title =
          job.status === "succeeded"
            ? "Image ready"
            : job.status === "rejected"
              ? "Job refused by the content filter"
              : "Job failed";
        new Notification(title, { body: job.prompt.slice(0, 120), tag: job.id });
      }
    }
  }, [jobs]);
}

/** Unfinished work in the tab title, for the alt-tabbed operator. */
export function useTitleCount(runningCount: number, progress: number | null) {
  useEffect(() => {
    const base = "openImageGen";
    if (runningCount === 0) {
      document.title = base;
      return;
    }
    const pct = progress != null ? `${Math.round(progress * 100)}% ` : "";
    document.title = `${pct}· ${runningCount} running — ${base}`;
    return () => {
      document.title = base;
    };
  }, [runningCount, progress]);
}
