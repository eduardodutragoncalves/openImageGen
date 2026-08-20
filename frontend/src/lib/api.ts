import type { components } from "./api-types";

type S = components["schemas"];

export type Health = S["HealthResponse"];
export type GpuInfo = S["GpuInfo"];
export type ModelInfo = S["ModelInfo"];
export type CatalogEntry = S["CatalogEntry"];
export type ModelStatus = S["ModelStatusResponse"];
export type JobSummary = S["JobSummary"];
export type JobPage = S["JobPage"];
export type JobStatus = S["JobStatusResponse"];
export type JobSubmitted = S["JobSubmitted"];
export type JobImage = S["JobImage"];
export type JobRequest = S["JobRequest"];
export type ProviderInfo = S["ProviderInfoResponse"];
export type RemoteModel = S["RemoteModelInfo"];
export type RemoteModelPage = S["RemoteModelPage"];
export type PinnedModel = S["PinnedModelInfo"];
export type JobState = S["JobState"];
export type HubModel = S["HubModelInfo"];
export type ProviderCheck = S["ProviderCheckResponse"];
export type Placement = "auto" | "split" | "single";

/** A 401. The studio answers this with the key gate, never with a broken page.
 *  It carries the server's own wording: on the sign-in screen the reason a key
 *  was refused is the whole message, and a generic fallback there is worse
 *  copy than what the API already said. */
export class Unauthorized extends Error {
  constructor(detail = "This server wants an API key.") {
    super(detail);
    this.name = "Unauthorized";
  }
}

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
    this.name = "ApiError";
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    credentials: "same-origin",
    ...init,
    headers: {
      ...(init?.body && !(init.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...init?.headers,
    },
  });

  if (!response.ok) {
    let detail = `${response.status} ${response.statusText}`;
    try {
      const body = await response.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail) && body.detail[0]?.msg) detail = body.detail[0].msg;
    } catch {
      /* the body was not JSON; the status line is the best we have */
    }
    if (response.status === 401) throw new Unauthorized(detail);
    throw new ApiError(response.status, detail);
  }
  if (response.status === 204) return undefined as T;
  return response.json() as Promise<T>;
}

export interface GenerationInput {
  prompt: string;
  /** A pinned provider model ("openrouter:...") or undefined for the local one. */
  model?: string;
  upsample_model?: string;
  width?: number;
  height?: number;
  num_steps?: number;
  guidance?: number;
  seed?: number | null;
  num_images?: number;
  upsample_prompt?: "none" | "local" | "openrouter";
}

export const api = {
  health: () => request<Health>("/healthz"),

  whoami: () =>
    request<{ authenticated: boolean; auth_required: boolean; owner: string | null }>("/v1/auth"),

  signIn: (key: string) =>
    request<{ authenticated: boolean; owner: string }>("/v1/auth", {
      method: "POST",
      body: JSON.stringify({ key }),
    }),

  signOut: () => request<{ authenticated: boolean }>("/v1/auth", { method: "DELETE" }),

  models: () => request<ModelInfo[]>("/v1/models"),
  catalog: () => request<CatalogEntry[]>("/v1/models/catalog"),
  modelStatus: () => request<ModelStatus>("/v1/models/status"),
  loadModel: (model: string, placement: Placement = "auto", device?: number) =>
    request<ModelStatus>("/v1/models/load", {
      method: "POST",
      body: JSON.stringify({ model, placement, device }),
    }),

  gpus: () => request<GpuInfo[]>("/v1/gpus"),

  hubSearch: (q: string, limit = 30) =>
    request<HubModel[]>(
      `/v1/models/search?q=${encodeURIComponent(q)}&limit=${limit}`,
    ),

  checkProviderKey: (provider: string) =>
    request<ProviderCheck>(`/v1/providers/${provider}/check`, { method: "POST" }),

  generate: (input: GenerationInput) =>
    request<JobSubmitted>("/v1/images/generations", {
      method: "POST",
      body: JSON.stringify({ ...input, response_format: "url" }),
    }),

  edit: (input: GenerationInput & { files: File[]; match_image_size?: number | null }) => {
    const form = new FormData();
    form.set("prompt", input.prompt);
    for (const file of input.files) form.append("images", file);
    if (input.width != null) form.set("width", String(input.width));
    if (input.height != null) form.set("height", String(input.height));
    if (input.num_steps != null) form.set("num_steps", String(input.num_steps));
    if (input.guidance != null) form.set("guidance", String(input.guidance));
    if (input.seed != null) form.set("seed", String(input.seed));
    if (input.num_images != null) form.set("num_images", String(input.num_images));
    if (input.upsample_prompt) form.set("upsample_prompt", input.upsample_prompt);
    if (input.upsample_model) form.set("upsample_model", input.upsample_model);
    if (input.model) form.set("model", input.model);
    if (input.match_image_size != null) {
      form.set("match_image_size", String(input.match_image_size));
    }
    form.set("response_format", "url");
    return request<JobSubmitted>("/v1/images/edits/upload", { method: "POST", body: form });
  },

  providers: () => request<ProviderInfo[]>("/v1/providers"),

  setProviderKey: (provider: string, key: string) =>
    request<ProviderInfo>(`/v1/providers/${provider}/key`, {
      method: "PUT",
      body: JSON.stringify({ key }),
    }),

  clearProviderKey: (provider: string) =>
    request<ProviderInfo>(`/v1/providers/${provider}/key`, { method: "DELETE" }),

  providerModels: (
    provider: string,
    params: { q?: string; kind?: "image" | "text" | "all" | "community"; limit?: number },
  ) => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== "") query.set(key, String(value));
    }
    return request<RemoteModelPage>(`/v1/providers/${provider}/models?${query}`);
  },

  pinned: () => request<PinnedModel[]>("/v1/providers/pinned"),

  pin: (provider: string, modelId: string) =>
    request<PinnedModel>(`/v1/providers/${provider}/pin`, {
      method: "POST",
      body: JSON.stringify({ model_id: modelId }),
    }),

  unpin: (key: string) =>
    request<void>(`/v1/providers/pinned?key=${encodeURIComponent(key)}`, { method: "DELETE" }),

  jobs: (params: {
    limit?: number;
    offset?: number;
    status?: string;
    kind?: string;
    model_id?: string;
    search?: string;
  }) => {
    const query = new URLSearchParams();
    for (const [key, value] of Object.entries(params)) {
      if (value !== undefined && value !== null && value !== "") query.set(key, String(value));
    }
    return request<JobPage>(`/v1/jobs?${query}`);
  },

  /** `wait` holds the connection open server-side; ADR-002's long poll. */
  job: (id: string, wait = 0) => request<JobStatus>(`/v1/jobs/${id}?wait=${wait}`),

  deleteJob: (id: string) => request<void>(`/v1/jobs/${id}`, { method: "DELETE" }),
};
