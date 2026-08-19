"""FastAPI application: image generation, the job archive, and the studio UI."""

from __future__ import annotations

import base64
import hashlib
import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import (
    Cookie,
    Depends,
    FastAPI,
    File,
    Form,
    Header,
    HTTPException,
    Query,
    Response,
    UploadFile,
)
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from .config import PROJECT_ROOT, Settings, get_settings
from .engines import EngineResult
from .images import InvalidImage, decode_image, encode_image, fit_to_budget, mime_type
from .jobs import Job, JobQueue, QueueFull
from .model_manager import ModelBusy, ModelManager, UnknownModel
from .schemas import (
    CatalogEntry,
    EditRequest,
    GenerationRequest,
    GenerationResponse,
    GpuInfo,
    HealthResponse,
    ImagePayload,
    JobImage,
    JobPage,
    JobState,
    JobStatusResponse,
    JobSubmitted,
    JobSummary,
    ModelInfo,
    ModelStatusResponse,
    ModelSwitchRequest,
    StorageInfo,
    UpsampleMode,
)
from .store import JobRecord, JobStore, directory_usage, enforce_retention

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("openimagegen")

WARMUP_TIMEOUT_S = 1800
SESSION_COOKIE = "oig_session"
STATIC_DIR = PROJECT_ROOT / "app" / "static"
# Paths the SPA fallback must never swallow.
API_PREFIXES = ("v1/", "healthz", "docs", "redoc", "openapi.json")


class AppState:
    """Mutable process-wide state shared by the routes."""

    def __init__(self) -> None:
        self.manager: ModelManager | None = None
        self.queue: JobQueue | None = None
        self.store: JobStore | None = None


state = AppState()


# ----------------------------------------------------------------------- auth
def key_owner(key: str) -> str:
    """Stable, non-reversible identity for one API key.

    History is scoped by this rather than by the key itself, so the archive
    can attribute work without the database ever holding a credential.
    """
    return hashlib.sha256(key.encode("utf-8")).hexdigest()[:16]


def _resolve_owner(
    settings: Settings, header_key: str | None, cookie: str | None
) -> str | None:
    """The caller's identity, or None when they are not authenticated."""
    if not settings.api_keys:
        return "local"
    if header_key and header_key in settings.api_keys:
        return key_owner(header_key)
    if cookie:
        # The cookie holds the hashed key, never the key: a stolen cookie is
        # already bad, but it should not also hand over the credential itself.
        for known in settings.api_keys:
            if key_owner(known) == cookie:
                return cookie
    return None


def require_owner(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    oig_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    settings: Settings = Depends(get_settings),
) -> str:
    owner = _resolve_owner(settings, x_api_key, oig_session)
    if owner is None:
        raise HTTPException(status_code=401, detail="invalid or missing API key")
    return owner


# --------------------------------------------------------------------- worker
def _handle_job(job: Job) -> GenerationResponse:
    """Runs on a queue worker thread."""
    manager = state.manager
    assert manager is not None

    if not manager.wait_ready(timeout=WARMUP_TIMEOUT_S):
        raise RuntimeError("models are still loading")
    engine = manager.engine
    if engine is None:
        raise RuntimeError(manager.status.detail or "no model is loaded")

    payload = job.payload
    assert isinstance(payload, dict)

    # The model may have been swapped between submission and execution; the
    # archive should record what actually produced the image.
    job.model_id = engine.spec.id
    job.model_label = engine.spec.label

    started = time.perf_counter()
    result: EngineResult = engine.generate(
        prompt=payload["prompt"],
        references=payload.get("references") or None,
        width=payload.get("width"),
        height=payload.get("height"),
        num_steps=payload.get("num_steps"),
        guidance=payload.get("guidance"),
        seed=payload.get("seed"),
        num_images=payload.get("num_images", 1),
        upsample_mode=payload.get("upsample_mode", "none"),
        progress=job.set_progress,
    )
    total = time.perf_counter() - started

    settings = get_settings()
    response_format = payload.get("response_format", "b64_json")
    output_format = payload.get("output_format", "png")

    images: list[ImagePayload] = []
    wrote_files = False
    for image, seed in zip(result.images, result.seeds):
        if response_format == "url":
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            name = f"{job.id}_{seed}.{output_format}"
            image.save(settings.output_dir / name)
            wrote_files = True
            images.append(
                ImagePayload(
                    url=f"/v1/files/{name}", seed=seed, width=image.width, height=image.height
                )
            )
        else:
            images.append(
                ImagePayload(
                    b64_json=encode_image(image, output_format),
                    seed=seed,
                    width=image.width,
                    height=image.height,
                )
            )

    if wrote_files:
        # Enforced on write, oldest first: ADR-003 trades base64 for disk, and
        # this is the cap that keeps the trade honest.
        enforce_retention(
            settings.output_dir,
            max_gb=settings.output_max_gb,
            max_age_days=settings.output_max_age_days,
        )

    return GenerationResponse(
        id=job.id,
        model=engine.spec.id,
        created=job.created,
        prompt=payload["prompt"],
        revised_prompt=result.revised_prompt,
        images=images,
        timings={**result.timings, "total_s": round(total, 3)},
    )


# -------------------------------------------------------------------- history
def _persist(job: Job) -> None:
    """Mirror a job into the archive on every state change."""
    store = state.store
    if store is None:
        return
    store.upsert(_record_for(job))


def _record_for(job: Job) -> JobRecord:
    payload = job.payload if isinstance(job.payload, dict) else {}
    result = job.result
    images = [
        {"url": img.url, "seed": img.seed, "width": img.width, "height": img.height}
        for img in (result.images if result else [])
    ]
    duration = None
    if result is not None:
        duration = result.timings.get("total_s")
    elif job.started and job.finished:
        duration = float(job.finished - job.started)

    return JobRecord(
        id=job.id,
        owner=job.owner,
        kind=job.kind,
        status=job.state.value,
        created=job.created,
        started=job.started,
        finished=job.finished,
        prompt=str(payload.get("prompt", "")),
        revised_prompt=result.revised_prompt if result else None,
        model_id=job.model_id,
        model_label=job.model_label,
        width=payload.get("width"),
        height=payload.get("height"),
        num_steps=payload.get("num_steps"),
        guidance=payload.get("guidance"),
        seed=payload.get("seed"),
        num_images=int(payload.get("num_images", 1) or 1),
        upsample_mode=payload.get("upsample_mode"),
        reference_count=len(payload.get("references") or []),
        images=images,
        error=job.error,
        duration_s=duration,
    )


def _summary(record: JobRecord, *, live: Job | None = None) -> JobSummary:
    settings = get_settings()
    images = []
    for entry in record.images:
        url = entry.get("url")
        available = True
        if url:
            available = (settings.output_dir / Path(url).name).is_file()
        images.append(
            JobImage(
                url=url,
                seed=int(entry.get("seed", 0)),
                width=int(entry.get("width", 0)),
                height=int(entry.get("height", 0)),
                available=available,
            )
        )

    progress = live.progress if live is not None else None
    queue_position = None
    if live is not None and state.queue is not None:
        queue_position = state.queue.position(live)

    return JobSummary(
        id=record.id,
        status=JobState(record.status),
        kind=record.kind,
        created=record.created,
        started=record.started,
        finished=record.finished,
        progress=progress,
        prompt=record.prompt,
        revised_prompt=record.revised_prompt,
        model_id=record.model_id,
        model_label=record.model_label,
        width=record.width,
        height=record.height,
        num_steps=record.num_steps,
        guidance=record.guidance,
        seed=record.seed,
        num_images=record.num_images,
        upsample_mode=record.upsample_mode,
        reference_count=record.reference_count,
        queue_position=queue_position,
        duration_s=record.duration_s,
        images=images,
        image_count=len(images),
        error=record.error,
    )


# ------------------------------------------------------------------- lifespan
def _check_exposure(settings: Settings) -> None:
    """Refuse to be an open image generator on a public interface."""
    if not settings.binds_publicly or settings.api_keys or settings.allow_open_access:
        return
    raise RuntimeError(
        f"OIG_HOST={settings.host} listens beyond this machine and OIG_API_KEYS is "
        "empty, which would leave generation open to anyone who can reach the port. "
        "Set OIG_API_KEYS=<comma-separated keys>, bind to 127.0.0.1, or set "
        "OIG_ALLOW_OPEN_ACCESS=true if something in front of this already "
        "authenticates."
    )


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    _check_exposure(settings)

    state.store = JobStore(settings.state_dir / "jobs.db")
    interrupted = state.store.mark_interrupted()
    if interrupted:
        logger.warning("%d job(s) were interrupted by the last restart", interrupted)

    state.manager = ModelManager(settings)
    state.queue = JobQueue(
        _handle_job,
        workers=settings.workers,
        max_size=settings.queue_max_size,
        ttl_seconds=settings.job_ttl_seconds,
    )
    state.queue.on_change(_persist)
    state.manager.attach_queue(state.queue)
    state.queue.start()
    state.manager.start_initial_load()

    if not settings.api_keys:
        logger.warning(
            "OIG_API_KEYS is empty: every caller shares one archive as 'local'"
        )

    yield

    if state.queue is not None:
        state.queue.stop()
    if state.manager is not None and state.manager.engine is not None:
        state.manager.engine.unload()


app = FastAPI(
    title="openImageGen",
    version="0.2.0",
    summary="FLUX image generation, editing and model management over HTTP",
    lifespan=lifespan,
)

_settings = get_settings()
if _settings.dev and _settings.dev_origins:
    # Only for `npm run dev`; the built UI is same-origin and needs no CORS.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_settings.dev_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )


# ------------------------------------------------------------------ utilities
def _gpu_report() -> list[GpuInfo]:
    if not torch.cuda.is_available():
        return []

    roles: dict[str, str] = {}
    manager = state.manager
    if manager is not None and manager.engine is not None:
        plan = manager.engine.plan
        roles[plan.transformer_device] = "transformer + vae"
        if plan.text_encoder_device != plan.transformer_device:
            roles[plan.text_encoder_device] = "text encoder"
        else:
            roles[plan.transformer_device] = "transformer + vae + text encoder"

    report = []
    for index in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(index)
        free, total = torch.cuda.mem_get_info(index)
        report.append(
            GpuInfo(
                index=index,
                name=props.name,
                memory_total_mb=total // (1024 * 1024),
                memory_used_mb=(total - free) // (1024 * 1024),
                role=roles.get(f"cuda:{index}"),
            )
        )
    return report


def _decode_references(payloads: list[str], settings: Settings):
    if len(payloads) > settings.max_reference_images:
        raise HTTPException(
            status_code=422,
            detail=f"at most {settings.max_reference_images} reference images are supported",
        )
    try:
        return [decode_image(p) for p in payloads]
    except InvalidImage as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


def _submit(kind: str, payload: dict, owner: str) -> JobSubmitted:
    assert state.queue is not None
    manager = state.manager
    assert manager is not None

    engine = manager.engine
    try:
        job = state.queue.submit(
            kind,
            payload,
            owner=owner,
            model_id=engine.spec.id if engine else manager.spec.id,
            model_label=engine.spec.label if engine else manager.spec.label,
        )
    except QueueFull as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    position = state.queue.position(job)
    return JobSubmitted(
        id=job.id,
        status=job.state,
        queue_position=position if position is not None else 0,
        poll_url=f"/v1/jobs/{job.id}",
    )


def _job_response(job: Job) -> JobStatusResponse:
    assert state.queue is not None
    return JobStatusResponse(
        id=job.id,
        status=job.state,
        created=job.created,
        started=job.started,
        finished=job.finished,
        queue_position=state.queue.position(job),
        progress=job.progress,
        result=job.result,
        error=job.error,
    )


def _build_payload(body: GenerationRequest | EditRequest, settings: Settings, references) -> dict:
    manager = state.manager
    engine = manager.engine if manager else None
    spec = engine.spec if engine else (manager.spec if manager else None)

    width = body.width or settings.default_width
    height = body.height or settings.default_height

    if isinstance(body, EditRequest) and body.match_image_size is not None:
        if body.match_image_size >= len(references):
            raise HTTPException(
                status_code=422,
                detail=f"match_image_size={body.match_image_size} is out of range",
            )
        width, height = references[body.match_image_size].size

    # The pixel cap comes from the resolved placement, not from a fixed
    # number: it depends on how much VRAM is left after the weights.
    max_pixels = engine.plan.max_pixels if engine else 1024 * 1024
    width, height = fit_to_budget(width, height, max_pixels)

    if references and spec is not None and not spec.supports_edit:
        raise HTTPException(
            status_code=409,
            detail=(
                f"{spec.label} is text-to-image only and cannot take reference "
                "images. Switch to a model with the image-edit capability."
            ),
        )
    if references and spec is not None and "multi-reference" not in spec.capabilities:
        if len(references) > 1:
            raise HTTPException(
                status_code=422,
                detail=f"{spec.label} accepts one reference image, not {len(references)}",
            )

    if body.upsample_prompt is UpsampleMode.local and engine is not None:
        if not engine.supports_local_upsample:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"{engine.spec.label} has no vision-language text encoder, so "
                    "local prompt upsampling is unavailable for it"
                ),
            )
    if body.upsample_prompt is UpsampleMode.openrouter and not settings.openrouter_api_key:
        raise HTTPException(
            status_code=400,
            detail="upsample_prompt='openrouter' requires OIG_OPENROUTER_API_KEY",
        )

    steps = body.num_steps or (engine.default_steps if engine else settings.default_steps)
    guidance = body.guidance
    if guidance is None:
        guidance = engine.default_guidance if engine else settings.default_guidance

    return {
        "prompt": body.prompt,
        "references": references,
        "width": width,
        "height": height,
        "num_steps": steps,
        "guidance": guidance,
        "seed": body.seed,
        "num_images": body.num_images,
        "upsample_mode": body.upsample_prompt.value,
        "response_format": body.response_format.value,
        "output_format": body.output_format.value,
    }


# ------------------------------------------------------------------- endpoints
@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    icon = STATIC_DIR / "favicon.ico"
    if icon.is_file():
        return FileResponse(icon)
    return Response(status_code=204)


@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz(settings: Settings = Depends(get_settings)) -> HealthResponse:
    manager = state.manager
    status_map = {"loading": "loading", "ready": "ok", "switching": "switching", "error": "error"}
    model_status = manager.status if manager else None
    used, count = directory_usage(settings.output_dir)

    return HealthResponse(
        status=status_map.get(model_status.state, "loading") if model_status else "loading",
        model_loaded=bool(manager and manager.engine and manager.engine.loaded),
        model=ModelStatusResponse(**model_status.as_dict())
        if model_status
        else ModelStatusResponse(state="loading", phase="starting", progress=0.0, started=0),
        queue_depth=state.queue.depth if state.queue else 0,
        queue_active=state.queue.active if state.queue else 0,
        queue_paused=state.queue.paused if state.queue else False,
        auth_required=settings.auth_required,
        gpus=_gpu_report(),
        storage=StorageInfo(
            used_bytes=used,
            file_count=count,
            max_bytes=int(settings.output_max_gb * 1024**3) if settings.output_max_gb else None,
            max_age_days=settings.output_max_age_days,
        ),
        detail=model_status.detail if model_status else None,
    )


# ----------------------------------------------------------------------- auth
@app.get("/v1/auth", tags=["auth"])
def whoami(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    oig_session: str | None = Cookie(default=None, alias=SESSION_COOKIE),
    settings: Settings = Depends(get_settings),
) -> dict:
    owner = _resolve_owner(settings, x_api_key, oig_session)
    return {
        "authenticated": owner is not None,
        "auth_required": settings.auth_required,
        "owner": owner,
    }


@app.post("/v1/auth", tags=["auth"])
def sign_in(
    body: dict,
    response: Response,
    settings: Settings = Depends(get_settings),
) -> dict:
    key = str(body.get("key", "")).strip()
    if not settings.api_keys:
        return {"authenticated": True, "auth_required": False, "owner": "local"}
    if key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="that key is not recognised")

    owner = key_owner(key)
    # HttpOnly so script cannot read it, and so <img src="/v1/files/..."> is
    # authenticated without putting the key in a URL.
    response.set_cookie(
        SESSION_COOKIE,
        owner,
        httponly=True,
        samesite="lax",
        max_age=60 * 60 * 24 * 30,
        path="/",
    )
    return {"authenticated": True, "auth_required": True, "owner": owner}


@app.delete("/v1/auth", tags=["auth"])
def sign_out(response: Response) -> dict:
    response.delete_cookie(SESSION_COOKIE, path="/")
    return {"authenticated": False}


# --------------------------------------------------------------------- models
@app.get("/v1/models", response_model=list[ModelInfo], tags=["models"])
def list_models() -> list[ModelInfo]:
    """The loaded model. A list for backwards compatibility with 0.1."""
    manager = state.manager
    if manager is None or manager.engine is None:
        raise HTTPException(status_code=503, detail="no model is loaded yet")
    return [ModelInfo(**manager.engine.describe())]


@app.get("/v1/models/catalog", response_model=list[CatalogEntry], tags=["models"])
def model_catalog() -> list[CatalogEntry]:
    """Every model this server knows, including ones it cannot run here."""
    manager = state.manager
    assert manager is not None
    return [CatalogEntry(**entry) for entry in manager.catalogue()]


@app.get("/v1/models/status", response_model=ModelStatusResponse, tags=["models"])
def model_status() -> ModelStatusResponse:
    manager = state.manager
    assert manager is not None
    return ModelStatusResponse(**manager.status.as_dict())


@app.post(
    "/v1/models/load",
    response_model=ModelStatusResponse,
    status_code=202,
    tags=["models"],
)
def load_model(
    body: ModelSwitchRequest,
    owner: str = Depends(require_owner),
) -> ModelStatusResponse:
    """Replace the loaded model. Returns immediately; poll /v1/models/status."""
    manager = state.manager
    assert manager is not None
    try:
        status = manager.switch(body.model)
    except UnknownModel as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ModelBusy as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    logger.info("model switch requested by %s: %s", owner, body.model)
    return ModelStatusResponse(**status.as_dict())


@app.get("/v1/gpus", response_model=list[GpuInfo], tags=["ops"])
def list_gpus() -> list[GpuInfo]:
    return _gpu_report()


# --------------------------------------------------------------------- images
@app.post(
    "/v1/images/generations",
    response_model=JobSubmitted,
    status_code=202,
    tags=["images"],
)
def create_generation(
    body: GenerationRequest,
    owner: str = Depends(require_owner),
    settings: Settings = Depends(get_settings),
) -> JobSubmitted:
    """Text to image. Returns a job id; poll /v1/jobs/{id} for the result."""
    return _submit("generation", _build_payload(body, settings, []), owner)


@app.post(
    "/v1/images/edits",
    response_model=JobSubmitted,
    status_code=202,
    tags=["images"],
)
def create_edit(
    body: EditRequest,
    owner: str = Depends(require_owner),
    settings: Settings = Depends(get_settings),
) -> JobSubmitted:
    """Edit or combine 1..N reference images supplied as base64/data URIs."""
    references = _decode_references(body.images, settings)
    return _submit("edit", _build_payload(body, settings, references), owner)


@app.post(
    "/v1/images/edits/upload",
    response_model=JobSubmitted,
    status_code=202,
    tags=["images"],
)
async def create_edit_upload(
    prompt: str = Form(...),
    images: list[UploadFile] = File(...),
    width: int | None = Form(default=None),
    height: int | None = Form(default=None),
    num_steps: int | None = Form(default=None),
    guidance: float | None = Form(default=None),
    seed: int | None = Form(default=None),
    num_images: int = Form(default=1),
    upsample_prompt: UpsampleMode = Form(default=UpsampleMode.none),
    match_image_size: int | None = Form(default=None),
    response_format: str = Form(default="b64_json"),
    owner: str = Depends(require_owner),
    settings: Settings = Depends(get_settings),
) -> JobSubmitted:
    """Same as /v1/images/edits but with multipart file uploads."""
    if len(images) > settings.max_reference_images:
        raise HTTPException(
            status_code=422,
            detail=f"at most {settings.max_reference_images} reference images are supported",
        )

    encoded = [base64.b64encode(await upload.read()).decode("ascii") for upload in images]
    body = EditRequest(
        prompt=prompt,
        images=encoded,
        width=width,
        height=height,
        num_steps=num_steps,
        guidance=guidance,
        seed=seed,
        num_images=num_images,
        upsample_prompt=upsample_prompt,
        match_image_size=match_image_size,
        response_format=response_format,
    )
    references = _decode_references(body.images, settings)
    return _submit("edit", _build_payload(body, settings, references), owner)


# ----------------------------------------------------------------------- jobs
@app.get("/v1/jobs", response_model=JobPage, tags=["jobs"])
def list_jobs(
    limit: int = Query(default=50, ge=1, le=200),
    offset: int = Query(default=0, ge=0),
    status: JobState | None = Query(default=None, description="Filter by job status"),
    kind: str | None = Query(default=None, description="'generation' or 'edit'"),
    model_id: str | None = Query(default=None),
    search: str | None = Query(default=None, description="Substring of the prompt"),
    owner: str = Depends(require_owner),
) -> JobPage:
    """The archive, newest first, scoped to the key that made the work."""
    store = state.store
    assert store is not None and state.queue is not None

    filters = {
        "owner": owner,
        "status": status.value if status else None,
        "kind": kind,
        "model_id": model_id,
        "search": search,
    }
    records = store.list(limit=limit, offset=offset, **filters)
    total = store.count(**filters)

    summaries = []
    for record in records:
        live = state.queue.get(record.id)
        summaries.append(_summary(record, live=live))
    return JobPage(jobs=summaries, total=total, limit=limit, offset=offset)


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
def get_job(
    job_id: str,
    wait: float = Query(default=0, ge=0, le=600, description="Seconds to block waiting"),
    owner: str = Depends(require_owner),
) -> JobStatusResponse:
    assert state.queue is not None and state.store is not None

    job = state.queue.get(job_id)
    if job is not None:
        if job.owner != owner:
            raise HTTPException(status_code=404, detail="unknown job id")
        if wait:
            job.done.wait(timeout=wait)
        return _job_response(job)

    # Not in memory: it belongs to an earlier run of this process.
    record = state.store.get(job_id)
    if record is None or record.owner != owner:
        raise HTTPException(status_code=404, detail="unknown job id")
    return JobStatusResponse(
        id=record.id,
        status=JobState(record.status),
        created=record.created,
        started=record.started,
        finished=record.finished,
        queue_position=None,
        progress=1.0 if record.status == "succeeded" else None,
        result=GenerationResponse(
            id=record.id,
            model=record.model_id or "",
            created=record.created,
            prompt=record.prompt,
            revised_prompt=record.revised_prompt,
            images=[
                ImagePayload(
                    url=entry.get("url"),
                    seed=int(entry.get("seed", 0)),
                    width=int(entry.get("width", 0)),
                    height=int(entry.get("height", 0)),
                )
                for entry in record.images
            ],
            timings={"total_s": record.duration_s or 0.0},
        )
        if record.status == "succeeded"
        else None,
        error=record.error,
    )


@app.delete("/v1/jobs/{job_id}", status_code=204, tags=["jobs"])
def delete_job(job_id: str, owner: str = Depends(require_owner)) -> Response:
    """Remove one job from the archive, and the files it produced."""
    store = state.store
    settings = get_settings()
    assert store is not None

    record = store.get(job_id)
    if record is None or record.owner != owner:
        raise HTTPException(status_code=404, detail="unknown job id")
    if record.status in ("queued", "running"):
        raise HTTPException(status_code=409, detail=f"job is {record.status}")

    for entry in record.images:
        url = entry.get("url")
        if not url:
            continue
        path = settings.output_dir / Path(url).name
        if path.is_file():
            path.unlink(missing_ok=True)
    store.delete(job_id)
    return Response(status_code=204)


@app.get("/v1/jobs/{job_id}/image", tags=["jobs"])
def get_job_image(
    job_id: str,
    index: int = Query(default=0, ge=0),
    owner: str = Depends(require_owner),
) -> Response:
    """Raw bytes of one produced image, handy for `curl -o out.png`."""
    assert state.queue is not None and state.store is not None
    settings = get_settings()

    job = state.queue.get(job_id)
    if job is not None:
        if job.owner != owner:
            raise HTTPException(status_code=404, detail="unknown job id")
        if job.state is not JobState.succeeded or job.result is None:
            raise HTTPException(status_code=409, detail=f"job is {job.state.value}")
        if index >= len(job.result.images):
            raise HTTPException(status_code=404, detail="image index out of range")
        payload = job.result.images[index]
        if payload.b64_json is None:
            return FileResponse(settings.output_dir / Path(payload.url or "").name)
        fmt = job.payload.get("output_format", "png") if isinstance(job.payload, dict) else "png"
        return Response(content=base64.b64decode(payload.b64_json), media_type=mime_type(fmt))

    record = state.store.get(job_id)
    if record is None or record.owner != owner:
        raise HTTPException(status_code=404, detail="unknown job id")
    if index >= len(record.images):
        raise HTTPException(status_code=404, detail="image index out of range")
    url = record.images[index].get("url")
    path = settings.output_dir / Path(url or "").name
    if not url or not path.is_file():
        raise HTTPException(status_code=410, detail="the file for this job is no longer on disk")
    return FileResponse(path)


@app.get("/v1/files/{name}", tags=["jobs"])
def get_file(
    name: str,
    owner: str = Depends(require_owner),
    settings: Settings = Depends(get_settings),
) -> FileResponse:
    # Reject traversal: only plain file names inside output_dir are served.
    if Path(name).name != name:
        raise HTTPException(status_code=400, detail="invalid file name")
    path = settings.output_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)


# ------------------------------------------------------------------------- ui
if STATIC_DIR.is_dir():
    assets = STATIC_DIR / "assets"
    if assets.is_dir():
        app.mount("/assets", StaticFiles(directory=assets), name="assets")


@app.get("/", include_in_schema=False)
def root() -> Response:
    index = STATIC_DIR / "index.html"
    settings = get_settings()
    if settings.serve_ui and index.is_file():
        return FileResponse(index)
    return RedirectResponse(url="/docs")


@app.get("/{full_path:path}", include_in_schema=False)
def spa_fallback(full_path: str) -> Response:
    """Serve the built studio, letting the client own its own routes.

    Registered last so it can never shadow /v1/*, and it refuses those
    prefixes explicitly rather than relying on ordering alone.
    """
    if full_path.startswith(API_PREFIXES):
        raise HTTPException(status_code=404, detail="not found")

    settings = get_settings()
    index = STATIC_DIR / "index.html"
    if not settings.serve_ui or not index.is_file():
        raise HTTPException(status_code=404, detail="not found")

    candidate = STATIC_DIR / full_path
    if full_path and candidate.is_file() and STATIC_DIR in candidate.resolve().parents:
        return FileResponse(candidate)
    return FileResponse(index)


def run() -> None:
    """Entry point used by scripts/serve.sh."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "app.main:app",
        host=settings.host,
        port=settings.port,
        workers=1,  # one process: the GPU state cannot be shared across workers
        log_level="info",
    )


if __name__ == "__main__":
    run()
