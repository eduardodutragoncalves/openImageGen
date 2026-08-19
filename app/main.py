"""FastAPI application exposing FLUX.2 image generation."""

from __future__ import annotations

import logging
import threading
import time
from contextlib import asynccontextmanager
from pathlib import Path

import torch
from fastapi import Depends, FastAPI, File, Form, Header, HTTPException, Query, UploadFile
from fastapi.responses import FileResponse, Response

from .config import Settings, get_settings
from .engine import EngineResult, Flux2Engine
from .images import InvalidImage, decode_image, encode_image, fit_to_budget, mime_type
from .jobs import Job, JobQueue, QueueFull
from .schemas import (
    EditRequest,
    GenerationRequest,
    GenerationResponse,
    GpuInfo,
    HealthResponse,
    ImagePayload,
    JobState,
    JobStatusResponse,
    JobSubmitted,
    ModelInfo,
    UpsampleMode,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
)
logger = logging.getLogger("openimagegen")

WARMUP_TIMEOUT_S = 1800


class AppState:
    """Mutable process-wide state shared by the routes."""

    def __init__(self) -> None:
        self.engine: Flux2Engine | None = None
        self.queue: JobQueue | None = None
        self.ready = threading.Event()
        self.load_error: str | None = None


state = AppState()


# --------------------------------------------------------------------- worker
def _handle_job(job: Job) -> GenerationResponse:
    """Runs on a queue worker thread."""
    if not state.ready.wait(timeout=WARMUP_TIMEOUT_S):
        raise RuntimeError(state.load_error or "models are still loading")
    if state.load_error:
        raise RuntimeError(state.load_error)

    assert state.engine is not None
    payload = job.payload
    assert isinstance(payload, dict)

    started = time.perf_counter()
    result: EngineResult = state.engine.generate(
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
    for image, seed in zip(result.images, result.seeds):
        if response_format == "url":
            settings.output_dir.mkdir(parents=True, exist_ok=True)
            name = f"{job.id}_{seed}.{output_format}"
            path = settings.output_dir / name
            image.save(path)
            images.append(
                ImagePayload(url=f"/v1/files/{name}", seed=seed, width=image.width, height=image.height)
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

    return GenerationResponse(
        id=job.id,
        model=state.engine.describe()["id"],
        created=job.created,
        prompt=payload["prompt"],
        revised_prompt=result.revised_prompt,
        images=images,
        timings={**result.timings, "total_s": round(total, 3)},
    )


def _load_models() -> None:
    """Background warm-up so the API answers /healthz while weights load."""
    try:
        assert state.engine is not None
        state.engine.load()
        state.ready.set()
        logger.info("engine ready")
    except Exception as exc:  # noqa: BLE001 - reported through /healthz
        state.load_error = f"{type(exc).__name__}: {exc}"
        logger.exception("model loading failed")
        state.ready.set()


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    state.engine = Flux2Engine(settings)
    state.queue = JobQueue(
        _handle_job,
        workers=settings.workers,
        max_size=settings.queue_max_size,
        ttl_seconds=settings.job_ttl_seconds,
    )
    state.queue.start()

    loader = threading.Thread(target=_load_models, name="flux2-loader", daemon=True)
    loader.start()

    yield

    if state.queue is not None:
        state.queue.stop()
    if state.engine is not None:
        state.engine.unload()


app = FastAPI(
    title="openImageGen",
    version="0.1.0",
    summary="FLUX.2 image generation and editing over HTTP",
    lifespan=lifespan,
)


# ----------------------------------------------------------------------- auth
def require_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
    settings: Settings = Depends(get_settings),
) -> None:
    if not settings.api_keys:
        return
    if x_api_key not in settings.api_keys:
        raise HTTPException(status_code=401, detail="invalid or missing X-API-Key")


# ------------------------------------------------------------------ utilities
def _gpu_report() -> list[GpuInfo]:
    if not torch.cuda.is_available():
        return []

    roles: dict[str, str] = {}
    if state.engine is not None:
        plan = state.engine.plan
        roles[plan.transformer_device] = "transformer + vae"
        # One card can hold both; keep a single, accurate label in that case.
        if plan.text_encoder_device == plan.transformer_device:
            roles[plan.transformer_device] = (
                "everything (cpu offload)" if plan.cpu_offload else "transformer + vae + text encoder"
            )
        else:
            roles[plan.text_encoder_device] = "text encoder + filters"
    report: list[GpuInfo] = []
    for index in range(torch.cuda.device_count()):
        free, total = torch.cuda.mem_get_info(index)
        props = torch.cuda.get_device_properties(index)
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


def _submit(kind: str, payload: dict) -> JobSubmitted:
    assert state.queue is not None
    try:
        job = state.queue.submit(kind, payload)
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
    max_pixels = state.engine.plan.max_pixels if state.engine else 1024 * 1024
    width, height = fit_to_budget(width, height, max_pixels)

    if body.upsample_prompt is UpsampleMode.openrouter and not settings.openrouter_api_key:
        raise HTTPException(
            status_code=400,
            detail="upsample_prompt='openrouter' requires OIG_OPENROUTER_API_KEY",
        )

    return {
        "prompt": body.prompt,
        "references": references,
        "width": width,
        "height": height,
        "num_steps": body.num_steps or settings.default_steps,
        "guidance": settings.default_guidance if body.guidance is None else body.guidance,
        "seed": body.seed,
        "num_images": body.num_images,
        "upsample_mode": body.upsample_prompt.value,
        "response_format": body.response_format.value,
        "output_format": body.output_format.value,
    }


# ------------------------------------------------------------------- endpoints
@app.get("/healthz", response_model=HealthResponse, tags=["ops"])
def healthz() -> HealthResponse:
    if state.load_error:
        status = "error"
    elif state.ready.is_set():
        status = "ok"
    else:
        status = "loading"
    return HealthResponse(
        status=status,
        model_loaded=state.engine is not None and state.engine.loaded and not state.load_error,
        queue_depth=state.queue.depth if state.queue else 0,
        gpus=_gpu_report(),
        detail=state.load_error,
    )


@app.get("/v1/models", response_model=list[ModelInfo], tags=["ops"])
def list_models() -> list[ModelInfo]:
    assert state.engine is not None
    return [ModelInfo(**state.engine.describe())]


@app.get("/v1/gpus", response_model=list[GpuInfo], tags=["ops"])
def list_gpus() -> list[GpuInfo]:
    return _gpu_report()


@app.post(
    "/v1/images/generations",
    response_model=JobSubmitted,
    status_code=202,
    tags=["images"],
    dependencies=[Depends(require_api_key)],
)
def create_generation(
    body: GenerationRequest,
    settings: Settings = Depends(get_settings),
) -> JobSubmitted:
    """Text to image. Returns a job id; poll /v1/jobs/{id} for the result."""
    return _submit("generation", _build_payload(body, settings, []))


@app.post(
    "/v1/images/edits",
    response_model=JobSubmitted,
    status_code=202,
    tags=["images"],
    dependencies=[Depends(require_api_key)],
)
def create_edit(
    body: EditRequest,
    settings: Settings = Depends(get_settings),
) -> JobSubmitted:
    """Edit or combine 1..N reference images supplied as base64/data URIs."""
    references = _decode_references(body.images, settings)
    return _submit("edit", _build_payload(body, settings, references))


@app.post(
    "/v1/images/edits/upload",
    response_model=JobSubmitted,
    status_code=202,
    tags=["images"],
    dependencies=[Depends(require_api_key)],
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
    settings: Settings = Depends(get_settings),
) -> JobSubmitted:
    """Same as /v1/images/edits but with multipart file uploads."""
    if len(images) > settings.max_reference_images:
        raise HTTPException(
            status_code=422,
            detail=f"at most {settings.max_reference_images} reference images are supported",
        )

    import base64

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
    )
    references = _decode_references(body.images, settings)
    return _submit("edit", _build_payload(body, settings, references))


@app.get("/v1/jobs/{job_id}", response_model=JobStatusResponse, tags=["jobs"])
def get_job(
    job_id: str,
    wait: float = Query(default=0, ge=0, le=600, description="Seconds to block waiting for the result"),
) -> JobStatusResponse:
    assert state.queue is not None
    job = state.queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    if wait:
        job.done.wait(timeout=wait)
    return _job_response(job)


@app.get("/v1/jobs/{job_id}/image", tags=["jobs"])
def get_job_image(job_id: str, index: int = Query(default=0, ge=0)) -> Response:
    """Raw bytes of one produced image, handy for `curl -o out.png`."""
    assert state.queue is not None
    job = state.queue.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="unknown job id")
    if job.state is not JobState.succeeded or job.result is None:
        raise HTTPException(status_code=409, detail=f"job is {job.state.value}")
    if index >= len(job.result.images):
        raise HTTPException(status_code=404, detail="image index out of range")

    payload = job.result.images[index]
    settings = get_settings()
    if payload.b64_json is None:
        return FileResponse(settings.output_dir / Path(payload.url or "").name)

    import base64

    fmt = "png"
    if isinstance(job.payload, dict):
        fmt = job.payload.get("output_format", "png")
    return Response(content=base64.b64decode(payload.b64_json), media_type=mime_type(fmt))


@app.get("/v1/files/{name}", tags=["jobs"])
def get_file(name: str, settings: Settings = Depends(get_settings)) -> FileResponse:
    # Reject traversal: only plain file names inside output_dir are served.
    if Path(name).name != name:
        raise HTTPException(status_code=400, detail="invalid file name")
    path = settings.output_dir / name
    if not path.is_file():
        raise HTTPException(status_code=404, detail="file not found")
    return FileResponse(path)


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
