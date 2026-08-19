"""In-process job queue.

A FLUX.2 [dev] generation can take minutes on consumer hardware, so requests
are queued and answered with a job id instead of blocking an HTTP worker. The queue
is deliberately in-process: the GPU state it guards lives in this process, and
a second process could not share it anyway.
"""

from __future__ import annotations

import logging
import queue
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Callable

from .schemas import GenerationResponse, JobState

logger = logging.getLogger(__name__)


class QueueFull(RuntimeError):
    """Raised when the backlog is at capacity."""


@dataclass
class Job:
    id: str
    kind: str
    payload: object
    # Which API key submitted this. History is scoped by it, so it is part of
    # the job, not of the request that happened to create it.
    owner: str = "local"
    model_id: str | None = None
    model_label: str | None = None
    state: JobState = JobState.queued
    created: int = field(default_factory=lambda: int(time.time()))
    started: int | None = None
    finished: int | None = None
    progress: float | None = None
    result: GenerationResponse | None = None
    error: str | None = None
    done: threading.Event = field(default_factory=threading.Event)

    def set_progress(self, value: float) -> None:
        self.progress = max(0.0, min(1.0, value))


class JobQueue:
    """Bounded FIFO queue drained by a fixed pool of worker threads.

    Worker threads (not processes) are enough here: every heavy call releases
    the GIL inside CUDA/torch, and keeping one address space avoids shipping
    model weights or images across a process boundary.
    """

    def __init__(
        self,
        handler: Callable[[Job], GenerationResponse],
        *,
        workers: int = 1,
        max_size: int = 32,
        ttl_seconds: int = 3600,
    ) -> None:
        self._handler = handler
        self._queue: queue.Queue[Job] = queue.Queue(maxsize=max_size)
        self._jobs: dict[str, Job] = {}
        self._order: list[str] = []
        self._lock = threading.Lock()
        self._ttl = ttl_seconds
        self._stopping = threading.Event()
        # Set means "workers may pick up work". Cleared while the model is
        # being swapped: 35GB of weights cannot be unloaded under a running
        # denoise loop, so the queue drains first and holds.
        self._gate = threading.Event()
        self._gate.set()
        self._active = 0
        self._idle = threading.Event()
        self._idle.set()
        self._on_change: Callable[[Job], None] | None = None
        self._threads = [
            threading.Thread(target=self._worker, name=f"flux2-worker-{i}", daemon=True)
            for i in range(max(1, workers))
        ]

    def on_change(self, callback: Callable[[Job], None]) -> None:
        """Called whenever a job changes state, so history can be persisted."""
        self._on_change = callback

    def _changed(self, job: Job) -> None:
        if self._on_change is None:
            return
        try:
            self._on_change(job)
        except Exception:  # noqa: BLE001 - history must never break generation
            logger.exception("job history update failed for %s", job.id)

    # -------------------------------------------------------------- draining
    def pause(self) -> None:
        """Stop handing work to the workers; a running job still finishes."""
        self._gate.clear()

    def resume(self) -> None:
        self._gate.set()

    @property
    def paused(self) -> bool:
        return not self._gate.is_set()

    @property
    def active(self) -> int:
        return self._active

    def wait_idle(self, timeout: float | None = None) -> bool:
        """Block until nothing is running. Pause first, or this may never win."""
        return self._idle.wait(timeout=timeout)

    # ------------------------------------------------------------- lifecycle
    def start(self) -> None:
        for thread in self._threads:
            thread.start()
        logger.info("job queue started with %d worker(s)", len(self._threads))

    def stop(self, timeout: float = 10.0) -> None:
        self._stopping.set()
        for _ in self._threads:
            try:
                self._queue.put_nowait(_SENTINEL)
            except queue.Full:  # pragma: no cover - best effort on shutdown
                pass
        for thread in self._threads:
            thread.join(timeout=timeout)

    # ---------------------------------------------------------------- public
    def submit(
        self,
        kind: str,
        payload: object,
        *,
        owner: str = "local",
        model_id: str | None = None,
        model_label: str | None = None,
    ) -> Job:
        job = Job(
            id=uuid.uuid4().hex,
            kind=kind,
            payload=payload,
            owner=owner,
            model_id=model_id,
            model_label=model_label,
        )
        with self._lock:
            self._evict_expired()
            self._jobs[job.id] = job
            self._order.append(job.id)
        try:
            self._queue.put_nowait(job)
        except queue.Full:
            with self._lock:
                self._jobs.pop(job.id, None)
                self._order.remove(job.id)
            raise QueueFull("generation queue is full, retry later") from None
        self._changed(job)
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

    def list(self, limit: int = 50, status: JobState | None = None) -> list[Job]:
        """Most recent jobs first, so a lost job id can be recovered."""
        with self._lock:
            jobs = [job for jid in self._order if (job := self._jobs.get(jid)) is not None]
        if status is not None:
            jobs = [job for job in jobs if job.state is status]
        return list(reversed(jobs))[:limit]

    def position(self, job: Job) -> int | None:
        """0 means "next to be picked up"; None once it left the queue."""
        if job.state is not JobState.queued:
            return None
        with self._lock:
            pending = [
                jid
                for jid in self._order
                if (j := self._jobs.get(jid)) is not None and j.state is JobState.queued
            ]
        try:
            return pending.index(job.id)
        except ValueError:
            return None

    @property
    def depth(self) -> int:
        return self._queue.qsize()

    # --------------------------------------------------------------- interns
    def _worker(self) -> None:
        while not self._stopping.is_set():
            # Held here while a model swap is in progress. The timeout keeps
            # the stop sentinel reachable during a shutdown mid-swap.
            if not self._gate.wait(timeout=0.5):
                continue
            try:
                job = self._queue.get(timeout=0.5)
            except queue.Empty:
                continue
            if job is _SENTINEL:
                self._queue.task_done()
                break
            with self._lock:
                self._active += 1
                self._idle.clear()
            job.state = JobState.running
            job.started = int(time.time())
            self._changed(job)
            try:
                job.result = self._handler(job)
                job.state = JobState.succeeded
            except RejectedContent as exc:
                job.state = JobState.rejected
                job.error = str(exc)
                logger.warning("job %s rejected: %s", job.id, exc)
            except Exception as exc:  # noqa: BLE001 - surfaced through the API
                job.state = JobState.failed
                job.error = f"{type(exc).__name__}: {exc}"
                logger.exception("job %s failed", job.id)
            finally:
                job.finished = int(time.time())
                job.done.set()
                with self._lock:
                    self._active -= 1
                    if self._active == 0:
                        self._idle.set()
                self._changed(job)
                self._queue.task_done()

    def _evict_expired(self) -> None:
        cutoff = time.time() - self._ttl
        keep: list[str] = []
        for job_id in self._order:
            job = self._jobs.get(job_id)
            if job is None:
                continue
            if job.finished is not None and job.finished < cutoff:
                self._jobs.pop(job_id, None)
            else:
                keep.append(job_id)
        self._order = keep


class RejectedContent(RuntimeError):
    """Prompt or image was blocked by the content filters."""


_SENTINEL = Job(id="__stop__", kind="__stop__", payload=None)
