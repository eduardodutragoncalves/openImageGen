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
        self._threads = [
            threading.Thread(target=self._worker, name=f"flux2-worker-{i}", daemon=True)
            for i in range(max(1, workers))
        ]

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
    def submit(self, kind: str, payload: object) -> Job:
        job = Job(id=uuid.uuid4().hex, kind=kind, payload=payload)
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
        return job

    def get(self, job_id: str) -> Job | None:
        with self._lock:
            return self._jobs.get(job_id)

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
            job = self._queue.get()
            if job is _SENTINEL:
                self._queue.task_done()
                break
            job.state = JobState.running
            job.started = int(time.time())
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
