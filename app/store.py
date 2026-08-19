"""Durable job history and output retention.

Jobs used to live only in memory, which meant a restart erased every prompt,
seed and parameter set the operator had accumulated. The archive is the second
half of this product — "never lose an image" — so it goes to disk.

SQLite rather than a directory of JSON: the archive is browsed with filters and
search over hundreds of rows, which is a query, and sqlite3 is already in the
standard library. One connection guarded by a lock is plenty for a service that
runs one generation at a time.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)

SCHEMA_VERSION = 1

_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    owner           TEXT NOT NULL DEFAULT 'local',
    kind            TEXT NOT NULL,
    status          TEXT NOT NULL,
    created         INTEGER NOT NULL,
    started         INTEGER,
    finished        INTEGER,
    prompt          TEXT NOT NULL DEFAULT '',
    revised_prompt  TEXT,
    model_id        TEXT,
    model_label     TEXT,
    width           INTEGER,
    height          INTEGER,
    num_steps       INTEGER,
    guidance        REAL,
    seed            INTEGER,
    num_images      INTEGER NOT NULL DEFAULT 1,
    upsample_mode   TEXT,
    reference_count INTEGER NOT NULL DEFAULT 0,
    images          TEXT NOT NULL DEFAULT '[]',
    error           TEXT,
    duration_s      REAL
);
CREATE INDEX IF NOT EXISTS jobs_created_idx ON jobs (created DESC);
CREATE INDEX IF NOT EXISTS jobs_owner_idx   ON jobs (owner, created DESC);
CREATE INDEX IF NOT EXISTS jobs_status_idx  ON jobs (status, created DESC);
"""


@dataclass
class JobRecord:
    id: str
    owner: str
    kind: str
    status: str
    created: int
    started: int | None = None
    finished: int | None = None
    prompt: str = ""
    revised_prompt: str | None = None
    model_id: str | None = None
    model_label: str | None = None
    width: int | None = None
    height: int | None = None
    num_steps: int | None = None
    guidance: float | None = None
    seed: int | None = None
    num_images: int = 1
    upsample_mode: str | None = None
    reference_count: int = 0
    images: list[dict] = field(default_factory=list)
    error: str | None = None
    duration_s: float | None = None


class JobStore:
    """Every job this server has run, newest first."""

    def __init__(self, db_path: Path) -> None:
        db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        with self._lock:
            self._db.executescript(_SCHEMA)
            self._db.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._db.commit()
        logger.info("job history at %s", db_path)

    # ----------------------------------------------------------------- write
    def upsert(self, record: JobRecord) -> None:
        data = asdict(record)
        data["images"] = json.dumps(record.images)
        columns = ", ".join(data)
        placeholders = ", ".join(f":{key}" for key in data)
        updates = ", ".join(f"{key}=excluded.{key}" for key in data if key != "id")
        with self._lock:
            self._db.execute(
                f"INSERT INTO jobs ({columns}) VALUES ({placeholders}) "
                f"ON CONFLICT(id) DO UPDATE SET {updates}",
                data,
            )
            self._db.commit()

    def mark_interrupted(self) -> int:
        """Anything still queued or running belongs to a process that is gone.

        Leaving those rows as 'running' would show a progress bar that can
        never advance, so they are closed honestly at startup instead.
        """
        with self._lock:
            cursor = self._db.execute(
                "UPDATE jobs SET status='failed', finished=?, "
                "error='interrupted by a server restart' "
                "WHERE status IN ('queued','running')",
                (int(time.time()),),
            )
            self._db.commit()
            return cursor.rowcount

    def delete(self, job_id: str) -> bool:
        with self._lock:
            cursor = self._db.execute("DELETE FROM jobs WHERE id = ?", (job_id,))
            self._db.commit()
            return cursor.rowcount > 0

    # ------------------------------------------------------------------ read
    def get(self, job_id: str) -> JobRecord | None:
        with self._lock:
            row = self._db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return _to_record(row) if row else None

    def list(
        self,
        *,
        limit: int = 50,
        offset: int = 0,
        owner: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        model_id: str | None = None,
        search: str | None = None,
    ) -> list[JobRecord]:
        where, params = self._filters(owner, status, kind, model_id, search)
        sql = f"SELECT * FROM jobs {where} ORDER BY created DESC LIMIT ? OFFSET ?"
        with self._lock:
            rows = self._db.execute(sql, (*params, limit, offset)).fetchall()
        return [_to_record(row) for row in rows]

    def count(
        self,
        *,
        owner: str | None = None,
        status: str | None = None,
        kind: str | None = None,
        model_id: str | None = None,
        search: str | None = None,
    ) -> int:
        where, params = self._filters(owner, status, kind, model_id, search)
        with self._lock:
            row = self._db.execute(f"SELECT COUNT(*) AS n FROM jobs {where}", params).fetchone()
        return int(row["n"])

    def stats(self, *, owner: str | None = None) -> dict:
        """Totals for the archive header: how much work this key has produced."""
        clauses = ["status = 'succeeded'"]
        params: list = []
        if owner is not None:
            clauses.append("owner = ?")
            params.append(owner)
        where = "WHERE " + " AND ".join(clauses)
        with self._lock:
            row = self._db.execute(
                "SELECT COUNT(*) AS jobs, "
                "COALESCE(SUM(num_images), 0) AS images, "
                f"COALESCE(SUM(duration_s), 0) AS seconds FROM jobs {where}",
                params,
            ).fetchone()
        return {
            "jobs": int(row["jobs"]),
            "images": int(row["images"]),
            "seconds": float(row["seconds"]),
        }

    def models_used(self, *, owner: str | None = None) -> list[str]:
        where, params = self._filters(owner, None, None, None, None)
        with self._lock:
            rows = self._db.execute(
                f"SELECT DISTINCT model_id FROM jobs {where} ORDER BY model_id", params
            ).fetchall()
        return [row["model_id"] for row in rows if row["model_id"]]

    @staticmethod
    def _filters(
        owner: str | None,
        status: str | None,
        kind: str | None,
        model_id: str | None,
        search: str | None,
    ) -> tuple[str, list]:
        clauses: list[str] = []
        params: list = []
        if owner is not None:
            clauses.append("owner = ?")
            params.append(owner)
        if status:
            clauses.append("status = ?")
            params.append(status)
        if kind:
            clauses.append("kind = ?")
            params.append(kind)
        if model_id:
            clauses.append("model_id = ?")
            params.append(model_id)
        if search:
            clauses.append("(prompt LIKE ? OR revised_prompt LIKE ?)")
            params.extend([f"%{search}%", f"%{search}%"])
        if not clauses:
            return "", params
        return "WHERE " + " AND ".join(clauses), params


def _to_record(row: sqlite3.Row) -> JobRecord:
    data = dict(row)
    data["images"] = json.loads(data.get("images") or "[]")
    return JobRecord(**data)


# ------------------------------------------------------------------ retention
def enforce_retention(
    output_dir: Path,
    *,
    max_gb: float | None,
    max_age_days: int | None,
) -> list[str]:
    """Delete the oldest generated files until the directory fits its budget.

    Returns the names removed. Files are the only thing deleted; the job rows
    that referenced them stay, because "the image expired" is information the
    archive should keep showing rather than a gap it should hide.
    """
    if not output_dir.is_dir() or (max_gb is None and max_age_days is None):
        return []

    entries = []
    for path in output_dir.iterdir():
        if not path.is_file() or path.suffix.lower() not in (".png", ".jpg", ".jpeg", ".webp"):
            continue
        try:
            stat = path.stat()
        except OSError:  # pragma: no cover - raced with another deletion
            continue
        entries.append((stat.st_mtime, stat.st_size, path))

    entries.sort()  # oldest first
    removed: list[str] = []

    if max_age_days is not None:
        cutoff = time.time() - max_age_days * 86400
        for mtime, _size, path in list(entries):
            if mtime >= cutoff:
                break
            _unlink(path, removed)
            entries.remove((mtime, _size, path))

    if max_gb is not None:
        budget = int(max_gb * 1024**3)
        total = sum(size for _m, size, _p in entries)
        for mtime, size, path in list(entries):
            if total <= budget:
                break
            _unlink(path, removed)
            total -= size
            entries.remove((mtime, size, path))

    if removed:
        logger.info("retention removed %d file(s) from %s", len(removed), output_dir)
    return removed


def _unlink(path: Path, removed: list[str]) -> None:
    try:
        path.unlink()
        removed.append(path.name)
    except OSError as exc:  # pragma: no cover - best effort
        logger.warning("could not delete %s: %s", path, exc)


def directory_usage(output_dir: Path) -> tuple[int, int]:
    """(bytes, file count) currently held by generated images."""
    if not output_dir.is_dir():
        return 0, 0
    total = 0
    count = 0
    for path in output_dir.iterdir():
        if path.is_file():
            try:
                total += path.stat().st_size
            except OSError:  # pragma: no cover
                continue
            count += 1
    return total, count
