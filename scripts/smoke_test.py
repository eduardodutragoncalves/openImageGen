#!/usr/bin/env python
"""End-to-end check against a running API.

    python scripts/smoke_test.py --base-url http://localhost:8000
    python scripts/smoke_test.py --edit path/to/reference.png
"""

from __future__ import annotations

import argparse
import base64
import sys
import time
from pathlib import Path

import httpx


def poll(client: httpx.Client, base_url: str, job_id: str, timeout: float) -> dict:
    deadline = time.time() + timeout
    last_progress = -1.0
    while time.time() < deadline:
        response = client.get(f"{base_url}/v1/jobs/{job_id}", params={"wait": 15}, timeout=30)
        response.raise_for_status()
        job = response.json()
        progress = job.get("progress")
        if progress is not None and progress != last_progress:
            print(f"  {job['status']}: {progress * 100:5.1f}%")
            last_progress = progress
        if job["status"] in ("succeeded", "failed", "rejected"):
            return job
    raise TimeoutError(f"job {job_id} did not finish within {timeout}s")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://localhost:8000")
    parser.add_argument("--api-key", default=None)
    parser.add_argument(
        "--prompt",
        default=(
            "a photo of a forest with mist swirling around the tree trunks, the words "
            "'openImageGen' painted over it in big red brush strokes with visible texture"
        ),
    )
    parser.add_argument("--edit", type=Path, default=None, help="Reference image for an edit request")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--size", default="1024x1024")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=float, default=1800)
    parser.add_argument("--out", type=Path, default=Path("output/smoke.png"))
    args = parser.parse_args()

    width, height = (int(v) for v in args.size.lower().split("x"))
    headers = {"X-API-Key": args.api_key} if args.api_key else {}

    with httpx.Client(headers=headers) as client:
        health = client.get(f"{args.base_url}/healthz", timeout=30).json()
        print(f"health: {health['status']} (model_loaded={health['model_loaded']})")
        for gpu in health["gpus"]:
            print(
                f"  GPU{gpu['index']} {gpu['name']}: "
                f"{gpu['memory_used_mb']}/{gpu['memory_total_mb']} MB  [{gpu['role'] or '-'}]"
            )
        if health["status"] == "loading":
            print("models are still loading; the job will wait in the queue")

        payload: dict = {
            "prompt": args.prompt,
            "width": width,
            "height": height,
            "seed": args.seed,
            "response_format": "b64_json",
        }
        if args.steps:
            payload["num_steps"] = args.steps

        if args.edit:
            payload["images"] = [base64.b64encode(args.edit.read_bytes()).decode("ascii")]
            endpoint = "/v1/images/edits"
        else:
            endpoint = "/v1/images/generations"

        print(f"POST {endpoint}")
        submitted = client.post(f"{args.base_url}{endpoint}", json=payload, timeout=120)
        submitted.raise_for_status()
        job_id = submitted.json()["id"]
        print(f"job {job_id} queued at position {submitted.json()['queue_position']}")

        job = poll(client, args.base_url, job_id, args.timeout)

    if job["status"] != "succeeded":
        print(f"FAILED: {job['status']}: {job.get('error')}", file=sys.stderr)
        return 1

    result = job["result"]
    print("timings:", result["timings"])
    if result.get("revised_prompt"):
        print("revised prompt:", result["revised_prompt"][:200])

    args.out.parent.mkdir(parents=True, exist_ok=True)
    image = result["images"][0]
    args.out.write_bytes(base64.b64decode(image["b64_json"]))
    print(f"saved {args.out} ({image['width']}x{image['height']}, seed {image['seed']})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
