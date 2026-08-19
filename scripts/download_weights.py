#!/usr/bin/env python
"""Pre-download the FLUX.2 [dev] 4-bit weights into the HuggingFace cache.

Running this before the API avoids a first request that blocks for the length
of a 34GB download. Set HF_HOME to move the cache off the system disk.
"""

from __future__ import annotations

import argparse
import os
import sys
import time

DEFAULT_REPO = "diffusers/FLUX.2-dev-bnb-4bit"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-id", default=os.environ.get("OIG_REPO_ID", DEFAULT_REPO))
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Restrict to these subfolders (e.g. --only transformer vae).",
    )
    args = parser.parse_args()

    # xet/hf_transfer speeds up the big shards considerably.
    os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")

    from huggingface_hub import snapshot_download

    allow_patterns = None
    if args.only:
        allow_patterns = ["*.json", "*.jinja"] + [f"{name}/*" for name in args.only]

    started = time.perf_counter()
    print(f"downloading {args.repo_id} ...")
    path = snapshot_download(
        repo_id=args.repo_id,
        allow_patterns=allow_patterns,
        max_workers=8,
    )
    elapsed = time.perf_counter() - started
    print(f"done in {elapsed / 60:.1f} min")
    print(f"cache path: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
