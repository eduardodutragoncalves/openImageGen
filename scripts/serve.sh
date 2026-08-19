#!/usr/bin/env bash
# Start the API inside the conda environment.
set -euo pipefail

ENV_NAME="${OIG_CONDA_ENV:-openimagegen}"
HOST="${OIG_HOST:-0.0.0.0}"
PORT="${OIG_PORT:-8000}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found on PATH. Install Miniconda/Anaconda, or activate the" >&2
  echo "environment yourself and run: uvicorn app.main:app --host $HOST --port $PORT" >&2
  exit 1
fi

# conda's shell hook dereferences $PS1, which is unset in a non-interactive
# shell, so `set -u` would abort here. Relax nounset just for the activation.
set +u
# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"
set -u

cd "$(dirname "$0")/.."

# CUDA_VISIBLE_DEVICES is deliberately left alone: the service inspects every
# visible GPU at startup and places the models itself. Set it yourself to hide
# cards from the service.
export HF_HUB_ENABLE_HF_TRANSFER="${HF_HUB_ENABLE_HF_TRANSFER:-1}"
# 4-bit matmuls fragment the allocator; expandable segments buy back headroom
# on memory-constrained cards.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

exec uvicorn app.main:app --host "$HOST" --port "$PORT" --workers 1 "$@"
