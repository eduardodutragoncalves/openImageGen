#!/usr/bin/env bash
# curl reference for every openImageGen endpoint.
#
# This doubles as documentation and as a small CLI: read the function bodies
# for copy-pasteable curl calls, or run it directly to exercise the API.
#
# Usage:
#   scripts/api_examples.sh <command> [args...]
#   scripts/api_examples.sh --print-only <command> [args...]   # print the curl calls, don't run them
#
# Commands:
#   health                                   GET  /healthz
#   models                                   GET  /v1/models
#   gpus                                     GET  /v1/gpus
#   generate <prompt> [width] [height]       POST /v1/images/generations, prints the job id
#   generate-wait <prompt> <out.png> [w] [h] generate, then poll until done and save the image
#   edit <image> <prompt> <out.png>          POST /v1/images/edits/upload (multipart), poll, save
#   edit-json <image> <prompt> <out.png>     POST /v1/images/edits (base64 JSON body), poll, save
#   job <job_id> [wait_seconds]              GET  /v1/jobs/{id}
#   job-image <job_id> <out.png> [index]     GET  /v1/jobs/{id}/image
#   file <name> <out_path>                   GET  /v1/files/{name}
#
# Environment:
#   BASE_URL      API base URL (default: http://localhost:8000)
#   API_KEY       sent as X-API-Key when set; omitted otherwise
#   POLL_TIMEOUT  seconds before the pollers give up (default: 3600)
#
# Examples:
#   scripts/api_examples.sh health
#   scripts/api_examples.sh generate "a fox in a misty forest"
#   scripts/api_examples.sh generate-wait "a fox in a misty forest" output/fox.png
#   scripts/api_examples.sh edit output/fox.png "make the fog orange" output/fox_edited.png
#   BASE_URL=http://gpu-host:8000 API_KEY=key-one scripts/api_examples.sh health
#   scripts/api_examples.sh --print-only generate "a fox"

set -euo pipefail

BASE_URL="${BASE_URL:-http://localhost:8000}"
API_KEY="${API_KEY:-}"
# Upper bound on how long the pollers wait before giving up, in seconds.
POLL_TIMEOUT="${POLL_TIMEOUT:-3600}"
PRINT_ONLY=false

if [[ "${1:-}" == "--print-only" ]]; then
  PRINT_ONLY=true
  shift
fi

COMMAND="${1:-}"
shift || true

# ------------------------------------------------------------------- helpers
# A RETURN trap inside a function only fires with `set -T`, so temp files are
# tracked here and removed by a single EXIT trap, which always runs.
TMP_FILES=()
cleanup() { [[ ${#TMP_FILES[@]} -gt 0 ]] && rm -f "${TMP_FILES[@]}"; return 0; }
trap cleanup EXIT

# Registers a temp file for cleanup. Takes the name of the variable to fill,
# because assigning through $(...) would run in a subshell and the append to
# TMP_FILES would be lost with it.
new_tmp() {
  local -n _out_path="$1"
  _out_path=$(mktemp "${TMPDIR:-/tmp}/oig-$2.XXXXXX")
  TMP_FILES+=("$_out_path")
}

auth_args=()
if [[ -n "$API_KEY" ]]; then
  auth_args=(-H "X-API-Key: $API_KEY")
fi

# Prints the curl command about to run (so this script reads as documentation
# even when it's executing for real), then optionally runs it.
run_curl() {
  printf '+ curl %s\n' "$*" >&2
  if [[ "$PRINT_ONLY" == true ]]; then
    return 0
  fi
  curl "$@"
}

need_jq() {
  command -v jq >/dev/null 2>&1 || {
    echo "this command needs 'jq' (apt install jq / brew install jq)" >&2
    exit 1
  }
}

b64_to_file() {
  # Portable base64 -w0 (GNU) vs -b 0 (BSD/macOS) without depending on either
  # flag. Writes to a file rather than returning the string: a 1MP PNG is
  # ~1.9MB of base64, well past ARG_MAX once it is passed as an argument.
  base64 <"$1" | tr -d '\n' >"$2"
}

# Extracts the job id from a submission response, or aborts with whatever the
# server said. Without this, a failed POST yields the literal "null" and the
# poller spins forever against /v1/jobs/null.
job_id_or_die() {
  local submitted="$1" job_id
  need_jq
  job_id=$(printf '%s' "$submitted" | jq -r '.id // empty' 2>/dev/null || true)
  if [[ -z "$job_id" ]]; then
    echo "submission failed; server said:" >&2
    printf '%s\n' "${submitted:-<empty response>}" >&2
    exit 1
  fi
  printf '%s' "$job_id"
}

poll_job() {
  local job_id="$1"
  local deadline=$(( SECONDS + POLL_TIMEOUT ))
  need_jq
  echo "polling /v1/jobs/$job_id ..." >&2
  while true; do
    if (( SECONDS > deadline )); then
      echo "giving up after ${POLL_TIMEOUT}s waiting for job $job_id" >&2
      return 1
    fi

    local job
    job=$(run_curl -s "$BASE_URL/v1/jobs/$job_id" --get --data-urlencode "wait=30" "${auth_args[@]}")
    [[ "$PRINT_ONLY" == true ]] && return 0

    local status
    status=$(printf '%s' "$job" | jq -r '.status // empty' 2>/dev/null || true)
    if [[ -z "$status" ]]; then
      # 404, an error body, or no body at all: never retry this forever.
      echo "unexpected response while polling job $job_id:" >&2
      printf '%s\n' "${job:-<empty response>}" >&2
      return 1
    fi

    echo "  status=$status progress=$(printf '%s' "$job" | jq -r '.progress // "null"')" >&2
    case "$status" in
      succeeded|failed|rejected)
        printf '%s' "$job"
        return 0
        ;;
    esac
  done
}

save_first_image() {
  local job_json="$1" out_path="$2"
  need_jq
  mkdir -p "$(dirname "$out_path")"
  echo "$job_json" | jq -r '.result.images[0].b64_json' | base64 -d >"$out_path"
  echo "saved $out_path" >&2
}

# ------------------------------------------------------------------ commands
cmd_health() {
  # GET /healthz — service status, whether the model is loaded, queue depth,
  # and per-GPU memory/role.
  run_curl -s "$BASE_URL/healthz" "${auth_args[@]}" | { [[ "$PRINT_ONLY" == true ]] || (need_jq && jq .); }
}

cmd_models() {
  # GET /v1/models — which checkpoint is loaded, its placement, and defaults.
  run_curl -s "$BASE_URL/v1/models" "${auth_args[@]}" | { [[ "$PRINT_ONLY" == true ]] || (need_jq && jq .); }
}

cmd_gpus() {
  # GET /v1/gpus — VRAM usage per card and each one's role in the pipeline.
  run_curl -s "$BASE_URL/v1/gpus" "${auth_args[@]}" | { [[ "$PRINT_ONLY" == true ]] || (need_jq && jq .); }
}

cmd_generate() {
  local prompt="${1:?usage: generate <prompt> [width] [height]}"
  local width="${2:-1024}"
  local height="${3:-1024}"

  # POST /v1/images/generations — text-to-image. Returns 202 with a job id;
  # the image is produced asynchronously, see `job` / `generate-wait`.
  run_curl -s -X POST "$BASE_URL/v1/images/generations" \
    -H 'Content-Type: application/json' \
    "${auth_args[@]}" \
    -d "$(jq -n --arg p "$prompt" --argjson w "$width" --argjson h "$height" \
      '{prompt: $p, width: $w, height: $h}')"
}

cmd_generate_wait() {
  local prompt="${1:?usage: generate-wait <prompt> <out.png> [width] [height]}"
  local out="${2:?usage: generate-wait <prompt> <out.png> [width] [height]}"
  local width="${3:-1024}"
  local height="${4:-1024}"
  need_jq

  local submitted
  submitted=$(cmd_generate "$prompt" "$width" "$height")
  [[ "$PRINT_ONLY" == true ]] && return 0
  printf '%s\n' "$submitted" | jq .

  local job_id job
  job_id=$(job_id_or_die "$submitted")
  job=$(poll_job "$job_id") || return 1
  # Drop the base64 payload so the terminal stays readable.
  printf '%s' "$job" | jq 'del(.result.images[]?.b64_json)'

  local status
  status=$(printf '%s' "$job" | jq -r '.status')
  if [[ "$status" != "succeeded" ]]; then
    echo "job did not succeed (status=$status)" >&2
    return 1
  fi
  save_first_image "$job" "$out"
}

cmd_edit() {
  # POST /v1/images/edits/upload — multipart edit: reference image(s) as file
  # uploads plus form fields. Simplest option from the shell.
  local image="${1:?usage: edit <image> <prompt> <out.png>}"
  local prompt="${2:?usage: edit <image> <prompt> <out.png>}"
  local out="${3:?usage: edit <image> <prompt> <out.png>}"

  local submitted
  submitted=$(run_curl -s -X POST "$BASE_URL/v1/images/edits/upload" \
    -F "prompt=$prompt" \
    -F "images=@$image" \
    -F "match_image_size=0" \
    "${auth_args[@]}")
  [[ "$PRINT_ONLY" == true ]] && return 0
  need_jq
  printf '%s\n' "$submitted" | jq .

  local job_id job
  job_id=$(job_id_or_die "$submitted")
  job=$(poll_job "$job_id") || return 1
  printf '%s' "$job" | jq 'del(.result.images[]?.b64_json)'
  save_first_image "$job" "$out"
}

cmd_edit_json() {
  # POST /v1/images/edits — same as above, but with the reference image
  # inlined as base64 in a JSON body. Useful when scripting without curl's
  # multipart support, or when the caller already has the bytes in memory.
  #
  # The payload never travels through argv: base64 of a 1MP PNG is ~1.9MB,
  # which exceeds ARG_MAX. It goes to a temp file, jq reads it with --rawfile,
  # and curl posts it with --data-binary @file.
  local image="${1:?usage: edit-json <image> <prompt> <out.png>}"
  local prompt="${2:?usage: edit-json <image> <prompt> <out.png>}"
  local out="${3:?usage: edit-json <image> <prompt> <out.png>}"
  need_jq

  local tmp_b64 tmp_json
  new_tmp tmp_b64 b64
  new_tmp tmp_json body

  b64_to_file "$image" "$tmp_b64"
  jq -n --arg p "$prompt" --rawfile img "$tmp_b64" \
    '{prompt: $p, images: [$img], match_image_size: 0}' >"$tmp_json"

  local submitted
  submitted=$(run_curl -s -X POST "$BASE_URL/v1/images/edits" \
    -H 'Content-Type: application/json' \
    "${auth_args[@]}" \
    --data-binary "@$tmp_json")
  [[ "$PRINT_ONLY" == true ]] && return 0
  printf '%s\n' "$submitted" | jq .

  local job_id job
  job_id=$(job_id_or_die "$submitted")
  job=$(poll_job "$job_id") || return 1
  printf '%s' "$job" | jq 'del(.result.images[]?.b64_json)'
  save_first_image "$job" "$out"
}

cmd_job() {
  # GET /v1/jobs/{id} — status, progress and (once finished) the result.
  # ?wait=N long-polls up to N seconds instead of returning immediately.
  local job_id="${1:?usage: job <job_id> [wait_seconds]}"
  local wait="${2:-0}"
  run_curl -s "$BASE_URL/v1/jobs/$job_id" --get --data-urlencode "wait=$wait" "${auth_args[@]}" \
    | { [[ "$PRINT_ONLY" == true ]] || (need_jq && jq 'if .result then .result.images[].b64_json = "<omitted>" else . end'); }
}

cmd_job_image() {
  # GET /v1/jobs/{id}/image — raw bytes of one produced image (no base64,
  # no JSON wrapper). Handy for `curl -o`.
  local job_id="${1:?usage: job-image <job_id> <out.png> [index]}"
  local out="${2:?usage: job-image <job_id> <out.png> [index]}"
  local index="${3:-0}"
  mkdir -p "$(dirname "$out")"
  run_curl -s "$BASE_URL/v1/jobs/$job_id/image" --get --data-urlencode "index=$index" \
    "${auth_args[@]}" -o "$out"
  [[ "$PRINT_ONLY" == true ]] || echo "saved $out" >&2
}

cmd_file() {
  # GET /v1/files/{name} — retrieves a file saved when a request used
  # response_format="url" instead of the default b64_json.
  local name="${1:?usage: file <name> <out_path>}"
  local out="${2:?usage: file <name> <out_path>}"
  mkdir -p "$(dirname "$out")"
  run_curl -s "$BASE_URL/v1/files/$name" "${auth_args[@]}" -o "$out"
  [[ "$PRINT_ONLY" == true ]] || echo "saved $out" >&2
}

usage() {
  sed -n '2,32p' "$0" | sed 's/^# \{0,1\}//'
}

case "$COMMAND" in
  health) cmd_health "$@" ;;
  models) cmd_models "$@" ;;
  gpus) cmd_gpus "$@" ;;
  generate) cmd_generate "$@" ;;
  generate-wait) cmd_generate_wait "$@" ;;
  edit) cmd_edit "$@" ;;
  edit-json) cmd_edit_json "$@" ;;
  job) cmd_job "$@" ;;
  job-image) cmd_job_image "$@" ;;
  file) cmd_file "$@" ;;
  ""|help|-h|--help) usage ;;
  *)
    echo "unknown command: $COMMAND" >&2
    usage >&2
    exit 1
    ;;
esac
