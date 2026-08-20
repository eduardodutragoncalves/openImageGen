#!/usr/bin/env bash
# Bring the whole thing up with one command, and take it back down with one
# Ctrl-C.
#
# There are two ways to run this project and they are not the same shape.
#
#   dev  (default) two processes: the API, and Vite serving the studio with
#        hot reload and proxying /v1. The API has to be told to allow the dev
#        origin, and you edit the frontend without rebuilding anything.
#
#   prod one process: the studio is built into app/static and the API serves
#        it, so there is one origin, one port and no CORS. This is what you
#        deploy and what ./scripts/serve.sh alone gives you.
#
# Either way this script stops whatever it finds already listening, waits until
# the thing is actually answering rather than merely started, and opens it.
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

MODE="dev"
OPEN_BROWSER=1
API_PORT="${OIG_PORT:-8000}"
WEB_PORT="${OIG_WEB_PORT:-5173}"
LOG_DIR="$ROOT/logs"

usage() {
  cat <<'USAGE'
usage: scripts/start.sh [--dev|--prod] [--no-open] [--api-port N] [--web-port N]

  --dev        API + Vite with hot reload (default). Two ports.
  --prod       Build the studio and serve everything from the API. One port.
  --no-open    Do not open a browser.
  --api-port   Where the API listens (default 8000, or $OIG_PORT).
  --web-port   Where Vite listens in dev (default 5173, or $OIG_WEB_PORT).
  --stop       Stop anything this script started and exit.
USAGE
}

STOP_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --dev) MODE="dev"; shift ;;
    --prod) MODE="prod"; shift ;;
    --no-open) OPEN_BROWSER=0; shift ;;
    --api-port) API_PORT="$2"; shift 2 ;;
    --web-port) WEB_PORT="$2"; shift 2 ;;
    --stop) STOP_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
  esac
done

# ------------------------------------------------------------------ reporting
# One prefix per process, so a single terminal can carry both without the two
# streams becoming one unreadable thing.
BOLD=$'\033[1m'; DIM=$'\033[2m'; RESET=$'\033[0m'
BLUE=$'\033[34m'; GREEN=$'\033[32m'; RED=$'\033[31m'; YELLOW=$'\033[33m'
if [[ ! -t 1 ]]; then BOLD=""; DIM=""; RESET=""; BLUE=""; GREEN=""; RED=""; YELLOW=""; fi

say()  { printf '%s==>%s %s\n' "$BOLD" "$RESET" "$*"; }
warn() { printf '%s==>%s %s\n' "$YELLOW" "$RESET" "$*" >&2; }
die()  { printf '%s==>%s %s\n' "$RED" "$RESET" "$*" >&2; exit 1; }

# ------------------------------------------------------------------- stopping
port_pids() {
  # `lsof` is not everywhere; `ss` usually is, and either beats parsing ps.
  if command -v lsof >/dev/null 2>&1; then
    lsof -ti "tcp:$1" -sTCP:LISTEN 2>/dev/null
  elif command -v ss >/dev/null 2>&1; then
    ss -lptnH "sport = :$1" 2>/dev/null | grep -oP 'pid=\K[0-9]+' | sort -u
  fi
}

stop_port() {
  local port="$1" label="$2" pids
  pids="$(port_pids "$port" || true)"
  [[ -z "$pids" ]] && return 0

  say "stopping $label on :$port ($(echo "$pids" | tr '\n' ' '))"
  # SIGTERM first: the API unloads the model and closes the SQLite archive on
  # the way out, and killing it outright leaves the weights on the card.
  kill $pids 2>/dev/null || true
  for _ in $(seq 1 40); do
    sleep 0.5
    [[ -z "$(port_pids "$port" || true)" ]] && return 0
  done
  warn "$label did not stop cleanly; forcing"
  kill -9 $(port_pids "$port" || true) 2>/dev/null || true
  sleep 1
}

stop_all() {
  stop_port "$API_PORT" "the API"
  stop_port "$WEB_PORT" "the studio"
}

if [[ "$STOP_ONLY" == 1 ]]; then
  stop_all
  say "stopped."
  exit 0
fi

# --------------------------------------------------------------- requirements
command -v conda >/dev/null 2>&1 || die "conda is not on PATH; see the Setup section of the README"
[[ -f "$ROOT/frontend/package.json" ]] || die "frontend/ is missing; are you in the right checkout?"
if [[ ! -d "$ROOT/frontend/node_modules" ]]; then
  say "installing frontend dependencies (first run)"
  (cd "$ROOT/frontend" && npm install) || die "npm install failed"
fi

mkdir -p "$LOG_DIR"
API_LOG="$LOG_DIR/api.log"
WEB_LOG="$LOG_DIR/studio.log"

CHILDREN=()
shutdown() {
  trap - INT TERM EXIT
  echo
  say "shutting down"
  for pid in "${CHILDREN[@]:-}"; do
    [[ -n "${pid:-}" ]] && kill "$pid" 2>/dev/null || true
  done
  # The API is the one that has to land softly; give it the time to.
  for _ in $(seq 1 30); do
    sleep 0.5
    [[ -z "$(port_pids "$API_PORT" || true)" ]] && break
  done
  stop_all
  say "down."
}
trap shutdown INT TERM EXIT

# `sed` rather than `awk` so the prefix appears line by line as it arrives
# instead of when the buffer happens to flush.
prefix() { sed -u "s/^/$1/"; }

wait_for_http() {
  local url="$1" label="$2" tries="${3:-180}" log="${4:-}"
  for ((i = 0; i < tries; i++)); do
    if curl -fsS -m 2 -o /dev/null "$url" 2>/dev/null; then return 0; fi
    # A process that has already died is not going to start answering.
    if [[ -n "${5:-}" ]] && ! kill -0 "$5" 2>/dev/null; then
      warn "$label exited before it answered; last lines of $log:"
      [[ -n "$log" && -f "$log" ]] && tail -n 15 "$log" >&2
      return 1
    fi
    sleep 1
  done
  warn "$label did not answer $url within ${tries}s"
  [[ -n "$log" && -f "$log" ]] && tail -n 15 "$log" >&2
  return 1
}

open_browser() {
  local url="$1"
  if [[ "$OPEN_BROWSER" != 1 ]]; then
    say "ready at $url"
    return
  fi
  say "opening $url"
  for opener in xdg-open open sensible-browser; do
    if command -v "$opener" >/dev/null 2>&1; then
      "$opener" "$url" >/dev/null 2>&1 &
      return
    fi
  done
  say "no browser opener found; open $url yourself"
}

stop_all

# ---------------------------------------------------------------------- start
say "starting the API on :$API_PORT ${DIM}($MODE)${RESET}"
if [[ "$MODE" == "prod" ]]; then
  say "building the studio into app/static"
  (cd "$ROOT/frontend" && npm run build) || die "the studio failed to build"
else
  # The dev origin is a different port, so the API has to be told to allow it.
  export OIG_DEV=true
fi

OIG_PORT="$API_PORT" "$ROOT/scripts/serve.sh" > >(tee "$API_LOG" | prefix "$BLUE[api]$RESET ") 2>&1 &
API_PID=$!
CHILDREN+=("$API_PID")

# The API answers /healthz as soon as the HTTP layer is up and reports the
# model as "loading" until the weights land, which is the whole point: you can
# queue work, browse the catalog and reach a web model while it warms.
wait_for_http "http://127.0.0.1:$API_PORT/healthz" "the API" 180 "$API_LOG" "$API_PID" \
  || die "the API never came up; see $API_LOG"
say "${GREEN}API up${RESET} on http://localhost:$API_PORT ${DIM}(docs at /docs)${RESET}"

TARGET="http://localhost:$API_PORT"
if [[ "$MODE" == "dev" ]]; then
  say "starting the studio on :$WEB_PORT"
  (cd "$ROOT/frontend" && OIG_PORT="$API_PORT" npm run dev -- --port "$WEB_PORT" --strictPort) \
    > >(tee "$WEB_LOG" | prefix "$GREEN[studio]$RESET ") 2>&1 &
  WEB_PID=$!
  CHILDREN+=("$WEB_PID")

  wait_for_http "http://127.0.0.1:$WEB_PORT/" "the studio" 120 "$WEB_LOG" "$WEB_PID" \
    || die "the studio never came up; see $WEB_LOG"
  TARGET="http://localhost:$WEB_PORT"
  say "${GREEN}studio up${RESET} on $TARGET ${DIM}(hot reload, proxying /v1)${RESET}"
fi

# The model is still loading in the background here, and that is fine: the
# studio shows it warming rather than pretending to be idle.
open_browser "$TARGET"
say "logs in $LOG_DIR — ${BOLD}Ctrl-C stops both${RESET}"

# Hand the terminal to whichever process exits first, so a crash surfaces here
# rather than being discovered later in a log.
wait -n "${CHILDREN[@]}"
warn "one of the processes exited"
