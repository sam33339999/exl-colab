#!/usr/bin/env bash
# Quick start: TabbyAPI + ExLlamaV3 serving Swift-1.5-Qwen3.8-27b (EXL3 3.5bpw)
#
# Usage: ./start.sh [--bg|--stop|--download|--help]
# Each start copies ./config.yml over tabbyAPI/config.yml, then applies $DRAFT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TABBY="$ROOT/tabbyAPI"
MODEL_REPO="sam33339999/Swift-1.5-Qwen3.8-27b-Uncensored-exl3-3.5bpw"
MODEL_DIR="$ROOT/models/Swift-1.5-Qwen3.8-27b-exl3-3.5bpw"
PIDFILE="$ROOT/server.pid"
LOG="$ROOT/server.log"

usage() {
  cat <<EOF
Usage: ./start.sh [option]

One-shot start. Copies $ROOT/config.yml onto tabbyAPI/config.yml on every
launch, then starts TabbyAPI. Edits belong in the repo config.yml.

Options:
  (none)        Run in the foreground
  --bg          Run in the background and wait until /health is ready
  --stop        Stop the background server
  --download    Download the model only
  --help, -h    Show this help

Environment:
  DRAFT=mtp|dflash2|off
      Override draft_mode after the config copy.
      Unset keeps the value in config.yml (currently mtp).

Examples:
  ./start.sh --bg
  DRAFT=off ./start.sh --bg
  ./start.sh --stop
EOF
}

download_model() {
  echo "[*] Downloading $MODEL_REPO ..."
  uvx --from huggingface_hub hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
}

sync_config() {
  if [ ! -f "$ROOT/config.yml" ]; then
    echo "[!] Missing $ROOT/config.yml"
    exit 1
  fi
  cp "$ROOT/config.yml" "$TABBY/config.yml"
  echo "[*] Applied config.yml -> tabbyAPI/config.yml"
}

ensure_setup() {
  command -v uv >/dev/null || { echo "[*] Installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  if [ ! -d "$TABBY" ]; then
    echo "[*] Cloning TabbyAPI..."
    git clone https://github.com/theroyallab/tabbyAPI.git "$TABBY"
  fi
  if [ ! -x "$TABBY/.venv/bin/python" ]; then
    echo "[*] Creating venv + installing TabbyAPI/ExLlamaV3 (cu13)"
    (cd "$TABBY" && uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -e ".[cu13]")
  fi
  ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || download_model
}

case "${1:-}" in
  --help|-h|help)
    usage; exit 0 ;;
  --stop)
    exec "$ROOT/stop.sh" ;;
  --download)
    download_model; exit 0 ;;
  --bg|"")
    ;;
  *)
    echo "[!] Unknown option: $1"
    usage
    exit 1
    ;;
esac

# The server's argv is ".venv/bin/python main.py" (cwd is tabbyAPI), so a
# pattern that requires the directory name never sees it.
if pgrep -f '[.]venv/bin/python main.py' >/dev/null 2>&1; then
  echo "[!] TabbyAPI is already running. Stop it first: ./start.sh --stop"
  exit 1
fi

ensure_setup
sync_config
cd "$TABBY"

# Speculative decoding mode: DRAFT=dflash2 | mtp | off  (unset = keep config.yml as is)
case "${DRAFT:-}" in
  dflash2) sed -i 's/^  draft_mode: .*/  draft_mode: model/' config.yml ;;
  mtp)     sed -i 's/^  draft_mode: .*/  draft_mode: mtp/' config.yml ;;
  off)     sed -i 's/^  draft_mode: .*/  draft_mode: disabled/' config.yml ;;
  "") ;;
  *) echo "[!] DRAFT must be dflash2|mtp|off"; exit 1 ;;
esac
echo "[*] draft_mode: $(grep -oP '^  draft_mode: \K\S+' config.yml)"

if [ "${1:-}" = "--bg" ]; then
  setsid nohup .venv/bin/python main.py >"$LOG" 2>&1 < /dev/null &
  echo $! >"$PIDFILE"
  echo "[*] Server starting in background (PID $(cat "$PIDFILE")), log: $LOG"
  echo -n "[*] Waiting for model to load"
  for _ in $(seq 1 180); do
    if curl -sf http://127.0.0.1:5000/health >/dev/null 2>&1; then
      echo; echo "[✓] Ready at http://0.0.0.0:5000/v1"
      grep -E "api_key|admin_key" "$TABBY/api_tokens.yml" 2>/dev/null | sed 's/^/    /'
      exit 0
    fi
    kill -0 "$(cat "$PIDFILE")" 2>/dev/null || { echo; echo "[!] Server died, see $LOG"; tail -30 "$LOG"; exit 1; }
    echo -n "."; sleep 2
  done
  echo; echo "[!] Timed out waiting, check $LOG"; exit 1
else
  exec .venv/bin/python main.py
fi
