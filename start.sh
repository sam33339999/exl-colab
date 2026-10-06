#!/usr/bin/env bash
# Quick start: TabbyAPI + ExLlamaV3 serving Swift-1.5-Qwen3.8-27b (EXL3 3.5bpw)
#
# Usage:
#   ./start.sh              # run in foreground
#   ./start.sh --bg         # run in background (log: server.log)
#   ./start.sh --stop       # stop background server
#   ./start.sh --download   # (re)download model only
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TABBY="$ROOT/tabbyAPI"
MODEL_REPO="sam33339999/Swift-1.5-Qwen3.8-27b-Uncensored-exl3-3.5bpw"
MODEL_DIR="$ROOT/models/Swift-1.5-Qwen3.8-27b-exl3-3.5bpw"
PIDFILE="$ROOT/server.pid"
LOG="$ROOT/server.log"

download_model() {
  echo "[*] Downloading $MODEL_REPO ..."
  uvx --from huggingface_hub hf download "$MODEL_REPO" --local-dir "$MODEL_DIR"
}

ensure_setup() {
  command -v uv >/dev/null || { echo "[*] Installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  if [ ! -d "$TABBY" ]; then
    echo "[*] Cloning TabbyAPI..."
    git clone https://github.com/theroyallab/tabbyAPI.git "$TABBY"
  fi
  if [ ! -f "$TABBY/config.yml" ] && [ -f "$ROOT/config.yml" ]; then
    cp "$ROOT/config.yml" "$TABBY/config.yml"
  fi
  if [ ! -x "$TABBY/.venv/bin/python" ]; then
    echo "[*] Creating venv + installing TabbyAPI/ExLlamaV3 (cu13)"
    (cd "$TABBY" && uv venv --python 3.13 .venv && uv pip install --python .venv/bin/python -e ".[cu13]")
  fi
  ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || download_model
}

case "${1:-}" in
  --stop)
    exec "$ROOT/stop.sh" ;;
  --download)
    download_model; exit 0 ;;
esac

ensure_setup
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
