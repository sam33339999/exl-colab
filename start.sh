#!/usr/bin/env bash
# Quick start: TabbyAPI + ExLlamaV3. The model comes from .env
# (.env.coder390 or .env.swift). ENV_FILE selects one for a single launch.
#
# Usage: ./start.sh [--bg|--stop|--download|--help]
# Each start copies ./config.yml over tabbyAPI/config.yml, then applies $DRAFT.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TABBY="$ROOT/tabbyAPI"
PIDFILE="$ROOT/server.pid"
LOG="$ROOT/server.log"

# Model file. ENV_FILE overrides .env for this launch. A value already
# exported in the shell wins over the file.
load_dotenv() {
  local file line key value
  if [ -n "${ENV_FILE:-}" ]; then
    case "$ENV_FILE" in
      /*) file="$ENV_FILE" ;;
      *) file="$ROOT/$ENV_FILE" ;;
    esac
    if [ ! -f "$file" ]; then
      echo "[!] Env file not found: $file"
      echo "    Use .env, .env.coder390, or .env.swift"
      exit 1
    fi
  else
    file="$ROOT/.env"
    [ -f "$file" ] || return 0
  fi
  echo "[*] Env file: ${file#"$ROOT"/}"
  while IFS= read -r line || [ -n "$line" ]; do
    case "$line" in
      ''|\#*) continue ;;
    esac
    key=${line%%=*}
    value=${line#*=}
    if [ -z "${!key+x}" ]; then
      export "$key=$value"
    fi
  done < "$file"
}

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

Environment (model defaults live in .env; an already-exported value wins):
  ENV_FILE
      Model file for this launch. A relative path is from the repo root.
      Unset loads .env, which matches .env.coder390 until you copy over it.
      .env.coder390   Qwen3.8-27B-Coder390-MTP-Exl3-3.5bpw
      .env.swift      Swift-1.5-Qwen3.8-27b-exl3-3.5bpw
  MODEL_NAME
      Model id other providers register. This is the /v1/models id and the
      weights folder name. Printed again after the model finishes loading.
  MODEL_REPO
      Hugging Face repo downloaded into models/\$MODEL_NAME.
  DRAFT=mtp|dflash2|off
      Override draft_mode after the config copy.
      mtp uses the MTP head inside the checkpoint (draft_num_tokens 4).
      dflash2 downloads DRAFT_REPO from the selected env file and loads it
      (draft_mode model, draft_num_tokens 7). MTP and DFlash2 are not combined.
      Unset keeps the value in config.yml (currently mtp).

Examples:
  ./start.sh --bg
  ENV_FILE=.env.swift ./start.sh --bg
  ENV_FILE=.env.coder390 ./start.sh --bg
  DRAFT=off ./start.sh --bg
  DRAFT=dflash2 ./start.sh --bg
  ./start.sh --stop
EOF
}

download_repo() {
  local repo="$1"
  local dest="$2"
  echo "[*] Downloading $repo ..."
  uvx --from huggingface_hub hf download "$repo" --local-dir "$dest"
}

download_model() {
  download_repo "$MODEL_REPO" "$MODEL_DIR"
}

ensure_draft() {
  ls "$DRAFT_DIR"/*.safetensors >/dev/null 2>&1 || download_repo "$DRAFT_REPO" "$DRAFT_DIR"
}

sync_config() {
  if [ ! -f "$ROOT/config.yml" ]; then
    echo "[!] Missing $ROOT/config.yml"
    exit 1
  fi
  cp "$ROOT/config.yml" "$TABBY/config.yml"
  # Keep the running config on the provider-facing name from the environment.
  local esc_model esc_draft
  esc_model=$(printf '%s' "$MODEL_NAME" | sed 's/[&|\\]/\\&/g')
  esc_draft=$(printf '%s' "$DRAFT_NAME" | sed 's/[&|\\]/\\&/g')
  sed -i "s|^  model_name: .*|  model_name: ${esc_model}|" "$TABBY/config.yml"
  sed -i "s|^  draft_model_name: .*|  draft_model_name: ${esc_draft}|" "$TABBY/config.yml"
  echo "[*] Applied config.yml -> tabbyAPI/config.yml"
  echo "[*] Provider model name: ${MODEL_NAME}"
}

# Print the id a non-admin client sees on /v1/models. That is the string
# other providers register. Falls back to MODEL_NAME if the API is not up.
# The same lines are appended to server.log, next to the startup keys.
print_provider_model() {
  local key ids
  key=$(awk '/^api_key:/ {print $2; exit}' "$TABBY/api_tokens.yml" 2>/dev/null || true)
  ids=""
  if [ -n "$key" ]; then
    ids=$(curl -sf http://127.0.0.1:5000/v1/models \
      -H "Authorization: Bearer $key" \
      | python3 -c 'import json,sys
try:
    data=json.load(sys.stdin).get("data") or []
except Exception:
    data=[]
print("\n".join(item.get("id","") for item in data if item.get("id")))' 2>/dev/null || true)
  fi
  {
    if [ -n "$ids" ]; then
      echo "[*] Loaded model name (register this with providers):"
      printf '%s\n' "$ids" | sed 's/^/    /'
    else
      echo "[*] Loaded model name (register this with providers): ${MODEL_NAME}"
    fi
    echo "[*] Hugging Face: ${MODEL_REPO}"
  } | tee -a "$LOG"
}

# An empty venv still has an executable python. Require the runtime
# packages, or a half-finished install is treated as ready and main.py
# dies on the first import. find_spec does not import torch.
deps_ready() {
  [ -x "$TABBY/.venv/bin/python" ] || return 1
  "$TABBY/.venv/bin/python" -c 'import importlib.util, sys
sys.exit(0 if all(importlib.util.find_spec(n) for n in ("loguru", "torch", "exllamav3")) else 1)'
}

ensure_setup() {
  command -v uv >/dev/null || { echo "[*] Installing uv"; curl -LsSf https://astral.sh/uv/install.sh | sh; export PATH="$HOME/.local/bin:$PATH"; }
  if [ ! -d "$TABBY" ]; then
    echo "[*] Cloning TabbyAPI..."
    git clone https://github.com/theroyallab/tabbyAPI.git "$TABBY"
  fi
  if ! deps_ready; then
    echo "[*] Installing TabbyAPI/ExLlamaV3 (cu13)"
    if [ ! -x "$TABBY/.venv/bin/python" ]; then
      (cd "$TABBY" && uv venv --python 3.13 .venv)
    fi
    (cd "$TABBY" && uv pip install --python .venv/bin/python -e ".[cu13]")
  fi
  ls "$MODEL_DIR"/*.safetensors >/dev/null 2>&1 || download_model
}

case "${1:-}" in
  --help|-h|help)
    usage; exit 0 ;;
  --stop)
    exec "$ROOT/stop.sh" ;;
esac

load_dotenv

export MODEL_REPO="${MODEL_REPO:-sam33339999/Qwen3.8-27B-Coder390-MTP-Exl3-3.5bpw}"
# Id returned by /v1/models. Other providers register this exact string.
export MODEL_NAME="${MODEL_NAME:-Qwen3.8-27B-Coder390-MTP-Exl3-3.5bpw}"
export MODEL_DIR="${MODEL_DIR:-$ROOT/models/${MODEL_NAME}}"
export DRAFT_REPO="${DRAFT_REPO:-sam33339999/Qwen3.8-27B-Coder390-dflash2}"
export DRAFT_NAME="${DRAFT_NAME:-Qwen3.8-27B-Coder390-dflash2}"
export DRAFT_DIR="${DRAFT_DIR:-$ROOT/models/${DRAFT_NAME}}"
# TabbyAPI applies TABBY_<section>_<field> on top of config.yml.
export TABBY_MODEL_MODEL_NAME="$MODEL_NAME"
export TABBY_DRAFT_MODEL_DRAFT_MODEL_NAME="$DRAFT_NAME"

case "${1:-}" in
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
# MTP reads the head inside the main checkpoint. DFlash2 is a separate model.
case "${DRAFT:-}" in
  dflash2)
    ensure_draft
    sed -i 's/^  draft_mode: .*/  draft_mode: model/' config.yml
    sed -i 's/^  draft_num_tokens: .*/  draft_num_tokens: 7/' config.yml
    ;;
  mtp)
    sed -i 's/^  draft_mode: .*/  draft_mode: mtp/' config.yml
    sed -i 's/^  draft_num_tokens: .*/  draft_num_tokens: 4/' config.yml
    ;;
  off)
    sed -i 's/^  draft_mode: .*/  draft_mode: disabled/' config.yml
    ;;
  "") ;;
  *) echo "[!] DRAFT must be dflash2|mtp|off"; exit 1 ;;
esac
echo "[*] draft_mode: $(grep -oP '^  draft_mode: \K\S+' config.yml)  draft_num_tokens: $(grep -oP '^  draft_num_tokens: \K\S+' config.yml)"
echo "[*] Loading model: ${MODEL_NAME}"

if [ "${1:-}" = "--bg" ]; then
  {
    echo "[*] Hugging Face: ${MODEL_REPO}"
    echo "[*] Loading model: ${MODEL_NAME}"
  } >"$LOG"
  setsid nohup .venv/bin/python main.py >>"$LOG" 2>&1 < /dev/null &
  echo $! >"$PIDFILE"
  echo "[*] Server starting in background (PID $(cat "$PIDFILE")), log: $LOG"
  echo -n "[*] Waiting for model to load"
  for _ in $(seq 1 180); do
    if curl -sf http://127.0.0.1:5000/health >/dev/null 2>&1; then
      echo; echo "[✓] Ready at http://0.0.0.0:5000/v1"
      print_provider_model
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
