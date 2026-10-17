#!/usr/bin/env bash
# Serve /home/sam/MiniCPM-V-4.7-35B-A3B-abliterated-EXL3-3.5bpw through TabbyAPI
# on this RTX 3090. The one-time install is in that directory's RUN.md.
# The sm_86 extension build this launcher depends on is docs/3090-extension-build.md.
# This script refuses to let start.sh create a venv: start.sh's installer
# would pull the upstream exllamav3 1.6.1 wheel, which does not contain
# MiniCPMV4_7ForConditionalGeneration.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TABBY="$ROOT/tabbyAPI"
ORIGIN="/home/sam/exllamav3-origin"

if [ "${1:-}" = "--stop" ]; then
  exec "$ROOT/start.sh" --stop
fi

export CUDA_HOME=/usr/local/cuda-13.3
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST=8.6
export MAX_JOBS=4

if [ ! -x "$TABBY/.venv/bin/python" ]; then
  echo "[!] $TABBY/.venv is not ready."
  echo "    Follow the TabbyAPI section of:"
  echo "    /home/sam/MiniCPM-V-4.7-35B-A3B-abliterated-EXL3-3.5bpw/RUN.md"
  echo "    Do not run ./start.sh yet. Its first install uses .[cu13]."
  exit 1
fi

"$TABBY/.venv/bin/python" - <<PY
import importlib.metadata as metadata
import importlib.util
import json
import sys
from pathlib import Path

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"[!] Tabby venv is Python {sys.version.split()[0]}, expected 3.12.")

torch_version = metadata.version("torch")
if torch_version != "2.14.1+cu130":
    raise SystemExit(
        f"[!] torch is {torch_version}, expected 2.14.1+cu130. "
        "The existing sm_86 extension was built against that build."
    )

# An editable install's locate_file() points at a dist-info path that is not
# the checkout. direct_url.json is the record of where the code is loaded from.
direct = json.loads(metadata.distribution("exllamav3").read_text("direct_url.json"))
installed = Path(direct["url"].removeprefix("file://")).resolve()
expected = Path("$ORIGIN").resolve()
editable = bool((direct.get("dir_info") or {}).get("editable"))
if not editable or installed != expected:
    raise SystemExit(
        "[!] exllamav3 is not the local feat/minicpm-v4.7 checkout.\n"
        f"    installed: {installed} editable={editable}\n"
        f"    expected:  {expected}"
    )

if importlib.util.find_spec("minicpm47_exl3.exl3") is None:
    raise SystemExit(
        "[!] minicpm47_exl3 is not installed in the Tabby venv. "
        "Install the native stub from RUN.md. Do not download the 4bpw adapter wheel."
    )
PY

export ENV_FILE=.env.minicpm-3090
exec "$ROOT/start.sh" "$@"
