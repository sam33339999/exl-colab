#!/usr/bin/env bash
# Stop TabbyAPI server gracefully and wait for GPU VRAM release
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PIDFILE="$ROOT/server.pid"

find_tabby_pids() {
  pgrep -f "tabbyAPI.*main.py" || true
}

PIDS=""
if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  if [ -n "$PID" ] && kill -0 "$PID" 2>/dev/null; then
    PIDS="$PID"
  fi
fi

if [ -z "$PIDS" ]; then
  PIDS="$(find_tabby_pids)"
fi

if [ -z "$PIDS" ]; then
  echo "[!] TabbyAPI is not running."
  rm -f "$PIDFILE"
  exit 0
fi

echo "[*] Stopping TabbyAPI (PID: $PIDS)..."
for pid in $PIDS; do
  kill "$pid" 2>/dev/null || true
done

# Wait up to 15 seconds for graceful shutdown (Tabby unloads model cleanly)
echo -n "[*] Waiting for model to unload"
for _ in $(seq 1 15); do
  STILL_ALIVE=0
  for pid in $PIDS; do
    if kill -0 "$pid" 2>/dev/null; then
      STILL_ALIVE=1
      break
    fi
  done
  if [ "$STILL_ALIVE" -eq 0 ]; then
    break
  fi
  echo -n "."
  sleep 1
done
echo ""

# Force kill if still running
for pid in $PIDS; do
  if kill -0 "$pid" 2>/dev/null; then
    echo "[!] Process did not exit gracefully, force killing PID $pid..."
    kill -9 "$pid" 2>/dev/null || true
  fi
done

rm -f "$PIDFILE"
echo "[✓] TabbyAPI stopped successfully and GPU VRAM released."
