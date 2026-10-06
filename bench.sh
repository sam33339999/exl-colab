#!/usr/bin/env bash
# Benchmark: Chinese (temp 0.7) + English/code (greedy), single & 8 concurrent. Run after warmup.
cd "$(dirname "$0")"
K=$(grep api_key tabbyAPI/api_tokens.yml | awk '{print $2}')
U=http://127.0.0.1:5000/v1/chat/completions
one(){ curl -s $U -H "Authorization: Bearer $K" -H "Content-Type: application/json" -d "{\"model\":\"x\",\"max_tokens\":300,\"temperature\":0,\"messages\":[{\"role\":\"user\",\"content\":\"$1\"}],\"template_vars\":{\"enable_thinking\":false}}" >/dev/null; grep -oE "[0-9.]+ T/s" server.log | tail -1; }
python3 test_concurrency.py 2 >/dev/null 2>&1  # warmup
echo "-- zh (temp 0.7) --"; python3 test_concurrency.py 8 2>&1 | grep total
echo "-- code single (greedy) --"; one "Write a Python quicksort function with comments."
echo "-- en single (greedy) --";   one "Explain how TCP handshake works in English."
nvidia-smi --query-gpu=memory.used --format=csv,noheader
