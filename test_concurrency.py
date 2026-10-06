#!/usr/bin/env python3
"""Concurrency smoke test for the TabbyAPI server. Usage: python3 test_concurrency.py [N]"""
import json, sys, time, urllib.request, concurrent.futures as cf, re, pathlib

N = int(sys.argv[1]) if len(sys.argv) > 1 else 6
TOK = pathlib.Path(__file__).parent / "tabbyAPI" / "api_tokens.yml"
KEY = re.search(r"api_key:\s*(\S+)", TOK.read_text()).group(1)
URL = "http://127.0.0.1:5000/v1/chat/completions"
CITIES = ["台北", "台中", "高雄", "台南", "新竹", "花蓮", "基隆", "嘉義", "宜蘭", "屏東", "桃園", "台東"]

def req(i):
    body = {"model": "x", "max_tokens": 256, "temperature": 0.7,
            "messages": [{"role": "user", "content": f"用三句話介紹{CITIES[i % len(CITIES)]}。"}],
            "template_vars": {"enable_thinking": False}}
    r = urllib.request.Request(URL, json.dumps(body).encode(),
                               {"Authorization": f"Bearer {KEY}", "Content-Type": "application/json"})
    t = time.time()
    d = json.load(urllib.request.urlopen(r, timeout=600))
    dt = time.time() - t
    n = d["usage"]["completion_tokens"]
    txt = (d["choices"][0]["message"].get("content") or "").replace("\n", " ")
    print(f"  [{i}] {n} tok in {dt:.1f}s ({n/dt:.1f} tok/s) | {txt[:50]}")
    return n

for label, n in (("single", 1), (f"{N} concurrent", N)):
    print(f"== {label} ==")
    t = time.time()
    with cf.ThreadPoolExecutor(n) as ex:
        total = sum(ex.map(req, range(n)))
    dt = time.time() - t
    print(f"  total {total} tok in {dt:.1f}s -> aggregate {total/dt:.1f} tok/s\n")
