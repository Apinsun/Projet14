#!/usr/bin/env python3
"""Benchmark de latence / robustesse de l'agent de triage (vLLM + FastAPI).

Mesures en conditions réalistes (température 0.6, max_tokens 1024, comme en prod) :
  1. TTFT (temps au 1er token) + tokens/s via streaming vLLM — prompt court vs historique long ;
  2. latence de bout en bout via l'API FastAPI (/chat) ;
  3. tenue en concurrence (N requêtes simultanées).

Usage :
    poetry run python scripts/benchmark_latency.py                  # scénarios + concurrence 4
    poetry run python scripts/benchmark_latency.py --n 5 --concurrency 8
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from concurrent.futures import ThreadPoolExecutor

import requests

from triage_agent.serving.config import SYSTEM_PROMPT

VLLM_URL = "http://localhost:8000/v1"
APP_URL = "http://localhost:8080"
MODEL = "models/lora_dpo_v3_merged"
TEMP = 0.6
MAX_TOKENS = 1024

COURT = [{"role": "user", "content": "bonjour, j'ai très mal au ventre depuis ce matin"}]

LONG = [
    {"role": "user", "content": "bonjour, j'ai très mal au ventre depuis ce matin, ça fait anormalement mal"},
    {"role": "assistant", "content": "Je comprends. Depuis combien de temps exactement ressentez-vous cette douleur ?"},
    {"role": "user", "content": "depuis ce matin, vers 7h, ça ne passe pas"},
    {"role": "assistant", "content": "Sur une échelle de 0 à 10, comment évalueriez-vous votre douleur ?"},
    {"role": "user", "content": "je dirais 7, c'est vraiment douloureux"},
    {"role": "assistant", "content": "Avez-vous des antécédents médicaux ou prenez-vous des médicaments ?"},
    {"role": "user", "content": "non rien de particulier, je ne prends pas de médicament"},
]


def _payload(messages: list[dict], stream: bool) -> dict:
    p = {
        "model": MODEL,
        "messages": [{"role": "system", "content": SYSTEM_PROMPT}, *messages],
        "temperature": TEMP,
        "top_p": 0.95,
        "top_k": 20,
        "max_tokens": MAX_TOKENS,
    }
    if stream:
        p["stream"] = True
        p["stream_options"] = {"include_usage": True}
    return p


def stream_measure(messages: list[dict]) -> dict:
    """TTFT + tokens/s via streaming vLLM (mesure niveau modèle)."""
    t0 = time.time()
    ttft = None
    n_tokens = 0
    with requests.post(
        f"{VLLM_URL}/chat/completions", json=_payload(messages, stream=True), stream=True, timeout=300
    ) as r:
        r.raise_for_status()
        for raw in r.iter_lines():
            if not raw:
                continue
            line = raw.decode("utf-8", errors="replace")
            if not line.startswith("data: "):
                continue
            data = line[6:].strip()
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            usage = chunk.get("usage")
            if usage:
                n_tokens = usage.get("completion_tokens", n_tokens)
            choices = chunk.get("choices") or []
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content and ttft is None:
                ttft = time.time() - t0
    total = time.time() - t0
    return {
        "ttft_s": round(ttft, 3) if ttft else None,
        "total_s": round(total, 3),
        "n_tokens": n_tokens,
        "tokens_s": round(n_tokens / total, 1) if total else 0,
    }


def e2e_measure(messages: list[dict], conversation_id: str) -> float:
    """Latence totale via l'API FastAPI (mesure niveau système)."""
    t0 = time.time()
    r = requests.post(
        f"{APP_URL}/chat",
        json={"messages": messages, "conversation_id": conversation_id},
        timeout=300,
    )
    r.raise_for_status()
    return time.time() - t0


def _pct(xs: list[float], p: float) -> float:
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    f = int(k)
    c = f + 1 if f + 1 < len(xs) else f
    return xs[f] + (xs[c] - xs[f]) * (k - f)


def concurrency_measure(messages: list[dict], n: int) -> dict:
    """n requêtes simultanées via /chat → latences + débit."""

    def one(i: int) -> float:
        return e2e_measure(messages, f"bench-{i}")

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=n) as ex:
        latencies = list(ex.map(one, range(n)))
    wall = time.time() - t0
    return {
        "n": n,
        "wall_s": round(wall, 2),
        "avg_s": round(statistics.mean(latencies), 2),
        "p50_s": round(_pct(latencies, 0.5), 2),
        "p95_s": round(_pct(latencies, 0.95), 2),
        "max_s": round(max(latencies), 2),
        "throughput_req_s": round(n / wall, 2),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="Benchmark de latence de l'agent de triage")
    ap.add_argument("--n", type=int, default=3, help="répétitions par scénario")
    ap.add_argument("--concurrency", type=int, default=4, help="concurrence max à tester")
    args = ap.parse_args()

    print("=" * 74)
    print("Benchmark latence — Qwen3-1.7B (bf16 r=32 + DPO), temp 0.6, max_tokens 1024")
    print("=" * 74)

    for name, msgs in [("court (1 tour)", COURT), ("long (7 tours)", LONG)]:
        print(f"\n### Scénario {name}")
        ttfts, toks = [], []
        for _ in range(args.n):
            m = stream_measure(msgs)
            ttfts.append(m["ttft_s"])
            toks.append(m["tokens_s"])
        print(f"  TTFT        : {statistics.mean(ttfts):.2f} s (min {min(ttfts):.2f})")
        print(f"  tokens/s    : {statistics.mean(toks):.1f}")
        e2e = [e2e_measure(msgs, f"bench-{name}-{i}") for i in range(args.n)]
        print(f"  latence API : {statistics.mean(e2e):.2f} s (moyenne sur {args.n} runs)")

    print("\n### Concurrence (scénario court)")
    for n in [1, 2, 4, args.concurrency]:
        c = concurrency_measure(COURT, n)
        print(f"  {n:>2} simultané(s) : wall {c['wall_s']}s | avg {c['avg_s']}s | "
              f"p95 {c['p95_s']}s | débit {c['throughput_req_s']} req/s")

    print("\nTerminé.")


if __name__ == "__main__":
    main()
