#!/usr/bin/env python3
"""Lance l'évaluation du triage sur les jeux gold préparés.

Usage :
    poetry run python scripts/run_gold_eval.py --model Leila_fast:latest --limit 10
"""

from __future__ import annotations

import argparse
import datetime
import json
from pathlib import Path

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.eval.gold import load_few_shot_examples
from triage_agent.eval.harness import evaluate, ollama_runner, openai_compat_runner, report


def main() -> None:
    ap = argparse.ArgumentParser(description="Évalue le triage sur les jeux gold")
    ap.add_argument("--model", default="Leila_fast:latest")
    ap.add_argument("--backend", choices=["ollama", "openai"], default="ollama",
                    help="openai = API OpenAI-compatible (vLLM)")
    ap.add_argument("--api-url", default="http://localhost:8000/v1",
                    help="URL de base de l'API (backend openai)")
    ap.add_argument("--limit", type=int, default=0, help="Limiter à N cas (0 = tous)")
    ap.add_argument("--files", nargs="*", help="Fichiers gold (défaut : tout data/processed/gold/*.jsonl)")
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--few-shot", action="store_true", help="Préfixer 2 exemples (niveaux 1 et 5)")
    ap.add_argument("--out", help="Chemin du fichier de sortie JSON (défaut : auto)")
    args = ap.parse_args()

    gold_dir = PROCESSED_DIR / "gold"
    files = [Path(f) for f in args.files] if args.files else sorted(gold_dir.glob("*.jsonl"))
    records = []
    for f in files:
        records.extend(json.loads(line) for line in f.open(encoding="utf-8"))
    if args.limit:
        records = records[: args.limit]

    few_shot = load_few_shot_examples() if args.few_shot else []
    if few_shot:
        for rec in records:
            rec["messages"] = rec["messages"][:1] + few_shot + rec["messages"][1:]

    if args.backend == "openai":
        runner = openai_compat_runner(
            args.api_url, args.model, args.temperature, args.max_tokens,
            args.top_p, args.top_k, args.seed,
        )
    else:
        runner = ollama_runner(args.model, args.temperature)

    metrics = evaluate(records, runner)
    print(report(metrics))
    print(f"\nModèle : {args.model} | {len(records)} cas | few-shot : {bool(few_shot)}")

    # Sauvegarde des résultats complets (métriques + détail par cas + outputs bruts).
    model_slug = args.model.replace("/", "_")
    out_path = Path(args.out) if args.out else (
        PROCESSED_DIR / "gold" / "results"
        / f"{model_slug}_{datetime.datetime.now():%Y%m%d_%H%M%S}.json"
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": {
            "model": args.model, "backend": args.backend, "temperature": args.temperature,
            "max_tokens": args.max_tokens, "few_shot": bool(few_shot), "n_cases": len(records),
        },
        "metrics": {k: v for k, v in metrics.items() if k != "per_case"},
        "per_case": metrics["per_case"],
    }
    with out_path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)
    print(f"Résultats : {out_path}")


if __name__ == "__main__":
    main()
