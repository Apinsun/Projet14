#!/usr/bin/env python3
"""Comparaison few-shot vs zero-shot, avant/après SFT (gold français).

Utilise les métriques sauvegardées (chaque fichier reflète le bon parser au moment du run).
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS = Path("data/processed/gold/results")
FIG_DIR = Path("reports/figures")

# (label, fichier, note)
CONFIGS = [
    ("Base · zero-shot", "Qwen_Qwen3-1.7B_20260914_122816.json"),
    ("Base · few-shot", "Qwen_Qwen3-1.7B_20260914_121454.json"),
    ("SFT · few-shot", "models_lora_stage2_merged_20260914_162148.json"),
    ("SFT · zero-shot", "models_lora_stage3_merged_20260914_170220.json"),
]


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    for label, fname in CONFIGS:
        m = json.loads((RESULTS / fname).read_text(encoding="utf-8"))["metrics"]
        n = m["n_total"]
        rows.append({
            "label": label,
            "parse": m["n_parsed"] / n * 100,
            "exact": (m["accuracy"] or 0) * 100,
            "under": (m["under_triage_rate"] or 0) * 100,
            "over": (m["over_triage_rate"] or 0) * 100,
        })

    labels = [r["label"] for r in rows]
    x = np.arange(len(rows))
    w = 0.35

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    b1 = ax.bar(x - w / 2, [r["parse"] for r in rows], w, label="Fiches parsées (%)", color="#4472c4")
    b2 = ax.bar(x + w / 2, [r["exact"] for r in rows], w, label="Exactitude (%)", color="#ed7d31")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=8)
    ax.set_ylabel("%")
    ax.set_title("Parse & exactitude — few-shot vs zero-shot, avant/après SFT (gold FR)")
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.0f", padding=2, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "compare_parse_exact.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    b = ax.bar(x, [r["under"] for r in rows], 0.5, color="#c00000")
    ax.bar_label(b, fmt="%.0f", padding=2, fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=8)
    ax.set_ylabel("%")
    ax.set_title("Sous-triage (l'erreur dangereuse) — few-shot vs zero-shot")
    ax.set_ylim(0, 60)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "compare_undertriage.png", dpi=150)
    plt.close(fig)

    print(f"{'config':18s} {'parse':>6s} {'exact':>6s} {'under':>6s} {'over':>6s}")
    print("-" * 46)
    for r in rows:
        print(f"{r['label']:18s} {r['parse']:5.0f}% {r['exact']:5.0f}% {r['under']:5.0f}% {r['over']:5.0f}%")
    print(f"\nGraphiques → {FIG_DIR}/")


if __name__ == "__main__":
    main()
