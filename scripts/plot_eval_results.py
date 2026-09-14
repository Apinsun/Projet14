#!/usr/bin/env python3
"""Génère les graphiques d'évaluation des jeux gold (parse, exactitude, sous/sur-triage,
matrice de confusion, distribution des priorités).

Usage :
    poetry run python scripts/plot_eval_results.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from triage_agent.data.config import PROCESSED_DIR

RESULTS_DIR = PROCESSED_DIR / "gold" / "results"
FIG_DIR = Path("reports/figures")

plt.rcParams.update({"font.size": 11, "axes.grid": True, "grid.alpha": 0.3})


def load_results() -> list[dict]:
    out = []
    for f in sorted(RESULTS_DIR.glob("*.json")):
        out.append(json.loads(f.read_text(encoding="utf-8")))
    return out


def method_label(config: dict) -> str:
    fs = "few-shot" if config.get("few_shot") else "zero-shot"
    return f"{fs}\nT={config.get('temperature')} · {config.get('max_tokens')} tok"


def _val(metrics: dict, key: str, default: float = 0.0) -> float:
    v = metrics.get(key)
    return default if v is None else v


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    results = load_results()
    if not results:
        print("Aucun résultat dans", RESULTS_DIR)
        return

    # Tri par progression (nb de parses croissant).
    results.sort(key=lambda d: d["metrics"].get("n_parsed", 0))
    labels = [method_label(d["config"]) for d in results]
    models = {d["config"]["model"] for d in results}

    parse_rate = [d["metrics"]["n_parsed"] / d["metrics"]["n_total"] * 100 for d in results]
    accuracy = [_val(d["metrics"], "accuracy") * 100 for d in results]
    under = [_val(d["metrics"], "under_triage_rate") * 100 for d in results]
    over = [_val(d["metrics"], "over_triage_rate") * 100 for d in results]

    x = np.arange(len(results))
    width = 0.38

    # --- Figure 1 : parse + exactitude ---
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - width / 2, parse_rate, width, label="Fiches parsées (%)", color="#4472c4")
    b2 = ax.bar(x + width / 2, accuracy, width, label="Exactitude (%)", color="#ed7d31")
    ax.set_ylabel("%")
    ax.set_title(f"Triage sur jeux gold — {', '.join(sorted(models))}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.0f", padding=2, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eval_parse_exactitude.png", dpi=150)
    plt.close(fig)

    # --- Figure 2 : sous-triage vs sur-triage ---
    fig, ax = plt.subplots(figsize=(7.5, 4.2))
    b1 = ax.bar(x - width / 2, under, width, label="Sous-triage (%)", color="#c00000")
    b2 = ax.bar(x + width / 2, over, width, label="Sur-triage (%)", color="#70ad47")
    ax.set_ylabel("%")
    ax.set_title("Erreurs de triage — le sous-triage est l'erreur dangereuse")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    for b in (b1, b2):
        ax.bar_label(b, fmt="%.0f", padding=2, fontsize=9)
    ax.set_ylim(0, 105)
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "eval_sous_surtriage.png", dpi=150)
    plt.close(fig)

    # --- Figure 3 : matrice de confusion (config la plus parsée) ---
    best = max(results, key=lambda d: d["metrics"].get("n_parsed", 0))
    parsed = [c for c in best["per_case"] if "priority" in c]
    if parsed:
        gold_labels = sorted({c["gold_label"] for c in parsed})
        prios = list(range(1, 6))
        mat = np.zeros((len(gold_labels), len(prios)), dtype=int)
        for c in parsed:
            mat[gold_labels.index(c["gold_label"]), c["priority"] - 1] += 1

        fig, ax = plt.subplots(figsize=(6.5, 4.5))
        im = ax.imshow(mat, cmap="Blues")
        ax.set_xticks(range(len(prios)))
        ax.set_xticklabels(prios)
        ax.set_yticks(range(len(gold_labels)))
        ax.set_yticklabels(gold_labels)
        ax.set_xlabel("Priorité prédite")
        ax.set_ylabel("Gold (label d'origine)")
        ax.set_title(f"Matrice de confusion — {method_label(best['config']).replace(chr(10), ' ')}")
        for i in range(len(gold_labels)):
            for j in range(len(prios)):
                if mat[i, j]:
                    ax.text(j, i, mat[i, j], ha="center", va="center",
                            color="white" if mat[i, j] > mat.max() / 2 else "black", fontsize=10)
        fig.colorbar(im, ax=ax, label="nb cas")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "eval_confusion.png", dpi=150)
        plt.close(fig)

        # --- Figure 4 : distribution des priorités prédites ---
        dist = Counter(c["priority"] for c in parsed)
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.bar([str(p) for p in prios], [dist.get(p, 0) for p in prios], color="#4472c4")
        ax.set_xlabel("Priorité prédite")
        ax.set_ylabel("Nombre de cas")
        ax.set_title("Distribution des priorités prédites (le modèle se replie sur 3 ?)")
        fig.tight_layout()
        fig.savefig(FIG_DIR / "eval_priorites.png", dpi=150)
        plt.close(fig)

    print(f"Graphiques générés dans {FIG_DIR}/")
    for f in sorted(FIG_DIR.glob("*.png")):
        print("  -", f.name)


if __name__ == "__main__":
    main()
