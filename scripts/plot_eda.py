#!/usr/bin/env python3
"""Génère les figures du rapport EDA dans ``reports/figures/``.

Usage :
    poetry run python scripts/plot_eda.py
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # pas d'affichage interactif
import matplotlib.pyplot as plt

from triage_agent.data.config import REPO_ROOT

FIG_DIR = REPO_ROOT / "reports" / "figures"


def load_results() -> dict:
    return json.loads((REPO_ROOT / "reports" / "eda_results.json").read_text(encoding="utf-8"))


def plot_volumes(results: dict) -> None:
    """Bar chart (log) des volumes totaux par dataset + volumes exploitables."""
    labels, totals, usable = [], [], []
    for name, r in results.items():
        total = sum(r["splits"].values())
        labels.append(name)
        totals.append(total)
        if name == "medquad":
            # Réponses non nulles uniquement.
            usable.append(total - r["missing"]["answer"])
        elif name == "mediqa":
            usable.append(total)  # tout est exploitable après extraction de la réponse gold
        else:
            usable.append(total)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    x = range(len(labels))
    ax.bar([i - 0.2 for i in x], totals, width=0.4, label="Total", color="#4C72B0")
    ax.bar([i + 0.2 for i in x], usable, width=0.4, label="Exploitable (SFT/DPO)", color="#55A868")
    ax.set_yscale("log")
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=20, ha="right")
    ax.set_ylabel("Nombre de lignes (échelle log)")
    ax.set_title("Volumes par dataset (brut vs exploitable)")
    for i, (t, u) in enumerate(zip(totals, usable)):
        ax.text(i - 0.2, t, f"{t:,}", ha="center", va="bottom", fontsize=8)
        if u != t:
            ax.text(i + 0.2, u, f"{u:,}", ha="center", va="bottom", fontsize=8)
    ax.legend()
    fig.tight_layout()
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / "volumes.png", dpi=130)
    plt.close(fig)


def plot_medquad_question_types(results: dict) -> None:
    """Top 10 des types de questions MedQuAD."""
    types = results["medquad"]["specific"]["question_type"]
    items = sorted(types.items(), key=lambda kv: kv[1]["count"], reverse=True)[:10]
    labels = [k for k, _ in items]
    counts = [v["count"] for _, v in items]

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(labels[::-1], counts[::-1], color="#C44E52")
    ax.set_xlabel("Nombre de questions")
    ax.set_title("MedQuAD — Top 10 des types de questions")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "medquad_question_types.png", dpi=130)
    plt.close(fig)


def main() -> None:
    results = load_results()
    plot_volumes(results)
    plot_medquad_question_types(results)
    print("Figures écrites dans", FIG_DIR)


if __name__ == "__main__":
    main()
