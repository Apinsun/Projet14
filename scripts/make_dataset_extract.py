#!/usr/bin/env python3
"""Génère un extrait lisible du dataset de triage pour présentation (mentor)."""

from __future__ import annotations

import json
import re
from pathlib import Path

PROCESSED = Path("data/processed")


def load(name: str) -> list[dict]:
    return [json.loads(line) for line in (PROCESSED / name).open(encoding="utf-8")]


def split_fiche(text: str) -> tuple[str, str]:
    """Sépare le bloc [FICHE]...[/FICHE] de l'explication qui suit."""
    m = re.search(r"(?:\[FICHE\]|<FICHE>)\s*(.*?)\s*(?:\[/FICHE\]|</FICHE>)\s*(.*)", text, re.S)
    if not m:
        return "", text
    fiche = m.group(1)
    expl = m.group(2).strip()
    try:
        fiche = json.dumps(json.loads(fiche), ensure_ascii=False, indent=2)
    except json.JSONDecodeError:
        pass
    return fiche, expl


def fmt_fiche(text: str) -> str:
    fiche, expl = split_fiche(text)
    out = "```json\n" + fiche + "\n```"
    if expl:
        out += f"\n**Explication :** {expl}"
    return out


def main() -> None:
    vignettes = load("triage/sft_vignettes.jsonl")
    dialogues = load("triage/sft_dialogues_dressed.jsonl")
    base = load("final/sft_train.jsonl")

    lines: list[str] = []
    lines.append("# Extrait du dataset de triage — POC agent IA (CHSA)")
    lines.append("")
    lines.append("> Exemples réels issus du dataset utilisé pour le fine-tuning de Qwen3-1.7B")
    lines.append("> (SFT LoRA). Niveau de triage **calculé** selon le protocole FRENCH (5 niveaux),")
    lines.append("> jamais deviné par le LLM.")
    lines.append("")

    # --- Composition ---
    lines.append("## Composition du dataset d'entraînement")
    lines.append("")
    lines.append("| Jeu | Langue | Exemples | Rôle |")
    lines.append("|---|---|---|---|")
    lines.append("| MedQuAD + FrenchMedMCQA + MediQA | FR + EN | 4 500 | connaissance médicale |")
    lines.append("| Vignettes de triage synthétiques | FR | 300 | patient → fiche (mono-tour) |")
    lines.append("| Dialogues de triage synthétiques | FR | 300 | questionnaire multi-tours |")
    lines.append("| **Total** | | **5 100** | |")
    lines.append("")

    # --- Vignettes par niveau ---
    lines.append("## 1. Vignettes — un exemple par niveau FRENCH")
    lines.append("")
    lines.append("> Le patient décrit ses symptômes ; l'agent répond par une **fiche structurée**")
    lines.append("> `[FICHE] ... [/FICHE]` + une explication en langage clair.")
    lines.append("")
    by_level = {}
    for v in vignettes:
        by_level.setdefault(v["metadata"]["true_level"], v)
    for lvl in range(1, 6):
        v = by_level[lvl]
        lines.append(f"### Niveau {lvl}")
        lines.append("")
        for m in v["messages"]:
            if m["role"] == "user":
                lines.append(f"**Patient :** {m['content']}")
                lines.append("")
            elif m["role"] == "assistant":
                lines.append("**Agent :**")
                lines.append("")
                lines.append(fmt_fiche(m["content"]))
                lines.append("")
        lines.append("---")
        lines.append("")

    # --- Dialogue multi-tours ---
    d = [x for x in dialogues if x["metadata"]["true_level"] == 2][0]
    lines.append("## 2. Dialogue — questionnaire multi-tours")
    lines.append("")
    lines.append("> L'agent pose **une question à la fois**, raisonne en interne dans des balises")
    lines.append("> `<think>...</think>` (masquées au patient, conservées pour l'audit), puis conclut")
    lines.append("> par la fiche.")
    lines.append("")
    for m in d["messages"]:
        if m["role"] == "system":
            lines.append(f"**Consigne système :** {m['content']}")
            lines.append("")
        elif m["role"] == "user":
            lines.append(f"**Patient :** {m['content']}")
            lines.append("")
        else:
            content = m["content"]
            think = re.search(r"<think>(.*?)</think>", content, re.S)
            if think:
                rest = content.replace(think.group(0), "").strip()
                lines.append(f"**Agent** *(raisonnement interne)* : `{think.group(1).strip()}`")
                lines.append("")
                if "[FICHE]" in rest or "<FICHE>" in rest:
                    lines.append(fmt_fiche(rest))
                else:
                    lines.append(f"**Agent :** {rest}")
                lines.append("")
            else:
                lines.append(f"**Agent :** {content}")
                lines.append("")

    # --- Base Q&A ---
    lines.append("## 3. Base de connaissance médicale (Q&A)")
    lines.append("")
    lines.append("### Anglais (MedQuAD)")
    lines.append("")
    ex = base[0]
    lines.append(f"**Question :** {ex['messages'][0]['content']}")
    lines.append("")
    lines.append(f"**Réponse :** {ex['messages'][1]['content'][:400]}…")
    lines.append("")
    lines.append("### Français (FrenchMedMCQA)")
    lines.append("")
    ex = base[1]
    lines.append(f"**Question :** {ex['messages'][0]['content']}")
    lines.append("")
    lines.append(f"**Réponse :** {ex['messages'][1]['content'][:400]}…")
    lines.append("")

    out = Path("reports/extrait_dataset.md")
    out.write_text("\n".join(lines), encoding="utf-8")
    print(f"Extrait généré → {out}")


if __name__ == "__main__":
    main()
