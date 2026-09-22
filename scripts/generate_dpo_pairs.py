#!/usr/bin/env python3
"""Génère les paires de préférence DPO pour le triage.

Deux types :
1. **sous-triage** (mono-tour, depuis les vignettes) :
   chosen = fiche au bon niveau (règle) · rejected = fiche sous-triée (+2, sans red flag).
2. **questionnement** (multi-tour, depuis les dialogues) :
   chosen = une question (ne conclut pas) · rejected = fiche prématurée.

Format TRL (conversationnel) : {"prompt": [...], "chosen": [...], "rejected": [...]}

Usage :
    poetry run python scripts/generate_dpo_pairs.py --pilot
    poetry run python scripts/generate_dpo_pairs.py --limit 400
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from pathlib import Path

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.data.fiche import explanation as _explanation
from triage_agent.eval.harness import parse_fiche

_TAG_RE = re.compile(r"(\[FICHE\]|<FICHE>)")
_CLOSE_RE = re.compile(r"(\[/FICHE\]|</FICHE>)")

_GENERIC_RECO = {
    1: "situation grave — prise en charge immédiate",
    2: "prise en charge rapide",
    3: "évaluation dans l'heure",
    4: "situation stable — consultation dans les 2 heures",
    5: "situation non urgente — consultation dans la journée",
}


def _tag_of(content: str) -> str:
    m = _TAG_RE.search(content)
    return m.group(1) if m else "<FICHE>"


def _rebuild_fiche(fiche: dict, tag: str, priority: int | None = None) -> str:
    """Reconstruit le bloc fiche, en option avec un autre niveau (sous-triage)."""
    args = copy.deepcopy(fiche.get("arguments", fiche))
    if priority is not None:
        args["priority"] = priority
        args["priority_range"] = [priority, priority]
        args["red_flags"] = []
        args["summary"] = "symptôme bénin, pas de signe de gravité"
        args["recommendation"] = _GENERIC_RECO[priority]
    body = json.dumps({"name": "finalize_triage", "arguments": args}, ensure_ascii=False)
    close = "</FICHE>" if tag == "<FICHE>" else "[/FICHE]"
    return f"{tag}\n{body}\n{close}"


def _explanation_text(rule_level_label: str) -> str:
    """Explication patient selon le niveau (repli sur le texte générique)."""
    try:
        return _explanation({"level": rule_level_label})
    except KeyError:
        return _GENERIC_RECO[int(rule_level_label[0])]


def build_undertriage_pairs(vignettes: list[dict]) -> list[dict]:
    """Chaque vignette → 1 paire : chosen=bon niveau, rejected=sous-trié (+2)."""
    pairs: list[dict] = []
    for rec in vignettes:
        msgs = rec["messages"]
        if len(msgs) < 3:
            continue
        prompt = msgs[:2]
        chosen_content = msgs[2]["content"]
        fiche = parse_fiche(chosen_content)
        if fiche is None:
            continue
        prio = int(fiche.get("arguments", fiche).get("priority", 3))
        new_prio = min(prio + 2, 5)
        tag = _tag_of(chosen_content)
        # Explication : on garde celle du chosen (niveau correct) mais on la remplace
        # par une version "bénigne" cohérente avec le sous-triage.
        rejected_content = (
            _rebuild_fiche(fiche, tag, new_prio) + "\n"
            + _GENERIC_RECO[new_prio].replace("situation", "Votre situation").replace(
                "prise en charge", "vous serez pris en charge").replace(
                "évaluation", "vous serez évalué")
        )
        pairs.append({
            "prompt": prompt,
            "chosen": [{"role": "assistant", "content": chosen_content}],
            "rejected": [{"role": "assistant", "content": rejected_content}],
        })
    return pairs


def build_questioning_pairs(dialogues: list[dict]) -> list[dict]:
    """Chaque dialogue → 1 paire : chosen=question (tour 1), rejected=fiche finale."""
    pairs: list[dict] = []
    for rec in dialogues:
        msgs = rec["messages"]
        if len(msgs) < 4:
            continue
        # Le dialogue doit bien commencer par un symptôme partiel → question.
        if msgs[0]["role"] != "system" or msgs[2]["role"] != "assistant":
            continue
        question = msgs[2]["content"]
        final = msgs[-1]["content"]
        if "<FICHE>" not in final and "[FICHE]" not in final:
            continue
        pairs.append({
            "prompt": msgs[:2],
            "chosen": [{"role": "assistant", "content": question}],
            "rejected": [{"role": "assistant", "content": final}],
        })
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Génère les paires DPO triage")
    ap.add_argument("--pilot", action="store_true", help="Petit jeu de validation (~100 paires)")
    ap.add_argument("--limit", type=int, default=0, help="Nombre max de paires par type (0 = tous)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(PROCESSED_DIR / "triage" / "dpo_pairs.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    v_path = PROCESSED_DIR / "triage" / "sft_vignettes.jsonl"
    d_path = PROCESSED_DIR / "triage" / "sft_multiturn_full.jsonl"
    vignettes = [json.loads(line) for line in v_path.open(encoding="utf-8")]
    dialogues = [json.loads(line) for line in d_path.open(encoding="utf-8")]

    under = build_undertriage_pairs(vignettes)
    quest = build_questioning_pairs(dialogues)

    if args.limit:
        under = under[: args.limit]
        quest = quest[: args.limit]
    if args.pilot:
        under = rng.sample(under, min(50, len(under)))
        quest = rng.sample(quest, min(50, len(quest)))

    pairs = under + quest
    rng.shuffle(pairs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Paires générées : {len(pairs)} (sous-triage {len(under)} + questionnement {len(quest)})")
    print(f"Sortie : {out}")


if __name__ == "__main__":
    main()
