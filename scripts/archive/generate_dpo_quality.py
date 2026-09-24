#!/usr/bin/env python3
"""Génère les paires DPO « qualité de service » (politesse, questionnement, fiche).

Le 27B écrit la version **rejected** (dégradée) ; le **chosen** est notre réponse
correcte existante. La fiche reste programmatique (règle FRENCH) et est injectée dans
le rejected final — on ne touche JAMAIS au niveau de triage.

Usage :
    poetry run python scripts/generate_dpo_quality.py --pilot
    poetry run python scripts/generate_dpo_quality.py --limit 150
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.data.dialogue_llm import _NO_THINK_STOP, MODEL
from triage_agent.data.vignette import ollama_chat

_FICHE_RE = re.compile(r"(?:\[FICHE\]|<FICHE>)\s*.*?\s*(?:\[/FICHE\]|</FICHE>)", re.S)

_FINAL_SYSTEM = (
    "Tu es un agent de triage médical MALPOLI, brusque et pressé. Rédige une réponse "
    "IMPOLIE et BRUSQUE pour ce patient : sans formule de politesse, sans rassurer, "
    "SANS expliquer la prise en charge, SANS fiche. Réponds uniquement avec la phrase brusque."
)

_QUEST_SYSTEM = (
    "Tu es un agent de triage médical MALPOLI et pressé. Réponds à ce patient de façon "
    "BRUSQUE, en UNE phrase, SANS poser de question, SANS l'aider à préciser son problème."
)


def _fiche_block(content: str) -> str:
    m = _FICHE_RE.search(content)
    return m.group(0) if m else ""


def _clean(text: str) -> str:
    text = text.strip().strip('"').strip()
    # garde seulement la 1re phrase si trop long
    if len(text) > 220:
        text = re.split(r"(?<=[.!?])\s+", text)[0]
    return text


def build_final_pairs(vignettes: list[dict], model: str, limit: int) -> list[dict]:
    pairs: list[dict] = []
    for rec in vignettes[:limit]:
        msgs = rec["messages"]
        if len(msgs) < 3:
            continue
        fiche = _fiche_block(msgs[2]["content"])
        if not fiche:
            continue
        user_text = msgs[1]["content"]
        try:
            rude = ollama_chat(
                model, _FINAL_SYSTEM,
                f"Patient : {user_text}\n\nRéponse brusque (sans fiche) :",
                temperature=0.8, timeout=600, think=False, stop=_NO_THINK_STOP,
            )
        except Exception:
            continue
        rejected = _clean(rude) + "\n\n" + fiche
        pairs.append({
            "prompt": msgs[:2],
            "chosen": [{"role": "assistant", "content": msgs[2]["content"]}],
            "rejected": [{"role": "assistant", "content": rejected}],
        })
    return pairs


def build_questioning_pairs(dialogues: list[dict], model: str, limit: int) -> list[dict]:
    pairs: list[dict] = []
    for rec in dialogues[:limit]:
        msgs = rec["messages"]
        if len(msgs) < 4 or msgs[2]["role"] != "assistant":
            continue
        user_text = msgs[1]["content"]
        try:
            rude = ollama_chat(
                model, _QUEST_SYSTEM,
                f"Patient : {user_text}\n\nRéponse brusque (sans question) :",
                temperature=0.8, timeout=600, think=False, stop=_NO_THINK_STOP,
            )
        except Exception:
            continue
        pairs.append({
            "prompt": msgs[:2],
            "chosen": [{"role": "assistant", "content": msgs[2]["content"]}],
            "rejected": [{"role": "assistant", "content": _clean(rude)}],
        })
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description="Génère les paires DPO qualité")
    ap.add_argument("--pilot", action="store_true", help="~30 paires de validation")
    ap.add_argument("--limit", type=int, default=150, help="Nb max de paires par type")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(PROCESSED_DIR / "triage" / "dpo_quality.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    v_path = PROCESSED_DIR / "triage" / "sft_vignettes.jsonl"
    d_path = PROCESSED_DIR / "triage" / "sft_multiturn_full.jsonl"
    vignettes = [json.loads(line) for line in v_path.open(encoding="utf-8")]
    dialogues = [json.loads(line) for line in d_path.open(encoding="utf-8")]
    rng.shuffle(vignettes)
    rng.shuffle(dialogues)

    lim = 15 if args.pilot else args.limit
    final = build_final_pairs(vignettes, args.model, lim)
    quest = build_questioning_pairs(dialogues, args.model, lim)

    pairs = final + quest
    rng.shuffle(pairs)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")

    print(f"Paires qualité : {len(pairs)} (final {len(final)} + questionnement {len(quest)})")
    print(f"Sortie : {out}")


if __name__ == "__main__":
    main()
