#!/usr/bin/env python3
"""Génère les paires DPO « qualité de service » v2 — rédigées en langage naturel.

Contrairement à v1 (hybride programmatique : fiche brute collée à une phrase du 27B) :
- le **prompt** est l'historique multi-tours COMPLET (jusqu'à la divergence) ;
- le **chosen** est le tour naturel de l'agent (déjà rédigé par le 27B lors du SFT :
  ``<think>`` + fiche + explication, ou ``<think>`` + question) ;
- le **rejected** est une version dégradée **RÉDIGÉE par le 27B** (même niveau, forme
  brusque) ; la fiche est injectée programmatiquement (règle FRENCH, jamais devinée) ;
- un **seul** system prompt (``<FICHE>``).

Deux types :
- **FINAL** : dernier tour d'un dialogue (tout est dit) → chosen = think+fiche+explication ;
  rejected = phrase brusque + même fiche (sans raisonnement, sans explication).
- **QUESTIONNEMENT** : premier tour (symptôme partiel) → chosen = think+question polie ;
  rejected = phrase brusque inutile (sans question, SANS escalade — le bug de v1).

Usage :
    poetry run python scripts/generate_dpo_quality_v2.py --pilot
    poetry run python scripts/generate_dpo_quality_v2.py --n 150
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

_FICHE_RE = re.compile(r"<FICHE>\s*.*?\s*</FICHE>", re.S)
_ESCALADE_RE = re.compile(r"\b(15|samu|urgences|hôpital|hopital|ambulance)\b", re.I)

RUDE_FINAL_SYSTEM = (
    "Tu es un agent de triage médical MALPOLI et brusque. Le patient a terminé de "
    "décrire son problème, tu dois conclure. Rédige UNE phrase de clôture IMPOLIE et "
    "BRUSQUE : sans formule de politesse, sans rassurer, sans expliquer la prise en "
    "charge, sans raisonnement, sans fiche, sans donner d'instruction ni d'orientation "
    "médicale (n'évoque ni les urgences, ni l'hôpital, ni le 15). Réponds uniquement "
    "avec cette phrase."
)

RUDE_QUEST_SYSTEM = (
    "Tu es un agent de triage médical MALPOLI et désagréable. Le patient vient de "
    "décrire son problème. Rédige UNE phrase BRUSQUE et INUTILE : sans poser de "
    "question, sans donner de conseil médical, sans orienter vers les urgences ni le "
    "15, sans rassurer, sans fiche. Réponds uniquement avec cette phrase."
)


def _fiche_block(content: str) -> str:
    m = _FICHE_RE.search(content)
    return m.group(0) if m else ""


def _clean(text: str) -> str:
    text = text.strip().strip('"').strip()
    if len(text) > 220:
        text = re.split(r"(?<=[.!?])\s+", text)[0]
    return text


def _rude(model: str, system: str, context: str) -> str:
    return ollama_chat(
        model, system, context,
        temperature=0.8, timeout=600, think=False, stop=_NO_THINK_STOP,
    )


def build_pairs(dialogues: list[dict], model: str, n: int) -> tuple[list[dict], list[dict]]:
    final: list[dict] = []
    quest: list[dict] = []
    for rec in dialogues:
        msgs = rec["messages"]
        if len(msgs) < 3:
            continue

        # --- paire FINAL : dernier tour (tout est dit) ---
        fiche = _fiche_block(msgs[-1]["content"])
        if fiche and len(final) < n:
            first_user = msgs[1]["content"]
            try:
                rude = _rude(model, RUDE_FINAL_SYSTEM,
                             f"Patient : {first_user}\n\nPhrase de clôture brusque :")
            except Exception:
                continue
            rude = _clean(rude)
            if _ESCALADE_RE.search(rude):  # pas de directive médicale contradictoire
                continue
            rejected = rude + "\n\n" + fiche
            final.append({
                "prompt": msgs[:-1],
                "chosen": [{"role": "assistant", "content": msgs[-1]["content"]}],
                "rejected": [{"role": "assistant", "content": rejected}],
            })

        # --- paire QUESTIONNEMENT : premier tour (symptôme partiel) ---
        if len(msgs) >= 5 and len(quest) < n:
            qfiche = _fiche_block(msgs[2]["content"])
            if not qfiche:
                continue
            try:
                rude = _rude(model, RUDE_QUEST_SYSTEM,
                             f"Patient : {msgs[1]['content']}\n\nPhrase brusque :")
            except Exception:
                continue
            rude = _clean(rude)
            if _ESCALADE_RE.search(rude):  # ne JAMAIS faire d'escalade correcte
                continue
            rejected = rude + "\n\n" + qfiche  # brusque + MÊME fiche incomplète
            quest.append({
                "prompt": msgs[:2],
                "chosen": [{"role": "assistant", "content": msgs[2]["content"]}],
                "rejected": [{"role": "assistant", "content": rejected}],
            })

        if len(final) >= n and len(quest) >= n:
            break
    return final, quest


def main() -> None:
    ap = argparse.ArgumentParser(description="Génère les paires DPO qualité v2 (rédigées)")
    ap.add_argument("--pilot", action="store_true", help="~15 paires/type de validation")
    ap.add_argument("--n", type=int, default=150, help="Nb de paires PAR type")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=str(PROCESSED_DIR / "triage" / "dpo_quality_v3.jsonl"))
    args = ap.parse_args()

    rng = random.Random(args.seed)
    dialogues = [json.loads(line) for line in
                 (PROCESSED_DIR / "triage" / "sft_multiturn_fiche_nat.jsonl").open(encoding="utf-8")]
    rng.shuffle(dialogues)

    n = 15 if args.pilot else args.n
    final, quest = build_pairs(dialogues, args.model, n)

    # vérification : fiche identique chosen/rejected (les deux types)
    ok_f = sum(1 for p in final
               if _fiche_block(p["chosen"][0]["content"]) == _fiche_block(p["rejected"][0]["content"]))
    ok_q = sum(1 for p in quest
               if _fiche_block(p["chosen"][0]["content"]) == _fiche_block(p["rejected"][0]["content"]))
    print(f"finales : {len(final)} | questionnement : {len(quest)}")
    print(f"fiche identique chosen/rejected : finales {ok_f}/{len(final) or 1} "
          f"| questionnement {ok_q}/{len(quest) or 1}")

    pairs = final + quest
    rng.shuffle(pairs)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    print(f"sortie : {out} ({len(pairs)} paires)")


if __name__ == "__main__":
    main()
