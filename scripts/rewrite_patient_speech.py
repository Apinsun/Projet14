#!/usr/bin/env python3
"""Réécrit les messages du PATIENT en langage naturel (27B), sans toucher à l'agent.

Les patients des dialogues synthétiques parlent de façon télégraphique et avec du jargon
médical (« J'ai obésité et HTA »). On fait réécrire UNIQUEMENT les tours patient par le
27B en français parlé naturel, en conservant tous les faits (chiffres, médicaments).
Les tours agent (``<think>`` + fiche + question) et les fiches restent inchangés.

Validation : même nombre de messages, chiffres conservés, noms de médicaments conservés.
Si échec → repli sur le tour original (jamais de perte d'info).

Usage :
    poetry run python scripts/rewrite_patient_speech.py --limit 20 --out .../test.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.data.dialogue_llm import _NO_THINK_STOP, MODEL
from triage_agent.data.generate import generate_balanced_fact_sheets
from triage_agent.data.vignette import ollama_chat

POOLS = [(30, 43), (200, 44), (12, 43)]

REWRITE_SYSTEM = (
    "Tu es un assistant qui réécrit des dialogues médicaux. Ta tâche : réécrire "
    "UNIQUEMENT les messages du PATIENT en français naturel et parlé, comme une vraie "
    "personne qui tape sur un chat. Règles strictes : "
    "1) conserve EXACTEMENT tous les faits médicaux : symptômes, durées, chiffres, "
    "intensités de douleur, température, noms de médicaments ; "
    "2) traduis le jargon médical en langage courant (ex. « cardiopathie ischémique » → "
    "« des problèmes de cœur », « HTA » → « de la tension », « obésité » → « du surpoids ») ; "
    "3) garde les noms de médicaments tels quels ; "
    "4) fais des phrases complètes et naturelles, pas de style télégraphique ; "
    "5) ne change PAS le sens ni l'ordre des messages. "
    "Réponds UNIQUEMENT en JSON : {\"patient_messages\": [\"...\", \"...\"]} avec le MÊME "
    "nombre de messages que l'entrée, dans le même ordre."
)


def build_lookup() -> dict[str, dict]:
    lookup: dict[str, dict] = {}
    for per_level, seed in POOLS:
        for f in generate_balanced_fact_sheets(per_level, seed):
            lookup.setdefault(f["case_id"], f)
    return lookup


def _parse_json(text: str) -> dict | None:
    text = text.strip()
    if text.startswith("```"):
        text = text.lstrip("`")
        if text.lower().startswith("json"):
            text = text[4:]
        text = text.rstrip("`").strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                return json.loads(m.group(0))
            except json.JSONDecodeError:
                return None
    return None


def validate(original: list[str], rewritten: list[str], treatments: list[str]) -> tuple[bool, str]:
    if len(original) != len(rewritten):
        return False, "nombre de messages différent"
    if any(not r.strip() for r in rewritten):
        return False, "message vide"
    orig_digits = set(re.findall(r"\d+", " ".join(original)))
    new_digits = set(re.findall(r"\d+", " ".join(rewritten)))
    if not orig_digits.issubset(new_digits):
        return False, "chiffre perdu"
    joined_new = " ".join(rewritten).lower()
    for name in treatments:
        if name.lower() in " ".join(original).lower() and name.lower() not in joined_new:
            return False, f"médicament perdu : {name}"
    return True, "ok"


def rewrite(model: str, patient_msgs: list[str]) -> list[str] | None:
    prompt = "\n".join(f"{i + 1}. {m}" for i, m in enumerate(patient_msgs))
    raw = ollama_chat(model, REWRITE_SYSTEM, prompt, temperature=0.4, timeout=600,
                      think=False, stop=_NO_THINK_STOP)
    data = _parse_json(raw)
    if not data:
        return None
    msgs = data.get("patient_messages")
    if not isinstance(msgs, list):
        return None
    return [str(m).strip() for m in msgs]


def main() -> None:
    ap = argparse.ArgumentParser(description="Réécrit les tours patient en langage naturel")
    ap.add_argument("--limit", type=int, default=0, help="Nb de dialogues à traiter (0 = tous)")
    ap.add_argument("--input", default=str(PROCESSED_DIR / "triage" / "sft_multiturn_fiche.jsonl"))
    ap.add_argument("--out", default=str(PROCESSED_DIR / "triage" / "sft_multiturn_fiche_nat.jsonl"))
    ap.add_argument("--model", default=MODEL)
    args = ap.parse_args()

    lookup = build_lookup()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    done = rewritten = failed = 0
    with out.open("w", encoding="utf-8") as fh:
        for line in Path(args.input).open(encoding="utf-8"):
            rec = json.loads(line)
            fact = lookup.get(rec["metadata"]["case_id"], {})
            treatments = fact.get("treatments", [])

            user_idx = [i for i, m in enumerate(rec["messages"]) if m["role"] == "user"]
            original = [rec["messages"][i]["content"] for i in user_idx]

            try:
                new = rewrite(args.model, original)
            except Exception:
                new = None

            if new is None:
                ok, reason = False, "erreur LLM/JSON"
            else:
                ok, reason = validate(original, new, treatments)

            if ok:
                for i, txt in zip(user_idx, new):
                    rec["messages"][i]["content"] = txt
                rewritten += 1
            else:
                failed += 1
                rec["metadata"]["rewrite_failed"] = reason

            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            done += 1
            if done % 5 == 0:
                print(f"  {done}... (réécrits {rewritten} | replis {failed})", flush=True)
            if args.limit and done >= args.limit:
                break

    print(f"Terminé : {done} dialogues | réécrits {rewritten} | replis {failed} -> {out}")


if __name__ == "__main__":
    main()
