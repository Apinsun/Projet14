#!/usr/bin/env python3
"""Traduit les messages patient des jeux gold en français naturel (via Ollama/Leila).

Conserve les métadonnées (gold_range, binary_urgent) et garde l'original dans
``metadata.original_text``.
"""

from __future__ import annotations

import json

from triage_agent.data.config import PROCESSED_DIR
from triage_agent.data.vignette import ollama_chat

MODEL = "Leila_fast:latest"
SYSTEM = (
    "Tu es un traducteur médical. Traduis ce message de patient en français naturel, "
    "comme si le patient s'exprimait lui-même aux urgences (langage parlé, pas de liste). "
    "Conserve EXACTEMENT tous les faits médicaux : symptômes, durées, intensités, "
    "antécédents, traitements, chiffres. Réponds uniquement avec le message traduit, "
    "sans guillemets, sans commentaire."
)


def translate(text: str, model: str = MODEL) -> str:
    return ollama_chat(model, SYSTEM, text, temperature=0.2).strip()


def main() -> None:
    src_dir = PROCESSED_DIR / "gold"
    dst_dir = src_dir / "fr"
    dst_dir.mkdir(parents=True, exist_ok=True)

    for name in ("ramaswamy", "levine"):
        records = [json.loads(line) for line in (src_dir / f"{name}.jsonl").open(encoding="utf-8")]
        out = []
        for i, rec in enumerate(records):
            user = rec["messages"][1]["content"]
            fr = translate(user)
            new = json.loads(json.dumps(rec))  # copie profonde
            new["messages"][1]["content"] = fr
            new["metadata"]["lang"] = "fr"
            new["metadata"]["original_text"] = user
            out.append(new)
            if (i + 1) % 5 == 0:
                print(f"  {name}: {i + 1}/{len(records)}")

        with (dst_dir / f"{name}.jsonl").open("w", encoding="utf-8") as fh:
            for r in out:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"{name}: {len(out)} traduits → {dst_dir / f'{name}.jsonl'}")


if __name__ == "__main__":
    main()
