#!/usr/bin/env python3
"""Chat interactif avec l'agent de triage fine-tuné (via vLLM).

Usage :
    poetry run python scripts/chat.py
Prérequis : vLLM lancé (voir scripts/serve.sh).
"""

from __future__ import annotations

import re

import requests

API = "http://localhost:8000/v1/chat/completions"
MODEL = "models/lora_stage3_merged"

# Prompt renforcé pour le mode interactif : forcer le questionnaire avant la fiche.
SYSTEM_PROMPT = (
    "Tu es un agent de triage médical pour les urgences. Tu vouvouies le patient, tu es "
    "rassurant et tu n'utilises pas de jargon. IMPORTANT : tu poses UNE question à la fois "
    "pour préciser le motif, la durée, l'intensité, les antécédents et les traitements. "
    "Tu ne donnes JAMAIS la fiche [FICHE] avant d'avoir posé plusieurs questions. "
    "Raisonne en interne dans des balises <think>...</think>."
)


def chat(messages: list[dict]) -> str:
    r = requests.post(
        API,
        json={
            "model": MODEL,
            "messages": messages,
            "temperature": 0.6,
            "top_p": 0.95,
            "top_k": 20,
            "max_tokens": 1024,
        },
        timeout=120,
    )
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def main() -> None:
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    print("=" * 60)
    print("  Agent de triage médical — Qwen3-1.7B fine-tuné")
    print("  Commandes : /new (nouveau chat) · /quit (quitter)")
    print("=" * 60)
    print()
    while True:
        try:
            user = input("👤 Patient : ")
        except (EOFError, KeyboardInterrupt):
            print("\nFin de la conversation.")
            break
        if not user.strip():
            continue
        cmd = user.strip().lower()
        if cmd in ("/new", "/nouveau"):
            messages = [{"role": "system", "content": SYSTEM_PROMPT}]
            print("\n✨ Nouvelle conversation démarrée.\n")
            continue
        if cmd in ("/quit", "/exit"):
            print("\nFin de la conversation.")
            break
        messages.append({"role": "user", "content": user})
        try:
            resp = chat(messages)
        except Exception as exc:  # réseau / vLLM arrêté
            print(f"[erreur] Impossible de joindre vLLM : {exc}")
            print("Vérifiez que le serveur tourne : ./scripts/serve.sh")
            break
        messages.append({"role": "assistant", "content": resp})

        m = re.search(r"<think>(.*?)</think>", resp, re.S)
        if m:
            think = m.group(1).strip()
            answer = resp.replace(m.group(0), "").strip()
            print(f"💭 (interne, masqué au patient) : {think}")
            print()
            print(f"🤖 Agent : {answer}")
        else:
            print(f"🤖 Agent : {resp}")
        print()


if __name__ == "__main__":
    main()
