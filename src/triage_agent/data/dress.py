"""Habillage LLM des dialogues scriptés + vérification anti-incohérence.

Le LLM reformule **chaque réponse patient** (tour par tour) pour la rendre
naturelle. On vérifie ensuite l'équivalence avec l'original :

1. aucun terme interdit (ECG, SpO2, GCS, PAS, saturation…) — pas de fuite ;
2. les **chiffres** sont préservés (douleur, durée, température) ;
3. les **mots de contenu** (maladies, médicaments) sont en grande partie conservés ;
4. pas d'allongement massif (pas d'ajout).

Si un contrôle échoue → on **garde la version scriptée** (correcte par construction).
"""

from __future__ import annotations

import re

from triage_agent.data.vignette import ollama_chat

MODEL = "Leila_fast:latest"

# Termes que le patient ne peut PAS connaître/mesurer lui-même.
# Les acronymes sont vérifiés en MAJUSCULE (sinon "pas" matcherait "PAS").
_ACRONYMS = re.compile(r"\b(ECG|SpO2|GCS|DEP|PAS)\b")
_FRENCH = re.compile(
    r"\b(saturation|oxymétrie|fréquence respiratoire|pression artérielle|tension artérielle)\b",
    re.I,
)

_STOPWORDS = {
    "bonjour", "vraiment", "beaucoup", "depuis", "combien", "temps", "autre",
    "encore", "jours", "heures", "minutes", "cette", "cette", "quelque",
}

DRESS_SYSTEM = (
    "Tu reformules la réponse d'un patient aux urgences en français parlé naturel "
    "(hésitations, langage simple, ton réaliste). Tu conserves exactement les mêmes "
    "informations : les chiffres, les durées, les noms de maladies et de médicaments. "
    "Tu n'ajoutes aucun symptôme, aucune constante mesurée, aucun diagnostic. "
    "Réponds uniquement avec la phrase reformulée, sans guillemets, sans commentaire."
)


def _content_words(text: str) -> set[str]:
    return {w for w in re.findall(r"[a-zA-Zà-ÿ]{5,}", text.lower()) if w not in _STOPWORDS}


def dress_text(question: str, answer: str, model: str = MODEL) -> str:
    """Reformule la réponse du patient (``answer``) pour la question ``question``."""
    user = (
        f"Question de l'agent : {question}\n"
        f"Réponse du patient (à reformuler) : {answer}\n"
        "Réponse reformulée :"
    )
    return ollama_chat(model, DRESS_SYSTEM, user, temperature=0.7).strip()


def verify(original: str, dressed: str) -> tuple[bool, str]:
    """Vérifie que ``dressed`` est équivalent à ``original`` (aucune incohérence)."""
    if _ACRONYMS.search(dressed) or _FRENCH.search(dressed):
        return False, "terme interdit"
    if not set(re.findall(r"\d+", original)).issubset(set(re.findall(r"\d+", dressed))):
        return False, "chiffre modifié"
    o_words = _content_words(original)
    if o_words:
        kept = len(o_words & _content_words(dressed)) / len(o_words)
        if kept < 0.6:
            return False, "information perdue"
    if len(dressed) > len(original) * 4 + 80:
        return False, "allongement suspect"
    return True, "ok"


def dress_dialogue(messages: list[dict], model: str = MODEL) -> tuple[list[dict], dict]:
    """Habille les réponses patient d'un dialogue ; retourne (messages, stats)."""
    out = [dict(m) for m in messages]
    stats = {"dressed": 0, "kept": 0, "reasons": {}}

    for i, msg in enumerate(out):
        if msg["role"] != "user" or i == 1:
            # On ne touche ni au système, ni aux tours agent, ni à la 1re ouverture vague.
            continue
        prev = out[i - 1]["content"] if i > 0 else ""
        try:
            dressed = dress_text(prev, msg["content"], model)
            ok, reason = verify(msg["content"], dressed)
        except Exception:  # résilience : échec réseau -> on garde l'original
            stats["kept"] += 1
            stats["reasons"]["erreur"] = stats["reasons"].get("erreur", 0) + 1
            continue
        if ok:
            msg["content"] = dressed
            stats["dressed"] += 1
        else:
            stats["kept"] += 1
            stats["reasons"][reason] = stats["reasons"].get(reason, 0) + 1

    return out, stats
