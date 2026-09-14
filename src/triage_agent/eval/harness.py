"""Harness d'évaluation du triage.

Pipeline : lancer le modèle sur ``messages`` → extraire la fiche ``[FICHE]`` → lire
``priority`` → comparer à la plage gold → agréger les métriques.

Métriques :
- exactitude (``priority`` dans la plage acceptée) ;
- taux de **sous-triage** (``priority`` > borne haute) et distance moyenne ;
- taux de **sur-triage** (``priority`` < borne basse) et distance moyenne ;
- **binaire de sécurité** : sous-triage sur les cas « urgences maintenant ».
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable

import requests

logger = __import__("logging").getLogger("triage_agent.eval.harness")

OLLAMA_URL = "http://localhost:11434/api/chat"
VLLM_URL = "http://localhost:8000/v1"
DEFAULT_MODEL = "Leila_fast:latest"

_FICHE_RE = re.compile(r"(?:\[FICHE\]|<FICHE>)\s*(.*?)\s*(?:\[/FICHE\]|</FICHE>)", re.S)


def parse_fiche(text: str) -> dict | None:
    """Extrait et parse le JSON entre ``[FICHE]`` et ``[/FICHE]``. ``None`` si absent/invalide."""
    m = _FICHE_RE.search(text or "")
    if not m:
        return None
    raw = re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", m.group(1)).strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        return None


def extract_priority(fiche: dict) -> int | None:
    """Lit la priorité de la fiche (``arguments.priority``, sinon borne prudente de la plage)."""
    try:
        args = fiche.get("arguments", fiche)
        prio = args.get("priority")
        if prio is None:
            rng = args.get("priority_range")
            if rng:
                prio = min(int(x) for x in rng)  # borne prudente = la plus urgente
        return int(prio) if prio is not None else None
    except (TypeError, ValueError, AttributeError):
        return None


def correct(priority: int, gold_range: list[int]) -> dict:
    """Compare une priorité à la plage gold → distances de sous/sur-triage."""
    lo, hi = gold_range
    under = max(0, priority - hi)
    over = max(0, lo - priority)
    return {"correct": under == 0 and over == 0, "under": under, "over": over}


def ollama_runner(model: str = DEFAULT_MODEL, temperature: float = 0.0) -> Callable:
    """Retourne un ``runner(messages) -> str`` adossé à Ollama (température 0 par défaut)."""

    def run(messages: list[dict]) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": temperature},
        }
        resp = requests.post(OLLAMA_URL, json=payload, timeout=180)
        resp.raise_for_status()
        return resp.json()["message"]["content"]

    return run


def openai_compat_runner(
    base_url: str = VLLM_URL,
    model: str = DEFAULT_MODEL,
    temperature: float = 0.6,
    max_tokens: int = 2048,
    top_p: float = 0.95,
    top_k: int = 20,
    seed: int = 42,
) -> Callable:
    """Retourne un ``runner(messages) -> str`` adossé à une API OpenAI-compatible (vLLM).

    Paramètres d'échantillonnage = recommandations Qwen3 (mode thinking : pas de greedy).
    """

    def run(messages: list[dict]) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "max_tokens": max_tokens,
            "seed": seed,
        }
        resp = requests.post(f"{base_url}/chat/completions", json=payload, timeout=300)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"]

    return run


def evaluate(records: list[dict], runner: Callable[[list[dict]], str]) -> dict:
    """Évalue une liste de records ; retourne métriques agrégées + détail par cas."""
    per_case = []
    for rec in records:
        gold = rec["metadata"]["gold_range"]
        base = {
            "case_id": rec["metadata"]["case_id"],
            "source": rec["metadata"]["source"],
            "domain": rec["metadata"]["domain"],
            "gold_label": rec["metadata"]["gold_label_original"],
            "gold_range": gold,
        }
        try:
            out = runner(rec["messages"])
        except Exception as exc:  # résilience réseau
            per_case.append({**base, "error": str(exc)})
            continue
        raw = out  # sortie complète (bornée par max_tokens, pas de risque de boucle infinie)
        fiche = parse_fiche(out)
        if fiche is None or extract_priority(fiche) is None:
            per_case.append({**base, "parse_fail": True, "raw_output": raw})
            continue
        prio = extract_priority(fiche)
        res = correct(prio, gold)
        per_case.append({**base, "priority": prio, "raw_output": raw, **res})

    parsed = [c for c in per_case if "priority" in c]
    n = len(records)
    n_parsed = len(parsed)

    under = [c for c in parsed if c["under"] > 0]
    over = [c for c in parsed if c["over"] > 0]
    urgent = [c for c in parsed if c["gold_range"][1] <= 3]
    under_urgent = [c for c in urgent if c["priority"] > 3]

    def _mean(lst: list[dict], key: str) -> float:
        return round(sum(c[key] for c in lst) / len(lst), 3) if lst else 0.0

    by_source: dict[str, dict] = {}
    for c in parsed:
        s = by_source.setdefault(c["source"], {"n": 0, "ok": 0})
        s["n"] += 1
        s["ok"] += int(c["correct"])

    return {
        "n_total": n,
        "n_parsed": n_parsed,
        "parse_fail": n - n_parsed,
        "accuracy": round(sum(c["correct"] for c in parsed) / n_parsed, 3) if n_parsed else None,
        "under_triage_rate": round(len(under) / n_parsed, 3) if n_parsed else None,
        "over_triage_rate": round(len(over) / n_parsed, 3) if n_parsed else None,
        "mean_under_distance": _mean(under, "under"),
        "mean_over_distance": _mean(over, "over"),
        "binary": {
            "n_urgent": len(urgent),
            "under_triage_rate": round(len(under_urgent) / len(urgent), 3) if urgent else None,
        },
        "by_source": {k: {"n": v["n"], "accuracy": round(v["ok"] / v["n"], 3)} for k, v in by_source.items()},
        "per_case": per_case,
    }


def report(metrics: dict) -> str:
    """Formate les métriques en texte lisible."""
    lines = [
        f"Total : {metrics['n_total']} | parses : {metrics['n_parsed']} "
        f"({metrics['parse_fail']} échecs)",
        f"Exactitude : {metrics['accuracy']}",
        f"Sous-triage : taux {metrics['under_triage_rate']} | "
        f"distance moyenne {metrics['mean_under_distance']}",
        f"Sur-triage  : taux {metrics['over_triage_rate']} | "
        f"distance moyenne {metrics['mean_over_distance']}",
    ]
    b = metrics["binary"]
    lines.append(f"[Binaire sécurité] urgences maintenant : {b['n_urgent']} | "
                 f"sous-triage {b['under_triage_rate']}")
    lines.append("Par source : " + ", ".join(
        f"{k}={v['accuracy']}" for k, v in metrics["by_source"].items()
    ))
    return "\n".join(lines)
