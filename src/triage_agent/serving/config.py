"""Configuration de l'application de serving (env / ``.env``, préfixe ``TRIAGE_``)."""

from __future__ import annotations

from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict

# Prompt système : identique au prompt d'entraînement (fiche + think à chaque tour).
SYSTEM_PROMPT = (
    "Tu es un agent de triage médical pour les urgences. Tu vouvouies le patient, tu es "
    "rassurant et tu n'utilises pas de jargon. Tu poses UNE question à la fois pour préciser "
    "le motif, la durée, l'intensité, les antécédents et les traitements. À CHAQUE tour, tu "
    "réponds dans cet ordre : <think> ton raisonnement interne (faits connus, red flags, "
    "plage d'urgence, prochaine question) </think>, puis une fiche entre <FICHE> et </FICHE> "
    "(incomplète tant que des infos manquent, complète au niveau final), puis ta question "
    "(si info manquante) ou ta conclusion au patient."
)


class Settings(BaseSettings):
    """Réglages lus depuis l'environnement (préfixe ``TRIAGE_``) et ``.env``."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="TRIAGE_", extra="ignore")

    # vLLM
    vllm_url: str = "http://localhost:8000/v1"
    vllm_model: str = "models/lora_dpo_v3_merged"

    # Génération (recommandations Qwen3 : thinking mode, pas de greedy)
    temperature: float = 0.6
    top_p: float = 0.95
    top_k: int = 20
    max_tokens: int = 1024

    # Audit
    audit_path: str = "logs/audit.jsonl"

    # Serving
    host: str = "0.0.0.0"
    port: int = 8080


@lru_cache
def get_settings() -> Settings:
    """Retourne les réglages (cache pour ne lire l'env qu'une fois)."""
    return Settings()
