# TODO — prochaines itérations

> Priorités définies avec le mentor. Mise à jour au fil des sessions.

## 🔴 Priorité 1 — Corriger le multi-tour (questionnement)

Le mode interactif est **sous-appris** (fiche prématurée, réponses hallucinées, boucles) :
le dataset contient autant de vignettes (fiche immédiate) que de dialogues (question →
fiche), et le comportement « vignette » domine.

- [x] **Étendre le dataset multi-tour** : 1074 dialogues générés par le 27B (100 %
      corrects, 85 % LLM, niveaux équilibrés, symptôme en ouverture, mode tiers). Rapport 14.
- [ ] **Corriger la détection de douleur** (ignorer « sans douleur »/« indolore »).
- [ ] **Renforcer le prompt système dès l'entraînement** (« pose une question avant de
      conclure »), pas seulement à l'inférence.
- [ ] **Rééquilibrer vignettes/dialogues** (ou curriculum : dialogues entraînés en dernier).
- [ ] **DPO triage** : récompenser « question avant fiche » + « triage prudent », pénaliser
      « fiche prématurée » + « sous-triage ».

## 🟠 Priorité 2 — Automatisation & CI/CD

- [ ] **Automatiser le pipeline** : génération dataset → publication → training → évaluation.
- [ ] **CI/CD avec GitHub Actions** (tests, lint, build).
- [ ] **Publication HuggingFace** des modèles fusionnés + dataset (`models/`, `data/`).

## 🟡 Priorité 3 — Suivi / versioning des runs

- [ ] **MLflow** (décision retenue) :
      - log des hyperparamètres (`r`, `lr`, `epochs`, `batch`, `max_seq_length`) ;
      - log des métriques (loss, temps, `train_samples_per_second`) et des artefacts (adapter, config) ;
      - `mlflow ui` en local pour comparer/versionner les runs ;
      - registry de modèles pour versionner les adaptateurs fusionnés.
- [ ] **Unsloth Studio** : écarté pour l'instant (UI no-code, moins adapté à notre pipeline
      scripté) — à revoir seulement si besoin d'un monitoring no-code.

## 🟢 Priorité 4 — Déploiement (semaine 4)

- [ ] **vLLM + FastAPI + Docker** (exigé par le brief) + raisonnement `<think>` masqué.
- [ ] Test de **concurrence** vLLM (plusieurs patients simultanés).

---

### État actuel (rappel)

- ✅ Dataset de triage synthétique (600 : 300 vignettes + 300 dialogues) + base Q&A 4 500.
- ✅ SFT LoRA (Unsloth, 2 étapes) : 95 % de fiches bien formées, 58 % d'exactitude,
  12 % de sous-triage (mono-tour).
- ❌ Multi-tour interactif : comportement à corriger (priorité 1).
- ⏭️ DPO, automatisation, déploiement.
