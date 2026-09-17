# TODO — prochaines itérations

> Priorités définies avec le mentor. Mise à jour au fil des sessions.

## 🔴 Priorité 1 — Corriger le multi-tour (questionnement)

Le mode interactif est **sous-appris** (fiche prématurée, réponses hallucinées, boucles) :
le dataset contient autant de vignettes (fiche immédiate) que de dialogues (question →
fiche), et le comportement « vignette » domine.

- [ ] **Étendre le dataset synthétique dédié au multi-tour** (×3-4 dialogues, varier les
      séquences de questions, cas de patients vagues).
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

- [ ] **MLflow** (recommandé) : log des hyperparamètres, métriques et artefacts dans
      `train_sft.py`, UI locale pour comparer/versionner les runs.
- [ ] Évaluer **Unsloth Studio** : UI no-code locale + monitoring live (loss, grad norm,
      GPU). Complémentaire, mais moins adapté à notre pipeline scripté.

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
