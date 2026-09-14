# Plan d'attaque — POC d'agent de triage médical (CHSA)

**Projet** : Fine-tune d'un LLM (Qwen3-1.7B) en agent de triage médical — SFT (LoRA) puis DPO, déploiement vLLM/FastAPI/Docker + CI/CD.
**Statut** : Étape 1 (données de base) terminée. Ce document fige la stratégie pour la suite.
**Date** : 2026-09-08

---

## 1. Contexte et objectif

Le CHSA veut un agent qui, en discutant avec un patient :

1. **collecte ses symptômes** via un questionnaire adaptatif ;
2. **évalue la priorité** (protocole reconnu) ;
3. **explique** clairement son évaluation ;
4. **s'intègre au SI** et garantit la **traçabilité** pour les audits.

Livrables (4 semaines) : dataset bilingue anonymisé, modèle fine-tuné, endpoint déployé, pipeline CI/CD, rapport technique.

---

## 2. État actuel (Étape 1 — terminée)

| Artefact | Contenu |
|---|---|
| Corpus de base standardisé | FrenchMedMCQA (FR), MedQuAD (EN), MediQA (EN) → SFT ; UltraMedical-Preference → DPO |
| Jeux SFT | `sft_train` 4 500 + `sft_val` 500 (2 400 FR / 2 600 EN) |
| Jeu DPO | `dpo_train` 5 000 (UltraMedical, stratifié `label_type`) |
| Évaluation clinique isolée | `clinical_eval` 1 038 (test natifs + réserve MedQuAD) |
| Anonymisation | Presidio (FR/EN) → **0 PII réelle** (faux positifs médicaux documentés dans l'audit) |
| Rapports | 01 EDA, 02 préparation, 03 bilan, 04 protocole FRENCH |

**Constat confirmé (support OpenClassrooms)** : aucun dataset de triage annoté n'est fourni. **Construire notre propre corpus de vignettes synthétiques annotées (protocole FRENCH ou CIMU) + dialogues de questionnaire fait partie de l'évaluation.** MIMIC-IV-ED est optionnel (contraintes d'accès).

---

## 3. Décisions clés (récapitulatif)

| Sujet | Décision |
|---|---|
| Protocole de triage | **FRENCH** (SFMU, 5 niveaux) — les 3 niveaux du brief n'étaient qu'un exemple |
| Niveaux | **5 niveaux** conservés (Tri 1→5) |
| Schéma de métadonnées | défini (source, split, license, lang, topic, question_type, symptoms, medical_history, vital_signs, confidence) ; les champs cliniques sont **vides côté entraînement**, remplis à l'inférence |
| Architecture de l'agent | raisonnement interne `<think>` + **fiche structurée** (format tool_call) |
| Fiche | JSON validé par schéma (Pydantic), stocké pour l'audit (intégration SI simulée) |
| Incertitude | **plage d'urgence + borne prudente + red flags + confiance** |
| Données de triage | servent **SFT (apprendre)** + **DPO (sécuriser)** |
| UltraMedical | conservé pour le DPO (rigueur/sécurité) — n'enseigne pas le ton patient |
| Jeux gold externes | Levine (48), Ramaswamy (78 symptômes seuls), IyàwóBench (200, optionnel) → **évaluation** (mapping plage FRENCH, cf. rapport 08) |
| Entraînement | **Unsloth** (SFT + DPO) — pas TRL brut |
| Inférence / serving | **vLLM** (exigé par le brief) — pas Ollama |
| LoRA en production | **fusionné dans les poids** (modèle unique servi par vLLM) |

---

## 4. Architecture de l'agent (ce qu'on construit)

### 4.1 Deux couches de données
- **Fiche clinique** (vérité terrain, cachée) : motif + signes objectifs + constantes mesurées → **niveau FRENCH vrai**.
- **Patient-observable** : ce que le patient *peut* dire (symptômes, intensité, durée, auto-mesures, antécédents) — jamais PAS/SpO₂/ECG/GCS.

### 4.2 Déroulé d'une interaction
```
<think>  (interne, masqué au patient, conservé pour l'audit)
  faits connus · red flags écartés/confirmés · plage [borne_urgente..borne_bénigne]
  · prochaine info à obtenir
</think>
→ question en langage naturel au patient
...
<think> …raisonnement complet… </think>
→ fiche (tool_call) + explication naturelle au patient
```

### 4.3 Fiche de triage (sortie structurée)
```json
{"name": "finalize_triage", "arguments": {
  "priority": 2, "priority_range": [2, 2],
  "red_flags": ["suspicion SCA"], "missing_info": ["ECG"],
  "confidence": 0.8, "summary": "…", "recommendation": "…"
}}
```

### 4.4 Gestion de l'incertitude
- Niveau = **plage** `[borne_urgente … borne_bénigne]` compatible avec les faits connus.
- On **retient la borne prudente** (principe de précaution : le sous-triage est dangereux, le sur-triage est coûteux mais sûr).
- Un **red flag non écarté = traité comme présent**.
- `confidence` ∝ étroitesse de la plage.

---

## 5. Référentiel d'annotation : FRENCH

Voir `reports/04_protocole_french.md` (grille SFMU : 5 niveaux, constantes adultes/pédiatrie, motifs de recours par catégorie).

- Le niveau est **calculé** (motif + signes + constantes → niveau), **jamais deviné**.
- Chaque critère est marqué **`observable`** (patient-reportable) ou non, avec un **proxy verbal** :
  - ECG typique SCA → « douleur serrante irradiant bras/mâchoire + sueurs »
  - PAS ≤ 90 → « malaise au lever »
  - SpO₂ 86–90 → « je ne peux pas finir mes phrases »
  - GCS/confusion → ⚠️ non auto-évaluable (via un proche uniquement)

---

## 6. Les datasets à construire

| Dataset | Rôle | Format | Volume cible |
|---|---|---|---|
| **Vignettes de triage** | SFT | fact sheet → vignette → fiche | ~800 |
| **Dialogues de questionnaire** | SFT | multi-tours `<think>` + questions → fiche | ~300 |
| **Paires de préférence triage** | DPO | `chosen` (prudent/juste) vs `rejected` (sous-triage/bénin) | ~500 |

Équilibrage : niveaux FRENCH 1–5 × catégories (cardio, respi, neuro, digestif, trauma, psy, pédiatrie…). Inclure des **patients vagues / « je ne sais pas »** (exercice de prudence).

---

## 7. Plan de construction (phases)

### Phase 0 — Encoder FRENCH + schémas *(aucun LLM)*
1. `french_rules.json` : motifs, conditions, constantes → niveau, flag `observable` + proxy.
2. Schémas des 4 objets : `fact_sheet`, `patient_observable`, `dialogue`, `fiche`.
3. Table de mapping FRENCH → patient-reportable.
→ **C'est le socle ; on commence ici.**

### Phase 1 — Générateur déterministe de fact sheets
Échantillonne les règles → `fact_sheet` (niveau **correct par construction**).

### Phase 2 — Pilot sur 2 catégories *(cardio + fièvre)*
~20–50 fact sheets → vignettes générées par LLM local → **validation** (ré-évaluation LLM + revue humaine). Si OK → généraliser.

### Phase 3 — Passage à l'échelle + dialogues + DPO
Génération complète, dialogues, paires chosen/rejected, validation couches 1–4.

**Validation (4 couches)** : ① par construction (déterministe) · ② ré-évaluation croisée LLM · ③ benchmark sur jeux gold · ④ revue humaine par échantillon.

---

## 8. Plan d'entraînement

| Étape | Semaine | Données | Ce qu'elle apprend |
|---|---|---|---|
| **SFT** (LoRA) | 2 | MedQuAD + FrenchMedMCQA + MediQA (~5 000) | connaissance médicale + suivi d'instructions |
|  |  | **+ triage SFT** (vignettes + dialogues) | **le comportement de triage** |
| **DPO** | 3 | UltraMedical (~5 000) | rigueur / qualité / sécurité |
|  |  | **+ triage DPO** (chosen/rejected) | **prudence du triage** |

*Note : le DPO ne peut pas apprendre un comportement inconnu — d'où le triage d'abord en SFT, ensuite en DPO.*

*Outillage : **Unsloth** pour SFT + DPO (2× plus rapide, −50/70 % de VRAM, intégré TRL). Le LoRA final est **fusionné** dans les poids avant serving vLLM.*

---

## 9. Évaluation

**Moteur** : **vLLM** (le même qu'en production — pas Ollama), exposé en API
OpenAI-compatible. Le harness (`src/triage_agent/eval/harness.py`) lance le modèle, parse
la fiche `[FICHE]` et compare à la plage gold.

**3 étages de granularité** (cf. `reports/08_analyse_jeux_gold.md`) :
1. **5 niveaux** sur notre set synthétique (matrice 5×5, sous-triage pondéré par la distance) ;
2. **4/3 niveaux** sur les jeux gold (Ramaswamy, Levine) après mapping plage FRENCH ;
3. **binaire de sécurité** (« urgences maintenant vs pas ») — synthèse du sous-triage.

**Métrique principale** : pénaliser le **sous-triage** bien plus que le sur-triage
(distance pondérée). **Baseline** : Qwen3-1.7B de base servi par vLLM, avant fine-tune.

**Critères de qualité** (repris des `feedback` UltraMedical) : pertinence, véracité,
clarté/profondeur, **sécurité**. Contrôles : hallucinations, recommandations dangereuses
(exigé par le brief).

---

## 10. Jalons (4 semaines)

| Semaine | Jalon |
|---|---|
| 1 | ✅ Corpus de base + EDA + anonymisation + jeux SFT/DPO/éval |
| 1.5 → 2 | **Construction du dataset de triage** (Phases 0→3) + méthodologie documentée |
| 2 | SFT LoRA **Unsloth** (petits runs → montée en charge) + évaluation via vLLM |
| 3 | DPO (UltraMedical + triage) + itérations + évaluation |
| 4 | **Fusion LoRA** + déploiement vLLM/FastAPI/Docker + CI/CD + rapport final |

---

## 11. Risques et questions ouvertes

| Risque / question | Statut |
|---|---|
| Modèle local pour la génération | ✅ résolu : Leila_fast (Ollama) — génération uniquement |
| Accès aux jeux gold | ✅ résolu : Ramaswamy (GitHub), Levine (GitHub), IyàwóBench (GitHub) ; psy ❌ indisponible |
| LoRA Qwen récents (bugs vLLM version-sensibles) | ⚠️ atténué : **fusion** du LoRA avant serving |
| Pas de clinicien dans la boucle | compenser par jeux gold + revue humaine + documentation |
| Sur/sous-triage : calibration de la borne prudente | à itérer en validation |
| Mapping 4 niveaux → FRENCH 5 | choix méthodologique à documenter |
| Taille DPO triage (500 ?) | ajustable |

---

## 12. Prochaine action immédiate

1. **Installer vLLM** (venv dédié ou Docker) + `vllm serve Qwen/Qwen3-1.7B` — pas encore FastAPI.
2. **Baseline** : lancer le harness sur Qwen3-1.7B de base (chiffre « avant fine-tune »).
3. **DPO triage** (optionnel, selon les résultats d'éval) + **SFT Unsloth** (semaine 2).
