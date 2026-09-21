# 14 — Dataset multi-tours généré par LLM 27B

**Date** : 21/09/2026 · **Statut** : ✅ (dataset final généré, SFT à venir)

## Objectif

Corriger le **mode interactif** du modèle de triage. Le SFT actuel (13) était bon en
mono-tour (95 % de fiches) mais cassé en multi-tours : fiche prématurée, répétitions,
hallucinations. Diagnostic racine : les 300 dialogues d'origine commençaient tous par
une ouverture **vague** (« je ne me sens pas bien »), donc le modèle n'avait jamais vu
un patient arriver avec un **symptôme précis** — il retombait sur le pattern vignette
(symptôme → fiche immédiate).

## Approche

- **Générateur** : `Qwen3.8-27B` (Ollama, `hf.co/unsloth/Qwen3.8-27B-GGUF:UD-Q4_K_M`)
  — uniquement pour **générer**, pas pour entraîner (on entraîne toujours Qwen3-1.7B).
- **Symptôme dès l'ouverture**, puis une info révélée par tour (durée, intensité,
  température, antécédents, traitements), dans un ordre varié.
- **Fiche déterministe** : le niveau est **toujours calculé** par la règle FRENCH
  (`fiche_block`), jamais par le LLM. Le LLM ne rédige que le langage naturel
  (`<think>`, questions, réponses, explication).
- **Balisage garanti par le code** : le 27B écrit son raisonnement dans un **champ JSON
  `think` séparé** (texte libre), et le code l'enveloppe dans `<think>` + injecte la
  `<FICHE>` programmatique.
- **Vérification + repli** : structure des tours, absence de fuite (ECG/SpO₂/PAS…),
  chiffres préservés. Échec → repli scripté (correct par construction).
- **Symptômes à la 3e personne** (13 règles) : traités par deux voies —
  *conscients* (8) réécrits en « je » par le 27B ; *inconscients* (5, arrêt cardiaque,
  coma) générés en **mode tiers** (un proche parle, infos patient au « il »).

## Résultats finaux

| Métrique | Valeur |
|---|---|
| Dialogues | **1074** |
| Corrects (fiche + niveau + `<think>`) | **1074/1074 (100 %)** |
| Générés par le 27B | **917 (85 %)** |
| Repli scripté | 157 (15 %) |
| Mode tiers (proche parle) | 124 |
| Tours patient | 4-5 en majorité |
| Niveaux | **équilibrés** : `{1: 219, 2: 218, 3: 210, 4: 208, 5: 219}` |

Fichier : `data/processed/triage/sft_multiturn_full.jsonl` (source
`synthetic_triage_multiturn_llm`).

## Pièges techniques (importants)

1. **`<think>` est un token spécial du 27B** : lui demander d'émettre la balise le
   faisait s'arrêter net (stop token) ou dériver (`<thought>`, `<br/>`). → raisonnement
   dans un champ JSON séparé, le code pose les balises.
2. **Modèle à raisonnement** : sans `think:false`, le 27B restait bloqué en phase de
   « thinking » (content vide). → `think:false` + retrait du stop `<think>` via
   `options.stop`.
3. **Symptômes 3e personne** : 13 règles décrivent un patient au « il ». 8 sont des
   patients conscients (réécrits en « je » par le 27B) ; 5 sont des inconscients
   (générés en mode « tiers », un proche parle).
4. **Outage disque/Ollama** : le SSD des modèles a été démonté pendant le batch 2
   (~899 replis « 404 »). → Résolu, rattrapage via `scripts/regenerate_fallbacks.py`.

## Enrichissement des fact sheets

`derive_patient_observable` enrichi : durée quasi systématique (adaptée au niveau),
douleur détectée par le vocabulaire, antécédents/traitements garantis ≥ 1 pour les
adultes, pools pédiatriques dédiés.

## Limites connues

- **15 % de repli scripté** (rigide mais correct) — dû aux échecs transitoires du 27B
  (`vide` ~10 %).
- **Contradiction « sans douleur » → douleur 6/10** : le motif « sans douleur » contient
  « douleur » → échelle générée à tort. Cas rares, à corriger (détection négative).
- **`summary`/`recommendation` programmatiques** (pas naturalisés) — choix assumé : la
  fiche JSON est un tool call, l'exactitude prime sur la variété.

## Prochaines étapes

1. Corriger la détection de douleur (ignorer « sans douleur »/« indolore »).
2. **SFT** sur ce dataset (remplace/complete les 300 dialogues d'origine).
3. Ré-évaluer le mode interactif (questionnement) + le mono-tour.
