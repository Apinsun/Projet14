# 14 — Dataset multi-tours généré par LLM 27B

**Date** : 21/09/2026 · **Statut** : ✅ (dataset généré, SFT à venir)

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
  `<FICHE>` programmatique. (Voir pièges ci-dessous.)
- **Vérification + repli** : structure des tours, absence de fuite (ECG/SpO₂/PAS…),
  chiffres préservés. Échec → repli scripté (correct par construction).

## Résultats

| Métrique | Valeur |
|---|---|
| Dialogues finaux | **893** (après filtrage 3e personne) |
| Corrects (fiche + niveau + `<think>`) | **893/893 (100 %)** |
| Générés par le 27B | **760 (85 %)** |
| Repli scripté | 133 (15 %) |
| Tours patient | 4-5 tours en majorité (505 à 4, 348 à 5) |

Fichier : `data/processed/triage/sft_multiturn_llm.jsonl` (source
`synthetic_triage_multiturn_llm`).

Répartition par niveau : `{1: 95, 2: 176, 3: 200, 4: 203, 5: 219}` — **déséquilibre sur
les niveaux 1-2** (voir limites).

## Pièges techniques (importants)

1. **`<think>` est un token spécial du 27B** : lui demander d'émettre la balise le
   faisait s'arrêter net (stop token) ou dériver (`<thought>`, `<br/>`). → Le 27B écrit
   le raisonnement dans un **champ JSON séparé**, le code pose les balises.
2. **Modèle à raisonnement** : sans `think:false`, le 27B restait bloqué en phase de
   « thinking » (content vide). → `think:false` + retrait du stop `<think>` via
   `options.stop`.
3. **Symptômes à la 3e personne** : 7 règles (`il ne respire plus`, `il ne se réveille
   pas`…) décrivent un patient inconscient/tiers → incohérent en dialogue 1re personne.
   → Exclus (filtre `is_third_person_symptom`), 196 cas retirés.
4. **Outage disque/Ollama** : le SSD des modèles a été démonté pendant le batch 2
   (~899 replis « 404 »). → Résolu par l'utilisateur, rattrapage via
   `scripts/regenerate_fallbacks.py` (631 régénérés).

## Enrichissement des fact sheets

`derive_patient_observable` enrichi pour garantir des dialogues riches :
- **durée quasi systématique**, adaptée au niveau (minutes → semaines) ;
- **douleur détectée par le vocabulaire** (colique, torsion, brûlure…) et non le seul
  mot « douleur » ;
- **antécédents/traitements garantis ≥ 1** pour les adultes ;
- **pools pédiatriques dédiés** (un bébé n'a plus « tabagisme actif » en antécédent).

## Limites connues

- **Déséquilibre niveau 1-2** : le filtrage 3e personne a surtout vidé le niveau 1
  (95 vs 219). À rééquilibrer (générer plus de niveau 1-2 en 1re personne).
- **15 % de repli scripté** (rigide mais correct) — dû aux échecs transitoires du 27B
  (`vide` ~10 %).
- **Contradiction « sans douleur » → douleur 6/10** : le motif « sans douleur » contient
  « douleur » → échelle générée à tort. Cas rares, à corriger (détection négative).
- **`summary`/`recommendation` programmatiques** (pas naturalisés) — choix assumé :
  la fiche JSON est un tool call, l'exactitude prime sur la variété.

## Prochaines étapes

1. Rééquilibrer les niveaux 1-2 (génération ciblée).
2. Corriger la détection de douleur (ignorer « sans douleur »/« indolore »).
3. **SFT** sur ce dataset (remplace/complete les 300 dialogues d'origine).
4. Ré-évaluer le mode interactif (questionnement) + le mono-tour.
