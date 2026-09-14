# Résultats du SFT (LoRA) — Qwen3-1.7B

**Objet** : synthèse des runs d'entraînement SFT, de leurs résultats sur les jeux gold
(traduits en français), et des découvertes méthodologiques.
**Date** : 2026-09-14
**Moteur** : Unsloth (QLoRA 4-bit, r=16, batch effectif 8, LR 2e-4 cosine).

---

## 1. Parcours des runs

| Run | Données | Epochs | Leçon |
|---|---|---|---|
| **Pilot** | 200 vignettes triage | 1 | pipeline OK, format appris |
| **Full** | 4 500 base + 600 triage ×3 | 2 | ❌ le format est **dilué** par la base |
| **Stage1** | 4 500 base (Q&A médicaux) | 1 | connaissance médicale |
| **Stage2** | 600 triage (reprise stage1) | 2 | format + comportement triage |
| **Stage3** | 600 triage (reprise stage2, LR 5e-5) | +1 | affinage (« cool down ») |

---

## 2. Résultats finaux (gold français, zero-shot)

| Métrique | Base (pré-SFT) | Stage2 | **Stage3 (final)** |
|---|---|---|---|
| Parse `[FICHE]` | 0 % | 92 % | **95 %** (83/87) |
| Exactitude | — | 55 % | **58 %** |
| Sous-triage | — | 11 % | 12 % |
| Sous-triage binaire (urgences) | — | 16 % | 16,7 % |

> Le few-shot n'est plus pertinent **après** SFT : il dégrade le modèle (cf. §4).

---

## 3. Découvertes méthodologiques

### 3.1 Le mélange base+triage dilue le format
Le run « Full » (base + triage mélangés) a fait chuter le parse à **25 %** : les 4 500
Q&A de base (sans system prompt, sans fiche) ont « noyé » le format `[FICHE]`.
→ **Solution : entraînement en 2 étapes** (base → triage), qui remonte le parse à 92 %.

### 3.2 La traduction FR des gold a débloqué Ramaswamy
Avant traduction, Ramaswamy était à **0 % de parse** (prompts anglais structurés).
Après traduction en **français naturel**, il parse normalement. Le 0 % était un artefact
de langue/format, pas un défaut du modèle.

### 3.3 Zero-shot > few-shot après SFT
Avant SFT, le few-shot était *nécessaire* (zero-shot = 0 %). **Après** SFT, c'est
l'inverse : le few-shot (2 exemples) **ancre le modèle sur « 5 »** (bénin) → 51 % de
sous-triage. Le zero-shot prédit un spectre sain de priorités.
→ **Le protocole d'éval et de prod est désormais zero-shot.**

### 3.4 Poursuite +1 epoch : gain léger + drift de format
Continuer +1 epoch à LR 5e-5 a apporté **+3 points d'exactitude** (55 % → 58 %), mais a
fait **dériver** `[FICHE]` → `<FICHE>` (chevrons, par mimétisme avec `<think>`).
→ Le parser a été rendu **tolérant** (accepte les deux) ; en prod aussi.

### 3.5 Plateau SFT atteint
Le SFT a atteint son plateau : ~58 % d'exactitude, ~12 % de sous-triage. Les epochs
supplémentaires n'apportent plus de gain significatif.

---

## 4. Ce qui reste

| Problème | Valeur actuelle | Outil |
|---|---|---|
| **Sous-triage** | 12 % (dont ~3 urgences/18 ratées) | **DPO** (chosen prudent / rejected sous-triage) |
| Exactitude | 58 % | amélioration marginale (DPO ou données) |

**Prochaine étape : DPO** — générer les paires de préférence triage (~400) + DPO
Unsloth (UltraMedical + triage), puis re-évaluer en zero-shot pour faire chuter le
sous-triage vers ~0.
