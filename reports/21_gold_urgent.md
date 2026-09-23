# 21 — Gold urgent enrichi (IyàwóBench REFER_NOW)

**Date** : 23/09/2026 · **Statut** : ✅ fait

## Pourquoi

L'analyse du rapport 20 a montré que le gold (87 cas) était **trop petit et trop bénin**
pour trancher l'effet du DPO : **18 cas urgents seulement** → IC95 du binaire sécurité
à **±15–17 pts**. Or c'est LA métrique qui compte (sous-triage des urgences).

## Source : IyàwóBench REFER_NOW

IyàwóBench (200 vignettes de fièvre indifférenciée en soins primaires) avait été écarté
par prudence (« constantes non-observables »). En réalité **les 100 cas REFER_NOW portent
tous un red flag patient-reportable** (convulsions, troubles de la conscience, dyspnée,
tirage, raideur de nuque) → le niveau d'urgence est dérivable **sans** les constantes.

Deux atouts décisifs :
1. **Labels externes** (indépendants de notre génération) → non circulaire, contrairement
   à des cas synthétiques écrits par le même 27B.
2. 100 cas urgents disponibles.

## Méthode

`scripts/build_urgent_gold.py` (déterministe) :
- sélectionne **48** REFER_NOW (échantillon régulier ; `--all` pour les 100) ;
- message patient **français naturel**, sans constantes (3e personne si enfant ou
  conscience/convulsions : « Ma fille de 8 ans a de la fièvre, une raideur de la
  nuque… ») ;
- mapping `REFER_NOW → FRENCH [1, 3]` (`binary_urgent=True`), constantes conservées en
  métadonnées pour traçabilité ;
- sortie : `data/processed/gold/fr/iyawobench_urgent.jsonl` (48 cas).

Le gold d'origine (Levine/Ramaswamy) reste **intact** → comparabilité préservée.

## Résultats (gold enrichi = 87 + 48 = 135, gold_range d'origine + 48 urgents)

### Binaire sécurité — l'IC se resserre de ~3×

| Modèle | Avant (18 urgents) | Après (62–66 urgents) |
|---|---|---|
| SFT 4B | 11,1 % [3,1–32,8] **±14,8** | **4,5 % [1,6–12,5] ±5,5** |
| DPO 4B | 16,7 % [5,8–39,2] **±16,7** | **4,5 % [1,6–12,5] ±5,5** |
| SFT 1.7B | — | 9,5 % [4,4–19,3] ±7,4 |
| DPO 1.7B | — | 8,1 % [3,5–17,5] ±7,0 |

### Exactitude (gold enrichi)

| Modèle | Exactitude | IC95 |
|---|---|---|
| SFT 4B | 79,3 % | [71,7–85,2] |
| DPO 4B | 75,6 % | [67,7–82,0] |
| SFT 1.7B | 71,5 % | [63,3–78,6] |
| DPO 1.7B | 70,2 % | [61,9–77,4] |

**Toutes les IC SFT/DPO se chevauchent** → la conclusion du rapport 20 est **confirmée
sur les deux tailles** : le DPO ne dégrade pas de façon démontrable.

Détail par source : IyàwóBench est très bien traité (4B : 47–48/48 ; 1.7B : ~93–96 %),
Levine/Ramaswamy restent les plus difficiles.

## Limites (à dire franchement)

1. **Contexte différent** : IyàwóBench = fièvre tropicale en soins primaires (paludisme,
   méningite), pas les urgences françaises. Les **signes** sont universels, mais la
   prévalence ne l'est pas.
2. **Cas homogènes** : tous fébriles → l'exactitude sur ce sous-jeu est favorable.
3. **Mapping grossier** : `REFER_NOW → [1, 3]` ne distingue pas FRENCH 1/2/3 ; valide
   pour le **binaire** (urgent / pas urgent), pas pour le niveau fin.
4. 48 cas sur 100 utilisés (`--all` possible) ; le gold reste petit en absolu.

## Conclusion

Le gold enrichi rend la **métrique de sécurité enfin exploitable** (±5,5 pts au lieu de
±15). Il confirme que **SFT ≈ DPO** et que le 4B sous-trie ~2× moins que le 1.7B
(4,5 % vs 8–9,5 %). Les chiffres de sécurité à retenir pour le livrable sont désormais
**ceux du gold enrichi**.
