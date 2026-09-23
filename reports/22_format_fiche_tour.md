# 22 — Format « fiche à chaque tour » + patient naturalisé (SFT v3)

**Date** : 23/09/2026 · **Statut** : ✅ testé (4B OK, 1.7B hors capacité)

## Objectif

Résoudre le conflit qui cassait le parse au DPO questionnement, et rendre le patient
réaliste :
1. **Fiche à chaque tour** de l'agent (le SI gère création/mise à jour, pas de
   `update`/`finalize`). Fiche incomplète (plage + `missing_info`) aux tours de
   question, fiche complète au dernier tour.
2. **Patient naturalisé** (27B) : langage courant, sans jargon, sans style télégraphique.

## Modifications des données

- `insert_incomplete_fiche.py` : injecte la fiche incomplète à chaque tour intermédiaire
  (1074 dialogues). `missing_info` dérivée des questions réelles de l'agent.
- `rewrite_patient_speech.py` : 27B réécrit les tours patient (876/1074 réécrits,
  198 replis « chiffre perdu » → original conservé). ~2,6 % de jargon restant.
- `SYSTEM_PROMPT` mis à jour (« à chaque tour, tu émets une fiche ») + propagé partout.
- Vignettes : ajout d'un `<think>` déterministe + standardisation `<FICHE>`.

## Résultats SFT (gold enrichi 135 cas)

| Modèle | Parse | Exactitude | Sous-triage | Binaire sécurité |
|---|---|---|---|---|
| **1.7B (4 epochs)** | 81 % ❌ | 65,1 % | 6,4 % (dist 3,1) | 5,7 % |
| **4B (2 epochs)** | **100 %** ✅ | 76,3 % | 5,9 % | 10,6 % |

Comparaison ancien format :
| Modèle (ancien format) | Parse | Exactitude | Sous-triage | Binaire |
|---|---|---|---|---|
| 1.7B `lora_stage2_v2` | 97,7 % | 71,5 % | ~10 % | 9,5 % |
| 4B `qwen35_stage2` | 100 % | 79,3 % | 3,0 % | **4,5 %** |

## Conclusion

- **Le 4B gère le format à 100 % de parse** (plus aucun JSON malformé) — objectif atteint
  côté robustesse. Mais les métriques **reculent un peu** (exactitude −3, binaire +6) :
  le format « fiche à chaque tour » + patient naturalisé coûte légèrement au 4B.
- **Le 1.7B est hors capacité** sur ce format : JSON de fiche malformé (19 % d'échecs),
  priorités hallucinées (10, 15). Malgré 4 epochs, ça ne rentre pas.

## Prochaine étape

DPO « qualité » sur le 4B (paires régénérées en format v3 : fiche incomplète des deux
côtés sur les tours de questionnement), puis éval. Les paires se génèrent en fond.
