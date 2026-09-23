# 23 — DPO v3 sur format « fiche à chaque tour » (4B)

**Date** : 23/09/2026 · **Statut** : ✅ testé

## Résultat

| Modèle | Parse | Exactitude | Sous-triage | Binaire sécurité |
|---|---|---|---|---|
| SFT 4B (nouveau format) | 100 % | 76,3 % | 5,9 % | 10,6 % |
| **DPO 4B (v3, 300 paires)** | **100 %** | 76,3 % | 5,9 % | 10,6 % |

→ DPO **identique** au SFT (pas de dégradation, pas de gain sur le gold). Attendue :
le DPO apprend la forme (politesse/explication), pas le niveau.

## Succès

**Le DPO questionnement ne casse plus le parse** (100 %, 0 échec) — avec l'ancien
format il s'effondrait à 25 %. Le « fiche à chaque tour » a débloqué le DPO.

## ⚠️ Régression du binaire sécurité (sur les 18 urgents d'origine)

| | Ancien format 4B | Nouveau format 4B |
|---|---|---|
| Binaire (18 urgents) | 11,1 % (2 ratés) | **27,8 % (5 ratés)** |

Les 3 nouveaux cas urgents sous-triés (gold [1,3] → prédit 4) : **levine 2, 9, 10**.
(levine 7 reste raté dans les deux ; F13 est corrigé dans le nouveau.)

## Conclusion honnête

Le format « fiche à chaque tour » + patient naturalisé :
- ✅ parse 100 % partout (SFT et DPO) ;
- ✅ DPO questionnement possible (objectif atteint) ;
- ❌ **coûte de la sécurité** : 3 cas urgents de plus sous-triés sur les 18 d'origine
  (exactitude −3 pts aussi).

Le patient naturalisé ou le format fiche ont déréglé le calibrage sur certains cas
urgents. À trancher : garder le format robuste avec ce coût, ou revenir à l'ancien
format (79,3 % / 4,5 % binaire) et renoncer au DPO questionnement.
