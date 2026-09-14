# Protocole FRENCH — référence pour l'annotation du dataset de triage

**Source** : SFMU (Société Française de Médecine d'Urgence) — *FRENCH Triage V1, mars 2018*
[grille officielle](https://www.sfmu.org/upload/referentielsSFMU/FRENCH_A4_v181003.pdf) — Sandra BERNARD.

> **FRENCH** = *FRench Emergency Nurses Classification in Hospital*. Échelle nationale
> de triage à l'accueil des urgences, utilisée par l'infirmier organisateur de
> l'accueil (IOA). Elle couvre **adultes et enfants**, sur **5 niveaux** (Tri 1 → 5).

---

## 1. Les 5 niveaux (description générale)

| Niveau | Situation clinique | Délai d'intervention | Risque d'aggravation | Survie attendue |
|---|---|---|---|---|
| **Tri 1** | **Détresse vitale majeure** — support d'une ou plusieurs fonctions vitales | **Sans délai** (IDE + médecin, dans les minutes) | ++++ (dans les min) | ≥ 90 % |
| **Tri 2** | **Atteinte patente d'un organe** ou lésion traumatique sévère (ou symptôme sévère justifiant une action < 20 min) | IDE < 10 min, **médecin < 20 min** (dans l'heure) | +++ | ≥ 80 % |
| **Tri 3A** | **Atteinte potentielle d'un organe** ou lésion traumatique instable, **ou** comorbidités lourdes, **ou** patient adressé par un médecin le jour même | **médecin < 60 min** (dans les 24 h) | + | ≥ 30 % |
| **Tri 3B** | Atteinte fonctionnelle ou lésionnelle **stable**, patient **sans** comorbidité lourde | **médecin < 90 min** | + | ≥ 50 % |
| **Tri 4** | Atteinte fonctionnelle ou lésionnelle stable, **pas** de risque d'aggravation, acte diagnostique/thérapeutique limité | **médecin < 120 min** (puis IDE si besoin) | Non | — |
| **Tri 5** | **Pas d'atteinte** fonctionnelle ou lésionnelle évidente, pas d'acte prévisible | **médecin < 240 min** (ou maison médicale de garde) | Non | — |

---

## 2. Modulation des constantes (adultes)

Seuils vitaux qui déclenchent un niveau (en l'absence d'autre motif).

| Constante | Tri 1 | Tri 2 | Tri 3 |
|---|---|---|---|
| **PAS** (mmHg) | < 70 | 70–90, ou 90–100 avec FC > 100 | > 90 |
| **FC** (/min) | > 180 ou < 40 | 130–180 | < 130 |
| **SpO₂** (%) | < 86 | 86–90 | > 90 |
| **FR** (/min) | > 40 | 30–40 | — |
| **GCS** | ≤ 8 | 9–13 | 14 |
| **Glycémie** | hyperglycémie ≥ 20 mmol/l + cétose + trouble de conscience | hyperglycémie ≥ 20 mmol/l + cétose | — |

*Rappels glycémiques (motif « DIVERS ») :* hypoglycémie avec coma (GCS ≤ 8) → Tri 1/2 ;
hyperglycémie ≥ 20 mmol/l ou cétose positive → Tri 3B ; cétose négative → Tri 4.

---

## 3. Constantes pédiatriques (repères)

**Normalité selon l'âge :**

| Âge | PAS (mmHg) | FC (/min) | FR (/min) |
|---|---|---|---|
| < 1 mois | ≥ 50 | 130 ± 45 | 30 ± 15 |
| 1–6 mois | 85 ± 30 | 130 ± 45 | 30 ± 15 |
| 1–2 ans | 100 ± 25 | 110 ± 40 | 25 ± 10 |
| 2–4 ans | 100 ± 20 | 105 ± 35 | 25 ± 10 |
| 4–10 ans | 110 ± 15 | 95 ± 35 | 25 ± 10 |
| 10–14 ans | 115 ± 15 | 85 ± 30 | 20 ± 5 |

**Seuils d'alerte :** hypotension (PAS < 50/65/70/80 mmHg selon âge), tachycardie
(FC > 180/160/130/120), polypnée (FR > 60/40/30/20) — voir grille SFMU p. 9.

---

## 4. Motifs de recours et niveaux associés

Le cœur de l'échelle : **un motif de recours + des signes → un niveau**. Résumé des
principaux motifs par catégorie (extrait de la grille SFMU).

### 4.1 Cardio-circulatoire
| Motif | Tri 1 | Tri 2 | Tri 3A | Tri 3B | Tri 4/5 |
|---|---|---|---|---|---|
| Arrêt cardiorespiratoire | ✅ | | | | |
| Hypotension / collapsus | PAS ≤ 70 | PAS ≤ 90 (ou ≤ 100 + FC > 100) | | | |
| Douleur thoracique / SCA | | ECG typique SCA | ECG non typique | ECG N + douleur typique | ECG N + douleur atypique |
| Malaise | | | | avec/sans prodrome | |
| Tachycardie | | FC ≥ 180 | FC ≥ 130 (ou ≥ 110 + TAS < 110) | FC ≥ 110 | |
| Bradycardie | | FC ≤ 40 | | 40–50 + signes associés | 40–50 isolée |
| Dyspnée / IC | | détresse respi, FR ≥ 40 | | FR 30–40, SpO₂ 86–90 | |
| Ischémie de membre | | durée ≤ 24 h + cyanose/déficit | | durée ≥ 24 h | |
| Phlébite | | | | signes francs / proximal | signes modérés |
| HTA | | | TAS ≥ 220 (ou ≥ 180 + signes) | TAS ≥ 180 | |
| Palpitations | | | FC ≥ 180 | FC ≥ 140 + malaise | |

### 4.2 Infectiologie
| Motif | Tri 2 | Tri 3A | Tri 4 | Tri 5 |
|---|---|---|---|---|
| Fièvre | | | t ≥ 40 °C ou ≤ 35,2 °C ou confusion/céphalée/purpura | signes AEG, hypotension |
| Exposition maladie contagieuse | | | risque vital de contage (méningite…) | sans risque vital |

### 4.3 Abdominal / digestif
| Motif | Tri 2 | Tri 3A | Tri 3B | Tri 4 | Tri 5 |
|---|---|---|---|---|---|
| Hématémèse | abondante | | strié de sang | | |
| Melaena / rectorragies | rectorragie abondante | | selles souillées | | |
| Douleur abdominale | | sévère + signes généraux | régressive / indolore | | |
| Hernie / occlusion | | | | douleur sévère / occlusion | |
| Vomissements | | | | | enfant ≤ 2 ans / douleur / abondants |
| Diarrhée | | | | | enfant ≤ 2 ans abondante |
| Rétention urinaire / anurie | | | douleur intense / agitation | | |
| Douleur de bourse / torsion | | douleur intense / suspicion torsion | avis référent | | |
| Hématurie | | | saignement abondant actif | | |
| Constipation | | | | | triade occlusion |

### 4.4 Neurologie
| Motif | Tri 2 | Tri 3A | Tri 3B | Tri 4 |
|---|---|---|---|---|
| Déficit moteur / AVC | délai ≤ 3–4 h 30 (avis MAO/MCO) | | délai ≥ 12 h | |
| Altération conscience / coma | avis référent | | | |
| Convulsions | | crises multiples ou en cours | confusion, déficit, fièvre | récupération complète |
| Céphalée | | inhabituelle (brutale, intense, fièvre) | habituelle / migraine | |
| Vertiges | | signes neuro associés | | troubles anciens stables |
| Confusion | | fièvre | régressive / indolore | |

### 4.5 Respiratoire
| Motif | Tri 2 | Tri 3A | Tri 3B | Tri 4/5 |
|---|---|---|---|---|
| Dyspnée avec sifflement | | | sifflement sans dyspnée | |
| Asthme / BPCO | | détresse respi / DEP ≤ 200 / tirage / orthopnée | | DEP ≥ 300 |
| Hémoptysie | | | détresse respi / abondante | |
| Embolie / pneumopathie / pneumothorax | | | FR 30–40, SpO₂ 86–90 | |
| Corps étranger voies aériennes | | détresse respi | dyspnée à la parole / tirage | pas de dyspnée |

### 4.6 Traumatologie
| Motif | Tri 1 | Tri 2 | Tri 3A | Tri 3B | Tri 4/5 |
|---|---|---|---|---|---|
| Amputation | ✅ | | | | |
| Trauma abdomen/thorax/cervical | | pénétrant / haute vélocité | | | faible vélocité sans signes |
| Trauma crânien | | coma GCS ≤ 8 | GCS ≤ 15 + déficit neuro | | perte de connaissance, plaie |
| Brûlure | | | étendue / main / visage | ≤ 24 mois peu étendue | consultation tardive |
| Plaie | | | délabrante / saignement actif | large / complexe / main | superficielle |
| Agression sexuelle / sévices | | avis référent | | | |

### 4.7 Psychiatrie
| Motif | Tri 2 | Tri 3A | Tri 3B | Tri 4 |
|---|---|---|---|---|
| Idée / comportement suicidaire | avis référent | | | |
| Troubles du comportement | | agitation / violence / délire | | |
| Anxiété / dépression | | | | attaque de panique / demande hospitalisation |

### 4.8 Autres (extraits)
- **Gynéco-obstétrique** : accouchement imminent → Tri 1 ; métrorragies/douleur 1er–2e trimestre → Tri 3A ; grossesse 3e trimestre (métrorragies/HTA/perte liquide) → Tri 3A.
- **Ophtalmo** : brûlure chimique / corps étranger → Tri 3A ; œil rouge → Tri 5.
- **ORL** : épistaxis abondante active → Tri 3B ; surdité brutale → Tri 4 ; otite → Tri 5.
- **Peau** : ecchymose spontanée → Tri 3B ; abcès fébrile → Tri 4 ; corps étranger sous peau → Tri 5.
- **Divers** : hypothermie ≤ 32 °C → Tri 2 ; coup de chaleur avec coma → Tri 3A ; renouvellement d'ordonnance → Tri 5 ; certificat administratif → Tri 5.
- **Pédiatrie ≤ 2 ans** : fièvre ≤ 3 mois → Tri 2 ; convulsion hyperthermique récidivante → Tri 3B ; diarrhée/vomissements avec perte de poids ≥ 10 % → Tri 3A.

---

## 5. Usage pour notre dataset

Cette grille sert de **référentiel d'annotation** :

1. **Détermination du niveau** : motif de recours + signes + constantes → niveau
   (règles **déterministes** ci-dessus). Le niveau est donc **calculé**, pas deviné.
2. **Champs de la fiche** : `symptoms` (motif + signes), `medical_history`
   (comorbidités), `vital_signs` (constantes), `priority` (Tri 1→5),
   `justification` (règle FRENCH appliquée), `confidence`.
3. **Mapping vers les 3 niveaux du brief** (si nécessaire) : Tri 1–2 = urgence
   maximale, Tri 3 = modérée, Tri 4–5 = différée.
