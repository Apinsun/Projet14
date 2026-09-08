# Étape 1 — Préparation des données : revue post-traitement

Pipeline exécuté : **standardisation → nettoyage → sélection SFT/DPO → anonymisation Presidio → découpage train/val**.

## Synthèse des jeux finaux

| Jeu | Nombre d'enregistrements |
|---|---|
| `sft_train` | 4,500 |
| `sft_val` | 500 |
| `dpo_train` | 5,000 |
| `clinical_eval` | 1,038 |

## Cheminement par source

### FrenchMedMCQA (`frenchmedmcqa`)

| Étape | Volume |
|---|---|
| Brut téléchargé | 3,105 |
| Standardisé (JSONL nettoyé) | 3,105 |
| Sélectionné (SFT + DPO + éval) | 3,022 |
| **Enregistrements avec PII masquée** | **0** |
| Occurrences `PERSON` détectées (non masquées, éponymes) | 3246 |

Entités détectées mais non masquées (faux positifs médicaux) : `LOCATION` (12575), `PERSON` (3246), `ORGANIZATION` (2840)

**Exemple (post-traitement) :**

- `user`      : Parmi les propositions suivantes concernant les Herpesviridae, lesquelles sont exactes ? (A) Ils établissent tous leur latence dans le système nerveux central (B) Les symptômes observés au cours des réactivations virales…
- `assistant` : Les réponses correctes sont : Ils présentent tous la même structure en microscopie électronique, Ils sont fréquemment impliqués dans les complications infectieuses des états d’immuno-dépression, Leur multiplication fait …

### MedQuAD (`medquad`)

| Étape | Volume |
|---|---|
| Brut téléchargé | 47,441 |
| Réponses non nulles (filtre structurel) | 16 407 |
| Standardisé (JSONL nettoyé) | 14,559 |
| Sélectionné (SFT + DPO + éval) | 2,870 |
| **Enregistrements avec PII masquée** | **0** |
| Occurrences `PERSON` détectées (non masquées, éponymes) | 2389 |

Entités détectées mais non masquées (faux positifs médicaux) : `ORGANIZATION` (6199), `PERSON` (2389), `LOCATION` (1068), `NRP` (932), `DATE_TIME` (898), `URL` (62)

**Exemple (post-traitement) :**

- `user`      : What causes Nephrotic Syndrome in Adults ?
- `assistant` : Nephrotic syndrome can be caused by diseases that affect only the kidneys, such as focal segmental glomerulosclerosis (FSGS) or membranous nephropathy. Diseases that affect only the kidneys are called primary causes of n…

### MediQA (`mediqa`)

| Étape | Volume |
|---|---|
| Brut téléchargé | 383 |
| Standardisé (JSONL nettoyé) | 319 |
| Sélectionné (SFT + DPO + éval) | 146 |
| **Enregistrements avec PII masquée** | **0** |
| Occurrences `PERSON` détectées (non masquées, éponymes) | 181 |

Entités détectées mais non masquées (faux positifs médicaux) : `ORGANIZATION` (289), `DATE_TIME` (188), `PERSON` (181), `LOCATION` (80), `NRP` (18), `URL` (10)

**Exemple (post-traitement) :**

- `user`      : can't find an answer. I was diagnosed with Fibromyalgia with chronic pain along with some other things and my blood work showed that I was missing a chromosone. How would I find out if I have a genetic for of Fibromyalgi…
- `assistant` : Fibromyalgia (What causes it?): Doctors don’t know the exact cause of fibromyalgia. Researchers continue to study fibromyalgia and think the following events may contribute to the cause of the disorder: - Stressful or tr…

### UltraMedical-Preference (`ultramedical_preference`)

| Étape | Volume |
|---|---|
| Brut téléchargé | 112,362 |
| Standardisé (JSONL nettoyé) | 67,429 |
| Sélectionné (SFT + DPO + éval) | 5,000 |
| **Enregistrements avec PII masquée** | **0** |
| Occurrences `PERSON` détectées (non masquées, éponymes) | 13509 |

Entités détectées mais non masquées (faux positifs médicaux) : `ORGANIZATION` (35399), `PERSON` (13509), `DATE_TIME` (10901), `LOCATION` (3815), `NRP` (3608), `URL` (415)

**Exemple (post-traitement) :**

- `prompt`   : Explore the 'hydrophobic collapse' hypothesis by elucidating the mechanism through which the initial condensation of a protein's nonpolar core not only precipitates the organization of its secondary and tertiary structur…
- `chosen`   : The hydrophobic collapse hypothesis proposes that the initial condensation of a protein's nonpolar core drives the organization of its secondary and tertiary structures, leading to the stabilization of quaternary structu…
- `rejected` : The "hydrophobic collapse" hypothesis proposes that the initial condensation of a protein's nonpolar core drives the formation of its secondary, a

## Note RGPD

- Les corpus sources sont publics et non nominatifs (QCM de pharmacie, encyclopédies de maladies, questions de patients anonymisées, Q&A synthétique).
- L'analyse Presidio (NER spaCy + recognizers regex) a montré que **toutes les entités détectées sont des faux positifs sur du contenu médical** : `PERSON` = éponymes de maladies, `URL` = « E.coli » / « word.Word », `MEDICAL_LICENSE` = identifiants SNP (`rs1063192`) ou d'essai clinique, `UK_NHS` = numéro de centre antipoison, `PHONE_NUMBER` = valeur de dosage.
- Seuls les identifiants **sans ambiguïté** (`EMAIL_ADDRESS`, `CREDIT_CARD`, `IBAN_CODE`, `IP_ADDRESS`) sont masqués ; aucune occurrence n'a été trouvée, confirmant l'absence de données personnelles dans le corpus final.
- Le détail complet est dans `data/processed/anonymized/audit.json`.
