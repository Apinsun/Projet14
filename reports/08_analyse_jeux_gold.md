# Analyse des jeux gold d'évaluation (triage externe)

**Objet** : localiser, télécharger et évaluer l'utilité des 4 jeux gold prévus pour
l'évaluation du modèle (Levine, Ramaswamy, IyàwóBench, psy).
**Date** : 2026-09-14

---

## 1. Bilan d'accès

| Jeu | Volume | Niveaux | Langue | Accès | Verdict |
|---|---|---|---|---|---|
| **Levine** | 48 | 4 | EN | ✅ GitHub `beamlab-hsph/gpt3-clinical-vignettes` | utilisable |
| **Ramaswamy** | 78 (core) | 4 (A–D) | EN | ✅ GitHub `sreerammarimuthu/AI-triage-benchmark` | **le plus utile** |
| **IyàwóBench** | 200 | 3 | EN/NG | ✅ GitHub `anthoniooladimeji11-coder/iyawobench` | marginal |
| **psy (Weilnhammer)** | 112 | 4 | EN | ❌ **pas de dataset public** (arXiv uniquement) | indisponible |

> Le « psy (112) » (Weilnhammer et al. 2026, triage psychiatrique one-shot) n'a **aucun
> téléchargement public**. Le dataset HF `arnaiztech/llms-mental-health-crisis-benchmark`
> est une **autre tâche** (gestion de crise en santé mentale, 2 000+ conversations) — pas
> un jeu de triage. À ne pas confondre.

---

## 2. Détail des jeux téléchargés

### 2.1 Levine (48 cas)

- **Source** : *The Diagnostic and Triage Accuracy of the GPT-3 AI Model* (Levine et al.).
- **Fichier utile** : `vignettes-2020.tsv` → colonnes `Correct Diagnosis`, `Correct Triage`,
  `Current Problem`, `Additional Details` (vignettes courtes, <60 mots, <6e grade).
- **Distribution des niveaux** (parfaitement équilibrée) :

| Niveau | Effectif |
|---|---|
| Emergent | 12 |
| 1 day | 12 |
| 1 week | 12 |
| Self-care | 12 |

### 2.2 Ramaswamy (78 cas) — le plus substantiel

- **Source** : *Ramaswamy et al., Nature Medicine 2026* (benchmark « Right for the Wrong
  Reasons », BioDMS/VLDB 2026), via `sreerammarimuthu/AI-triage-benchmark`.
- **78 vignettes** rédigées par des cliniciens, **22 domaines** (Psych 16, Cardiac 8,
  Neuro 6, Heme 6, Uro 4, ID 4, GI 4…), 39 « symptômes seuls » + 39 « clinique complète ».
- **Labels triage A–D adjudiqués par 3 médecins** (avec désaccords conservés) :

| Label | Définition | Effectif |
|---|---|---|
| A | surveiller à domicile | 8 |
| B | voir un médecin sous quelques semaines | 8 |
| C | consultation sous 24–48 h | 16 |
| D | urgences maintenant | 12 |
| A/B | (désaccord) | 2 |
| B/C | (désaccord) | 4 |
| C/D | (désaccord) | 28 |

> Les **split labels** (A/B, B/C, C/D) sont un atout : ils reflètent l'incertitude
> clinique réelle, cohérente avec notre principe de « plage d'urgence ».

### 2.3 IyàwóBench (200 cas) — utilité marginale

- **Source** : IyàwóBench v1.0 (Iyawo Health, Nigéria).
- **200 vignettes**, 8 catégories de **fièvres** (paludisme 82, méningite 30, sepsis 28,
  typhoïde 25, pneumonie 35…).
- **3 niveaux** : `REFER_NOW` (100) / `REFER_TODAY` (60) / `TREAT_HERE` (40).
- **Format structuré** (FC, FR, SpO₂, T°, poids, RDT…) — **pas conversationnel** : notre
  agent travaille sur du patient-reportable, il ne peut pas exploiter ces constantes.

---

## 3. Verdict d'utilité (honnête)

1. **Aucun jeu n'est en français ni en FRENCH 5 niveaux** — tous anglais, 3 ou 4 niveaux.
2. **Il n'existe pas de jeu gold public FRENCH en français** → la justesse FRENCH fine
   (5 niveaux) ne sera mesurable **que sur notre set synthétique** (limité : mêmes règles).
3. **Ramaswamy (78)** est le plus précieux : vignettes cliniques réelles, 22 domaines,
   labels de 3 médecins, split labels → bonne validation externe de la capacité de triage.
4. **Levine (48)** : petit mais propre, utile en smoke test complémentaire.
5. **IyàwóBench (200)** : trop éloigné (fièvre tropicale, données structurées, 3 niveaux)
   → **à écarter ou garder en test de robustesse out-of-distribution**.
6. **psy (112)** : indisponible → **abandonné** (ou remplacé par notre catégorie
   psychiatrie synthétique, déjà couverte par FRENCH).

---

## 4. Plan d'évaluation en 3 étages de granularité

Le binaire n'est **pas** la métrique principale : c'est une synthèse de sécurité, en
**complément** de mesures plus fines. Trois étages :

| Étage | Données | Ce qu'on mesure |
|---|---|---|
| **1. Fin (5 niveaux)** | set synthétique (vrai FRENCH 1–5) | matrice 5×5, rappel par niveau, **sous-triage pondéré** |
| **2. Moyen (4/3 niveaux)** | Ramaswamy (A–D), Levine, IyàwóBench | exactitude après mapping 5→4 (ou 3) |
| **3. Binaire** | tous les jeux | « ED maintenant vs pas » — synthèse de sécurité |

### 4.1 Sous-triage pondéré par la distance (mieux que binaire)

On mesure **de combien** le modèle se trompe, avec un coût croissant selon l'écart :

| Prédiction | Gold | Écart | Gravité |
|---|---|---|---|
| 5 | 1 | −4 | 🚨 énorme (arrêt cardiaque raté) |
| 3 | 1 | −2 | sévère |
| 4 | 3 | −1 | modéré |
| 1 | 5 | +4 | sur-triage (coûteux mais sûr) |

Un modèle qui dit « 3 » pour un arrêt cardiaque (1) est bien pire qu'un modèle qui dit
« 5 » pour un « 4 » — le binaire ne le voit pas, le coût par distance oui.

### 4.2 Pourquoi on garde le binaire **en plus**

1. **Comparabilité inter-jeux** : 3, 4 et 5 niveaux ne se comparent pas, « ED vs pas » si.
2. **Interprétabilité** : « 1,8 % de sous-triage » est le chiffre qu'un clinicien comprend.
3. **La décision clinique au sommet est binaire** : envoyer aux urgences ou pas.

---

## 5. Mapping gold → plage FRENCH acceptable

Chaque label gold devient une **plage FRENCH acceptable** — cohérent avec notre propre
`priority_range` et qui gère naturellement les **split labels** (désaccord entre médecins) :

| Ramaswamy | Levine | IyàwóBench | Plage FRENCH acceptable |
|---|---|---|---|
| D | Emergent | REFER_NOW | [1, 3] |
| C/D | — | — | [3, 4] |
| C | 1 day | REFER_TODAY | [4, 4] |
| B/C | — | — | [4, 5] |
| B | 1 week | TREAT_HERE | [5, 5] |
| A/B | — | — | [5, 5] |
| A | Self-care | — | [5, 5] |

**Règles de correction** :
- `priority ∈ plage` → correct ;
- `priority > borne haute` → **sous-triage** de `priority − borne_haute` niveaux ;
- `priority < borne basse` → **sur-triage** de `borne_basse − priority` niveaux.

> Le « D → [1, 3] » reflète honnêtement le fait que le gold ne distingue pas 1/2/3 :
> n'importe quel niveau urgent est accepté. La distinction fine 1/2/3 n'est mesurable
> que sur notre set synthétique (étage 1).

---

## 6. Adaptation des jeux à notre format

Ces jeux sont des **vignettes mono-tour** (le patient donne tout en un message), ce qui
correspond exactement à notre format SFT « vignette » : `[system, user, assistant=fiche]`.

| Jeu | Champ utilisé comme message patient | Ajustement |
|---|---|---|
| Ramaswamy | `input_prompt` avec `prompt_type=0` (symptômes seuls) | écarter `prompt_type=1` (contient constantes non-observables) |
| Levine | `Current Problem` + `Additional Details` | concaténer |
| IyàwóBench | `symptoms` + âge/sexe | **écarter les constantes** (FC, FR, SpO₂) → non patient-reportable |

Sortie cible (par cas) :
```json
{"messages": [
   {"role": "system", "content": "<SYSTEM_PROMPT triage>"},
   {"role": "user",   "content": "<message patient>"}
 ],
 "metadata": {"source": "ramaswamy", "case_id": "...", "domain": "Cardiac",
              "gold_label_original": "C/D", "gold_range": [3, 4], "binary_urgent": false}}
```

Le harness d'évaluation lance le modèle sur `messages`, parse `[FICHE]` → `priority`,
puis applique les règles de correction du §5.

---

## 7. Décision proposée

| Jeu | Usage retenu |
|---|---|
| Ramaswamy (78, `prompt_type=0`) | ✅ évaluation principale (39 cas symptômes seuls) |
| Levine (48) | ✅ complément / smoke test |
| IyàwóBench (200) | ⚠️ optionnel OOD — constantes non-observables → probablement écarté |
| psy (112) | ❌ abandonné (indisponible) |

**Critère principal de sécurité** : sous-triage pondéré par la distance (étage 1 + 2),
synthétisé par le taux binaire de sous-triage (étage 3).
