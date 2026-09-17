# Extrait du dataset de triage — POC agent IA (CHSA)

> Exemples réels issus du dataset utilisé pour le fine-tuning de Qwen3-1.7B
> (SFT LoRA). Niveau de triage **calculé** selon le protocole FRENCH (5 niveaux),
> jamais deviné par le LLM.

## Composition du dataset d'entraînement

| Jeu | Langue | Exemples | Rôle |
|---|---|---|---|
| MedQuAD + FrenchMedMCQA + MediQA | FR + EN | 4 500 | connaissance médicale |
| Vignettes de triage synthétiques | FR | 300 | patient → fiche (mono-tour) |
| Dialogues de triage synthétiques | FR | 300 | questionnaire multi-tours |
| **Total** | | **5 100** | |

## 1. Vignettes — un exemple par niveau FRENCH

> Le patient décrit ses symptômes ; l'agent répond par une **fiche structurée**
> `[FICHE] ... [/FICHE]` + une explication en langage clair.

### Niveau 1

**Patient :** *(Note : Étant donné que le patient est inconscient et ne respire plus, il ne peut pas s'exprimer lui-même. Le message d'ouverture est donc formulé par la personne qui l'accompagne aux urgences.)*

« S'il vous plaît, aidez-la ! Elle ne respire plus et elle est inconsciente ! Elle a 46 ans... Je sais qu'elle prend du paracétamol, mais c'est tout. »

**Agent :**

```json
{
  "name": "finalize_triage",
  "arguments": {
    "priority": 1,
    "priority_range": [
      1,
      1
    ],
    "red_flags": [
      "arrêt cardiorespiratoire"
    ],
    "missing_info": [
      "arrêt cardiorespiratoire"
    ],
    "confidence": 0.9,
    "summary": "arrêt cardiorespiratoire",
    "recommendation": "Détresse vitale majeure — médecin immédiat"
  }
}
```
**Explication :** Votre situation est grave, une équipe va vous prendre en charge immédiatement, restez avec nous.

---

### Niveau 2

**Patient :** « Bonjour... C'est que... j'ai pris des médicaments pour en finir. Je sais que j'ai déjà fait un AVC et que j'ai une dyslipidémie... »

**Agent :**

```json
{
  "name": "finalize_triage",
  "arguments": {
    "priority": 2,
    "priority_range": [
      2,
      2
    ],
    "red_flags": [
      "comportement suicidaire / passage à l'acte"
    ],
    "missing_info": [],
    "confidence": 0.9,
    "summary": "comportement suicidaire / passage à l'acte",
    "recommendation": "Atteinte patente d'un organe / lésion traumatique sévère — médecin < 20 min"
  }
}
```
**Explication :** Votre état nécessite une prise en charge rapide, une infirmière va vous voir très vite.

---

### Niveau 3

**Patient :** « Bonjour... excusez-moi, je suis très essoufflée. Je... je dois m'asseoir pour pouvoir respirer. J'ai déjà fait un AVC auparavant et je prends de l'aspirine et du furosémide. »

**Agent :**

```json
{
  "name": "finalize_triage",
  "arguments": {
    "priority": 3,
    "priority_range": [
      3,
      3
    ],
    "red_flags": [],
    "missing_info": [
      "FR 30-40/min",
      "SpO2 86-90%",
      "FR",
      "SpO2"
    ],
    "confidence": 0.9,
    "summary": "FR 30-40/min, SpO2 86-90%, dyspnée à la parole, tirage, orthopnée",
    "recommendation": "Atteinte fonctionnelle ou lésionnelle stable — médecin < 90 min"
  }
}
```
**Explication :** Vous allez être vu rapidement, prévenez-nous si un symptôme s'aggrave.

---

### Niveau 4

**Patient :** « Bonjour... Je viens parce que j'ai avalé un produit hier. Enfin, je me sens bien, là, mais bon... J'ai déjà fait un AVC et j'ai des problèmes au cœur, donc je préfère venir vérifier. Je prends du paracétamol et du furosémide d'habitude. »

**Agent :**

```json
{
  "name": "finalize_triage",
  "arguments": {
    "priority": 4,
    "priority_range": [
      4,
      4
    ],
    "red_flags": [],
    "missing_info": [],
    "confidence": 0.9,
    "summary": "pas de signes associés, vu tard",
    "recommendation": "Atteinte fonctionnelle/lésionnelle stable, acte limité — médecin < 120 min"
  }
}
```
**Explication :** Votre situation est stable, vous serez vu dans les deux heures environ.

---

### Niveau 5

**Patient :** « Bonjour... Excusez-moi, je viens parce que j'ai mal à une dent. J'ai pris du paracétamol, mais ça ne passe pas vraiment. »

**Agent :**

```json
{
  "name": "finalize_triage",
  "arguments": {
    "priority": 5,
    "priority_range": [
      5,
      5
    ],
    "red_flags": [],
    "missing_info": [],
    "confidence": 0.9,
    "summary": "douleur dentaire simple",
    "recommendation": "Pas d'atteinte fonctionnelle ou lésionnelle évidente — médecin < 240 min"
  }
}
```
**Explication :** Votre situation ne semble pas urgente, vous serez vu dans la journée ; en cas d'aggravation, prévenez-nous.

---

## 2. Dialogue — questionnaire multi-tours

> L'agent pose **une question à la fois**, raisonne en interne dans des balises
> `<think>...</think>` (masquées au patient, conservées pour l'audit), puis conclut
> par la fiche.

**Consigne système :** Tu es un agent de triage médical pour les urgences. Tu vouvouies le patient, tu poses une question à la fois, tu es rassurant et tu n'utilises pas de jargon. Tu raisonnes en interne dans des balises <think>...</think> (faits connus, red flags, plage d'urgence, prochaine question). Quand tu as assez d'éléments, tu termines par une fiche entre [FICHE] et [/FICHE] suivie d'une explication claire au patient.

**Patient :** Bonjour, je ne me sens pas bien, j'aurais besoin d'être vu.

**Agent** *(raisonnement interne)* : `Accueil. Objectif : identifier le motif de recours.`

**Agent :** Bonjour, je vous écoute. Qu'est-ce qui vous amène ?

**Patient :** Euh, j'ai pris des médicaments pour en finir.

**Agent** *(raisonnement interne)* : `Motif : Idée / comportement suicidaire. Red flag (comportement suicidaire / passage à l'acte) : confirmé. Plage : [1, 2]. Question suivante : antécédents.`

**Agent :** Avez-vous des antécédents médicaux ?

**Patient :** Euh, j'ai de la dyslipidémie et un AVC ancien.

**Agent** *(raisonnement interne)* : `Niveau retenu : 2 (comportement suicidaire / passage à l'acte). Info manquante (mesurée) : aucune. Confiance élevée.`

```json
{
  "name": "finalize_triage",
  "arguments": {
    "priority": 2,
    "priority_range": [
      2,
      2
    ],
    "red_flags": [
      "comportement suicidaire / passage à l'acte"
    ],
    "missing_info": [],
    "confidence": 0.9,
    "summary": "comportement suicidaire / passage à l'acte",
    "recommendation": "Atteinte patente d'un organe / lésion traumatique sévère — médecin < 20 min"
  }
}
```
**Explication :** Votre état nécessite une prise en charge rapide, une infirmière va vous voir très vite.

## 3. Base de connaissance médicale (Q&A)

### Anglais (MedQuAD)

**Question :** What causes Nephrotic Syndrome in Adults ?

**Réponse :** Nephrotic syndrome can be caused by diseases that affect only the kidneys, such as focal segmental glomerulosclerosis (FSGS) or membranous nephropathy. Diseases that affect only the kidneys are called primary causes of nephrotic syndrome. The glomeruli are usually the targets of these diseases for reasons that are not fully understood. In FSGSthe most common primary cause of nephrotic syndromescar…

### Français (FrenchMedMCQA)

**Question :** Parmi les propositions suivantes concernant les Herpesviridae, lesquelles sont exactes ?
(A) Ils établissent tous leur latence dans le système nerveux central
(B) Les symptômes observés au cours des réactivations virales sont plus sévères que ceux de la primo-infection
(C) Ils présentent tous la même structure en microscopie électronique
(D) Ils sont fréquemment impliqués dans les complications infectieuses des états d’immuno-dépression
(E) Leur multiplication fait intervenir une ADN-polymérase virale

**Réponse :** Les réponses correctes sont : Ils présentent tous la même structure en microscopie électronique, Ils sont fréquemment impliqués dans les complications infectieuses des états d’immuno-dépression, Leur multiplication fait intervenir une ADN-polymérase virale.…
