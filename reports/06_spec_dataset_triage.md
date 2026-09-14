# Spec — Dataset de triage médical (FRENCH)

**Objet** : spécification détaillée du dataset synthétique de triage à construire
(vignettes + dialogues + paires de préférence), utilisé pour le SFT et le DPO de
l'agent de triage.
**Protocole** : FRENCH (SFMU) — voir `reports/04_protocole_french.md`.
**Statut** : référence pour la Phase 0 (encodage) et suivantes.

---

## 1. Les 4 objets de données

### 1.1 `fact_sheet` — vérité terrain (cachée)

Représente ce que saurait une **infirmière** après examen. Détermine le **niveau
FRENCH vrai** (calculé, jamais deviné).

```json
{
  "case_id": "cardio_001",
  "category": "cardio",
  "motif": "douleur_thoracique",
  "age": 58, "sex": "H",
  "signs_objectifs": ["ECG typique SCA"],
  "vital_signs": {"PAS": 95, "FC": 110, "SpO2": 94, "FR": 22, "T": 37.8},
  "medical_history": ["HTA", "tabagisme"],
  "treatments": ["amiodarone"],
  "true_level": 2,
  "justification": "ECG typique SCA → Tri 2"
}
```

### 1.2 `patient_observable` — ce que le patient peut dire

Sous-ensemble du fact sheet que le patient **peut révéler**, et uniquement quand
on le lui demande.

```json
{
  "can_report": {
    "symptoms": ["douleur serrante rétrosternale", "irradiation bras gauche", "sueurs"],
    "pain_scale": 8,
    "duration": "1h",
    "self_measured": {"temperature": null},
    "history": ["HTA"],
    "treatments": ["amiodarone"]
  },
  "cannot_report": ["ECG", "PAS", "SpO2"]
}
```

### 1.3 `dialogue` — l'exemple d'entraînement SFT

Conversation où l'**assistant joue l'agent**. Chaque tour assistant = `<think>`
+ sortie visible (question ou fiche).

```json
{"messages": [
  {"role": "system", "content": "Tu es un agent de triage médical. Vouvoie le patient, sois rassurant, pose une question à la fois."},
  {"role": "user", "content": "Bonjour, j'ai mal à la poitrine depuis une heure."},
  {"role": "assistant", "content": "<think>Motif: douleur thoracique. Red flags non écartés: SCA (caractère ?), dissection (irradiation ?). Plage [2,5]. Prochaine info: caractère + irradiation.</think>\nLa douleur est-elle serrante ? Part-elle vers le bras ou la mâchoire ?"},
  {"role": "user", "content": "Oui ça serre fort, et ça part dans le bras gauche. Je transpire."},
  {"role": "assistant", "content": "<think>SCA typique confirmé (serrante + irradiation + sueurs). Plage [2,2]. Niveau retenu: 2. Info manquante: ECG (côté infirmière).</think>\n[FICHE]\n{\"priority\": 2, \"priority_range\": [2, 2], \"red_flags\": [\"suspicion SCA\"], \"missing_info\": [\"ECG\"], \"confidence\": 0.9, \"summary\": \"Douleur thoracique serrante irradiant au bras gauche avec sueurs → suspicion de syndrome coronarien aigu.\", \"recommendation\": \"Prise en charge immédiate, médecin < 20 min, ECG en urgence.\"}\n[/FICHE]\nJe vous oriente tout de suite vers une prise en charge rapide, restez calme, une infirmière va vous voir immédiatement."}
]}
```

### 1.4 `fiche` — la sortie structurée (tool_call simulé)

Extrait entre `[FICHE]` et `[/FICHE]`, validé contre un schéma Pydantic, stocké
pour l'audit (intégration SI simulée).

| Champ | Type | Description |
|---|---|---|
| `priority` | int (1–5) | niveau retenu = **borne prudente** |
| `priority_range` | [int, int] | plage possible `[borne_urgente, borne_bénigne]` |
| `red_flags` | list[str] | red flags confirmés ou non écartés |
| `missing_info` | list[str] | infos manquantes (ex. ECG, PAS) |
| `confidence` | float (0–1) | confiance dans le niveau (∝ étroitesse de plage) |
| `summary` | str | résumé du cas pour le SI |
| `recommendation` | str | conduite à tenir (délai, orientation) |

---

## 2. Conventions de dialogue

- **Vouvoiement**, phrases courtes, **une question à la fois**.
- **Réassurance** sans condescendance ; pas de jargon (dire « cœur », pas « myocarde »).
- **`<think>`** : masqué au patient à l'inférence, **conservé pour l'audit**.
- **Fin de dialogue** : `<think>` + `[FICHE]` + explication naturelle au patient.
- **Patient vague** (« je ne sais pas », « je ne me souviens plus ») : réponse
  **valide** → l'agent applique la borne prudente et le note.

---

## 3. Gestion de l'incertitude (règle de décision)

1. À partir des faits connus, calculer la **plage** `[borne_urgente, borne_bénigne]`.
2. **Retenir la borne urgente** (précaution : le sous-triage est dangereux).
3. **Red flag non écarté = traité comme présent**.
4. `confidence` : échelle simple (à calibrer en Phase 1) —
   - plage largeur 1 → 0.9–1.0
   - largeur 2 → 0.6–0.8
   - largeur ≥ 3 → 0.3–0.5
   - ajustée selon la criticité de l'info manquante.

---

## 4. Règles de génération

### 4.1 Fact sheets (déterministe, niveau correct par construction)
1. Échantillonner `(catégorie, motif, règle)` dans `french_rules.json`.
2. La règle impose le `true_level` + les conditions (signes/constantes).
3. Compléter (âge, antécédents, traitements) en cohérence.
4. Déduire `patient_observable` (séparer `can_report` / `cannot_report`).

### 4.2 Vignettes (LLM = « habilleur », pas « décideur »)
- Le LLM (Gemma 4 26B) transforme le fact sheet en **langage patient naturel**,
  **sans changer les faits** ni le niveau.
- Prompt incluant : le fact sheet, les contraintes de ton (§2), et l'interdiction
  de mentionner les champs non observables.

### 4.3 Dialogues (fact sheet caché)
- Le LLM génère le dialogue : l'agent pose des questions ciblées (red flags), le
  patient ne révèle que `can_report`, et **seulement à la question posée**.
- Le niveau final de l'agent **doit** correspondre à `true_level` (borne prudente).
- Inclure une proportion de **patients vagues**.

### 4.4 Paires de préférence (DPO)
- Même cas → `chosen` = triage prudent + justification ; `rejected` = sous-triage
  ou hypothèse bénigne sans justification.

---

## 5. Règles de validation (4 couches)

| Couche | Méthode | Garantit |
|---|---|---|
| 1. Construction | niveau dérivé des règles FRENCH | annotation correcte à 100 % |
| 2. Ré-évaluation | LLM juge re-évalue la vignette → taux d'accord | cohérence interne |
| 3. Benchmark gold | exactitude sur Levine/Ramaswamy/IyàwóBench/psy (mapping 4→5) | justesse de triage réelle |
| 4. Revue humaine | échantillon 50–100 relu (réalisme, niveau, ton, RGPD) | qualité terrain |

---

## 6. Traçabilité et versioning

- Chaque exemple porte : `case_id`, `category`, `motif`, `true_level`, `source`
  (`synthetic_french`), `generator` (modèle + version), `validated` (bool).
- La **méthodologie** (protocole, règles, échantillonnage, validation) est
  documentée — exigé par le brief.
- Sorties dans `data/processed/triage/` (vignettes, dialogues, dpo).

---

## 7. Dépendance au modèle de génération

- **Modèle local** : Gemma 4 26B (MoE, rapide) — à confirmer via Ollama.
- Sert uniquement à **rédiger** (vignettes, dialogues, re-évaluation juge).
- Le **niveau est toujours calculé** par les règles FRENCH, jamais par le LLM.
