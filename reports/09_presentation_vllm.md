# vLLM — présentation, mise en place et différences avec Ollama

**Objet** : présenter le moteur d'inférence vLLM (exigé par le brief), sa mise en place
dans ce projet, et le comparer à Ollama.
**Date** : 2026-09-14

---

## 1. Qu'est-ce que vLLM ?

vLLM est un **moteur d'inférence et de serving de LLM**, développé initialement à UC
Berkeley, aujourd'hui le standard de l'industrie pour **servir des modèles en production**
(utilisé par Anyscale, Databricks, NVIDIA…).

Deux innovations fondatrices :

### 1.1 PagedAttention
Gère le cache KV (la mémoire des tokens déjà générés) **comme un OS gère la mémoire
virtuelle** : par pages, avec allocation à la demande. Résultat : on ne gaspille plus la
VRAM en réservant de gros blocs contigus → plus de requêtes servies en parallèle, plus
longues séquences, moins de fragmentation.

### 1.2 Continuous batching (batching continu)
Un serveur classique attend qu'un batch soit plein avant de traiter. vLLM **ajoute/retire
les requêtes au fil de l'eau** : dès qu'une séquence finit, sa place est reprise par une
autre. → latence plus basse, débit (throughput) bien plus élevé sous charge.

**Autres capacités** : API **compatible OpenAI** (`/v1/chat/completions`), quantification
(AWQ, GPTQ, FP8), **serving de plusieurs adaptateurs LoRA** sur une même base,
parallélisme multi-GPU, Docker/Kubernetes natif.

---

## 2. vLLM vs Ollama

| Critère | Ollama | vLLM |
|---|---|---|
| **Cible** | développement / usage perso | production / entreprise |
| **Backend** | llama.cpp (GGUF) | noyaux CUDA natifs + PagedAttention |
| **Batching** | limité | **continu** |
| **Débit sous concurrence** | ~41 TPS (pic) | **~793 TPS** (pic) — ≈19× (benchmark Red Hat) |
| **Latence P99 au pic** | ~673 ms | **~80 ms** |
| **API** | OpenAI-*like* (partielle) | **OpenAI-complète** (`/v1/*`) |
| **LoRA (multi-adapter)** | non | **oui** (`--enable-lora`) |
| **Multi-GPU** | limité | tensor parallelism |
| **Quantification** | GGUF (Q4_K_M…) | AWQ / GPTQ / FP8 |
| **Installation** | 1 binaire, trivial | venv/Docker + GPU CUDA |
| **Déploiement** | mono-process | Docker, k8s, autoscaling |

> **En résumé** : Ollama = « télécharger et essayer un modèle en 30 s ». vLLM = « servir
> un modèle à des dizaines de requêtes concurrentes avec un débit et une latence de
> production ». À une requête à la fois, ils se valent ; **dès qu'il y a de la
> concurrence, vLLM écrase Ollama** (et c'est exactement le scénario d'un service de
> triage avec plusieurs patients simultanés).

---

## 3. Installation : projet ou système ?

vLLM est un **paquet Python** — il s'installe **dans un environnement Python**, pas
« dans le système » au sens OS. Trois options, du plus rapide au plus propre :

| Option | Portée | Quand |
|---|---|---|
| **venv du projet** (`poetry`/`uv`) | ce projet uniquement | dev + tests locaux |
| **Docker** (`vllm/vllm-openai`) | conteneur isolé | **livrable final** (exigé par le brief) |
| pipx / pip système | machine entière | ⚠️ déconseillé (déps CUDA, conflits) |

**Points concrets pour notre machine** :

- **GPU** : RTX 3090 24 Go — largement suffisant pour Qwen3-1.7B (≈3,4 Go en FP16).
- **CUDA** : driver 595.84 (CUDA 13.2) — les wheels pip de vLLM embarquent leurs propres
  noyaux CUDA, donc pas besoin de faire matcher la version CUDA système.
- **Python** : le système est en **3.14** (incompatible avec l'écosystème ML). vLLM exige
  Python 3.9–3.12 → on passera **forcément** par le venv du projet (3.12 via `uv`) ou par
  Docker.
- **Docker** : déjà installé (29.8) → c'est la voie recommandée pour le livrable.

---

## 4. Où vLLM se place dans notre projet

Architecture cible (semaine 4) :

```
[Client / SI]
     │  requêtes triage
     ▼
[FastAPI]  ← logique de triage : questionnaire, <think>, validation Pydantic,
     │        intégration SI simulée, traçabilité
     │  appel OpenAI-compatible (/v1/chat/completions)
     ▼
[vLLM]  ← Qwen3-1.7B fine-tuné (serveur d'inférence, port 8000)
```

Le tout conteneurisé (docker-compose) + pipeline CI/CD.

> **Pourquoi FastAPI devant vLLM ?** vLLM ne fait que « texte → texte ». Toute la logique
> de notre agent (gestion du `<think>` masqué, extraction/validation de la fiche
> `[FICHE]` via Pydantic, conservation pour audit) vit dans FastAPI. vLLM est le « moteur »,
> FastAPI est le « cerveau applicatif ».

---

## 5. Le point LoRA (important pour nous)

vLLM sait servir un adaptateur LoRA **sur** une base :

```bash
vllm serve Qwen/Qwen3-1.7B \
  --enable-lora \
  --lora-modules triage=./adapters/triage_lora
```

Mais ⚠️ : le support LoRA de certaines archis Qwen récentes (3.5/3.6 GatedDeltaNet) a des
bugs version-sensibles. Pour **Qwen3-1.7B** (architecture transformer classique), les
cibles Unsloth (`q_proj, k_proj, v_proj, o_proj, gate/up/down_proj`) sont standards.

**Recommandation pour le POC** : **fusionner le LoRA dans les poids** avant de servir
(`model.save_pretrained_merged()` chez Unsloth). On sert alors un **modèle unique** :

```bash
vllm serve ./models/qwen3-1.7b-triage-merged
```

→ zéro dépendance à la mécanique LoRA de vLLM, robuste, et suffisant pour une tâche
unique. Le multi-LoRA (`--enable-lora`) ne deviendrait utile que si on servait plusieurs
variantes sur une même base — hors périmètre du POC.

---

## 6. Plan de mise en place (concret)

### Étape A — dev local (venv du projet)
```bash
# dans l'environnement Python 3.12 du projet
uv pip install vllm            # wheel CUDA précompilé
vllm serve Qwen/Qwen3-1.7B --port 8000
```

### Étape B — production (Docker, exigé par le brief)
```dockerfile
FROM vllm/vllm-openai:latest
# copier le modèle fusionné + exposer 8000
```
```yaml
# docker-compose.yml
services:
  vllm:
    image: vllm/vllm-openai:latest
    command: ["--model", "/models/qwen3-1.7b-triage-merged", "--port", "8000"]
    deploy: { resources: { reservations: { devices: [{ driver: nvidia, count: all }] } } }
  api:
    build: ./api            # FastAPI → vllm:8000
    ports: ["8080:8080"]
```

### Étape C — vérification
```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"qwen3-1.7b-triage-merged","messages":[{"role":"user","content":"Bonjour..."}]}'
```

---

## 7. Points de vigilance

1. **GPU NVIDIA requis** (CUDA). vLLM ne tourne pas sur CPU/Apple Silicon (Ollama, si).
2. **Python 3.9–3.12** — pas 3.14 (le système). → venv projet ou Docker obligatoires.
3. **VRAM** : Qwen3-1.7B FP16 ≈ 3,4 Go → OK sur 24 Go, avec marge pour le batching.
4. **Version vLLM** : épingler une version stable (les LoRA Qwen récents sont sensibles).
5. **Fusion LoRA** plutôt que `--enable-lora` pour le POC (cf. §5).
