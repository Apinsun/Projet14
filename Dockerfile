# Image de l'application FastAPI (serving uniquement).
# vLLM tourne dans un conteneur séparé (voir docker-compose.yml).
FROM python:3.12-slim

WORKDIR /app

# Dépendances légères de serving uniquement (pas spacy/presidio/datasets…)
COPY requirements-serving.txt .
RUN pip install --no-cache-dir -r requirements-serving.txt

# Code applicatif (package triage_agent, layout src/)
COPY src/ /app/src/
ENV PYTHONPATH=/app/src

EXPOSE 8080

CMD ["uvicorn", "triage_agent.serving.main:app", "--host", "0.0.0.0", "--port", "8080"]
