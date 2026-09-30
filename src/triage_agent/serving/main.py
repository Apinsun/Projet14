"""API FastAPI de l'agent de triage médical (CHSA).

Routes :
- ``GET  /health`` — état du service ;
- ``POST /chat``  — envoie l'historique complet, reçoit la réponse structurée ;
- ``GET  /``      — interface web de démonstration (statique).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from triage_agent.serving.agent import TriageAgent
from triage_agent.serving.config import Settings, get_settings
from triage_agent.serving.schemas import ChatRequest, ChatResponse
from triage_agent.serving.vllm_client import VLLMClient, VLLMError

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Agent de triage médical — CHSA",
    description=(
        "POC : agent conversationnel de triage (protocole FRENCH, 5 niveaux), "
        "fine-tuné Qwen3-1.7B (SFT LoRA + DPO), servi par vLLM."
    ),
    version="0.1.0",
)


def build_agent(settings: Settings) -> TriageAgent:
    """Construit l'agent branché sur le vrai vLLM (remplaçable en test)."""
    return TriageAgent(VLLMClient(settings.vllm_url, settings.vllm_model), settings)


def get_agent(settings: Settings = Depends(get_settings)) -> TriageAgent:
    """Dépendance FastAPI : l'agent (surchargeable via ``dependency_overrides``)."""
    return build_agent(settings)


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "service": "triage-agent"}


@app.post("/chat", response_model=ChatResponse)
def chat(req: ChatRequest, agent: TriageAgent = Depends(get_agent)) -> ChatResponse:
    messages = [{"role": m.role, "content": m.content} for m in req.messages]
    try:
        result = agent.handle(messages, req.conversation_id)
    except VLLMError as exc:
        raise HTTPException(status_code=502, detail=f"vLLM indisponible : {exc}") from exc
    return ChatResponse(**result)


@app.get("/", include_in_schema=False)
def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")
