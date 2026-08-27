"""
server.py — FastAPI application exposing:
  1. The original OpenEnv-compliant environment API (/reset, /step, /state, /health)
  2. New agent-facing endpoints backed by the Gemini + Google ADK agent
     (/api/agent/*), persisted via storage.firestore.SessionStore.

Run locally:
    uvicorn server:app --reload --port 8080

Cloud Run sets $PORT automatically; see Dockerfile / README for deployment.
"""

import os
import uuid
from typing import Dict, Optional

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from agent.agent import AgentRunResult, run_ticket
from agent.evaluator import evaluate
from env.environment import CustomerSupportEnv
from env.models import Action
from storage.firestore import SessionStore

load_dotenv()  # no-op in prod if .env is absent; picks up local .env for dev

app = FastAPI(title="Customer Support Resolution Agent", version="2.0.0")

# Global session stores
envs: Dict[str, CustomerSupportEnv] = {}
agent_runs: Dict[str, AgentRunResult] = {}
store = SessionStore()

FRONTEND_DIST = os.path.join(os.path.dirname(__file__), "frontend", "dist")


# ---------------------------------------------------------------------------
# OpenEnv-compliant models — all fields optional so validator's bare POST works
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    task_level: Optional[str] = "easy"


class StepRequest(BaseModel):
    session_id: Optional[str] = "default"
    action: Action


class ResolveRequest(BaseModel):
    task_level: str = "easy"
    session_id: Optional[str] = None


# ---------------------------------------------------------------------------
# OpenEnv endpoints (preserved from the original environment API)
# ---------------------------------------------------------------------------
@app.post("/reset")
def reset_env(req: ResetRequest = ResetRequest()):
    """Reset the environment. session_id is auto-generated if not supplied."""
    session_id = req.session_id or str(uuid.uuid4())
    task_level = req.task_level or "easy"

    env = CustomerSupportEnv(task_level=task_level)
    obs = env.reset()
    envs[session_id] = env

    return {
        "session_id": session_id,
        "observation": obs.model_dump(),
    }


@app.post("/step")
def step_env(req: StepRequest):
    session_id = req.session_id or "default"
    if session_id not in envs:
        # Auto-init a default session so bare step calls don't 404
        env = CustomerSupportEnv(task_level="easy")
        env.reset()
        envs[session_id] = env

    env = envs[session_id]
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info,
    }


@app.get("/state/{session_id}")
def get_state(session_id: str):
    if session_id not in envs:
        return JSONResponse(status_code=404, content={"detail": "Session not found"})
    return {"state": envs[session_id].get_state().model_dump()}


@app.get("/health")
def health():
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Agent endpoints — the closed-loop Gemini + ADK + OpenEnv pipeline
# ---------------------------------------------------------------------------
@app.post("/api/agent/resolve")
def resolve_ticket(req: ResolveRequest):
    """Runs the full autonomous agent loop against a fresh ticket and
    persists the resulting session, trace, and evaluation metrics."""
    session_id = req.session_id or str(uuid.uuid4())
    result = run_ticket(session_id=session_id, task_level=req.task_level)
    agent_runs[session_id] = result

    metrics = evaluate(result)
    state = result.env.get_state()

    payload = {
        "session_id": session_id,
        "status": result.status,
        "final_response": result.final_response,
        "reward": round(result.total_reward, 4),
        "metrics": metrics,
        "actions": [t.to_dict() for t in result.trace],
        # "adk" = real Google ADK Agent + Runner (production path, used
        # whenever Gemini credentials are configured); "deterministic_fallback"
        # = offline/local-dev oracle path (no Gemini credentials present).
        "execution_path": result.execution_path,
    }

    store.save_session(session_id, {
        "ticket": state.ticket.model_dump(),
        "conversation_history": state.conversation_history,
        "actions": payload["actions"],
        "refund_issued": state.refund_issued,
        "metrics": metrics,
        "customer_context": state.customer_context.model_dump(),
    })

    return payload


@app.get("/api/agent/session/{session_id}")
def get_agent_session(session_id: str):
    session = store.get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"detail": "Session not found"})
    return session


@app.get("/api/agent/trace/{session_id}")
def get_agent_trace(session_id: str):
    session = store.get_session(session_id)
    if session is None:
        return JSONResponse(status_code=404, content={"detail": "Session not found"})
    return {"session_id": session_id, "actions": session["actions"]}


@app.get("/api/agent/evaluation/{session_id}")
def get_agent_evaluation(session_id: str):
    metrics = store.get_evaluation(session_id)
    if metrics is None:
        return JSONResponse(status_code=404, content={"detail": "Session not found"})
    return {"session_id": session_id, "metrics": metrics}


@app.get("/api/agent/sessions")
def list_agent_sessions():
    return {"sessions": store.list_sessions()}


# ---------------------------------------------------------------------------
# Frontend (static files) — served if a build exists; API still works without it
# ---------------------------------------------------------------------------
if os.path.isdir(FRONTEND_DIST):
    app.mount("/", StaticFiles(directory=FRONTEND_DIST, html=True), name="frontend")
