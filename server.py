import uuid
from fastapi import FastAPI, Body
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Dict, Optional
from env.environment import CustomerSupportEnv
from env.models import Action

app = FastAPI(title="OpenEnv - Customer Support Environment")

# Global session store
envs: Dict[str, CustomerSupportEnv] = {}

# ---------------------------------------------------------------------------
# OpenEnv-compliant models — all fields optional so validator's bare POST works
# ---------------------------------------------------------------------------
class ResetRequest(BaseModel):
    session_id: Optional[str] = None
    task_level: Optional[str] = "easy"

class StepRequest(BaseModel):
    session_id: Optional[str] = "default"
    action: Action

# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------
@app.post("/reset")
def reset_env(req: Optional[ResetRequest] = Body(None)):
    """Reset the environment. session_id is auto-generated if not supplied."""
    if req is None:
        req = ResetRequest()
    
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
    return {"state": envs[session_id].state().model_dump()}

@app.get("/health")
def health():
    return {"status": "ok"}
