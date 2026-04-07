from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, Any
from env.environment import CustomerSupportEnv
from env.models import Action, Observation

app = FastAPI(title="OpenEnv - Customer Support Environment")

envs: Dict[str, CustomerSupportEnv] = {}

class ResetRequest(BaseModel):
    session_id: str
    task_level: str = "easy"

class StepRequest(BaseModel):
    session_id: str
    action: Action

@app.post("/reset")
def reset_env(req: ResetRequest):
    env = CustomerSupportEnv(task_level=req.task_level)
    obs = env.reset()
    envs[req.session_id] = env
    return {"observation": obs.model_dump()}

@app.post("/step")
def step_env(req: StepRequest):
    if req.session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found")
        
    env = envs[req.session_id]
    obs, reward, done, info = env.step(req.action)
    return {
        "observation": obs.model_dump(),
        "reward": reward,
        "done": done,
        "info": info
    }

@app.get("/state/{session_id}")
def get_state(session_id: str):
    if session_id not in envs:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"state": envs[session_id].get_state().model_dump()}
