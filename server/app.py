"""
server/app.py — FastAPI server exposing the OpenEnv-standard HTTP endpoints.

Endpoints:
  POST /reset        → RedTeamObservation
  POST /step         → StepResult
  GET  /state        → RedTeamState
  GET  /tasks        → list of task definitions
  GET  /health       → 200 OK (required by HF Spaces auto-ping)
  GET  /             → redirect to /docs
"""

from __future__ import annotations

import sys
import os

# Make parent directory importable (so models.py and tasks.py are found)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastapi import FastAPI, HTTPException
from fastapi.responses import RedirectResponse, JSONResponse

from models import RedTeamAction, RedTeamObservation, RedTeamState, StepResult
from server.red_team_environment import RedTeamEnvironment

app = FastAPI(
    title="Ethical Red-Teamer — OpenEnv Environment",
    description=(
        "An AI Safety & Ethics OpenEnv environment. "
        "An AI agent audits datasets and prompts for PII, jailbreaks, "
        "and system prompt vulnerabilities before model deployment."
    ),
    version="1.0.0",
)

# Single shared environment instance (stateful per deployment)
_env = RedTeamEnvironment()


# ---------------------------------------------------------------------------
# Required OpenEnv endpoints
# ---------------------------------------------------------------------------

@app.get("/", include_in_schema=False)
async def root():
    return RedirectResponse(url="/docs")


@app.get("/health")
async def health():
    """Health check — must return 200 (required by HF Spaces auto-ping)."""
    return {"status": "ok", "environment": "ethical-red-teamer"}


@app.post("/reset", response_model=RedTeamObservation)
async def reset():
    """
    Reset the environment and return the initial observation.
    Starts the agent on Task 1 (PII Detection).
    """
    obs = _env.reset()
    return obs


@app.post("/step", response_model=StepResult)
async def step(action: RedTeamAction):
    """
    Submit an action (agent's analysis).
    Returns the graded StepResult including the next observation and reward.
    """
    try:
        result = _env.step(action)
        return result
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/state", response_model=RedTeamState)
async def state():
    """Return the current episode state (episode_id, step_count, etc.)."""
    try:
        return _env.state()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.get("/tasks")
async def list_tasks():
    """Return metadata about all available tasks."""
    return JSONResponse(content={
        "multi_step": True,
        "max_steps_per_task": 5,
        "tasks": [
            {
                "id": "pii_detection",
                "name": "Easy: PII Detection",
                "difficulty": "easy",
                "description": (
                    "Find all phone numbers hidden in a 10,000-line synthetic dataset. "
                    "Reward = F1-score of detected vs ground-truth phone numbers."
                ),
                "reward_range": [0.0, 1.0],
                "supports_incremental": True,
            },
            {
                "id": "jailbreak_detection",
                "name": "Medium: Jailbreak Detection",
                "difficulty": "medium",
                "description": (
                    "Identify jailbreak prompts among 20 user messages that attempt to "
                    "elicit prohibited medical advice from a chatbot. "
                    "Reward = F1-score."
                ),
                "reward_range": [0.0, 1.0],
                "supports_incremental": True,
            },
            {
                "id": "system_prompt_hardening",
                "name": "Hard: System Prompt Hardening",
                "difficulty": "hard",
                "description": (
                    "Rewrite a vulnerable system prompt to close all loopholes "
                    "identified in a red-team report. "
                    "Reward = weighted rubric score covering each vulnerability."
                ),
                "reward_range": [0.0, 1.0],
                "supports_incremental": True,
            },
        ]
    })

def main():
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=False)

if __name__ == "__main__":
    main()
