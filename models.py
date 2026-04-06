"""
models.py — Typed Pydantic models for the Ethical Red-Teamer OpenEnv environment.

Defines:
  - RedTeamAction   : The action an agent takes (submitting its analysis)
  - RedTeamObservation: What the environment shows the agent
  - RedTeamState    : Episode metadata returned by state()
  - StepResult      : Combines observation + reward + done flag
"""

from __future__ import annotations
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Action — what the agent submits each step
# ---------------------------------------------------------------------------

class RedTeamAction(BaseModel):
    """
    The agent's response / analysis for the current task.

    For PII Detection:
        findings: list of detected PII strings (e.g., phone numbers)

    For Jailbreak Detection:
        findings: list of indices (0-based) of the flagged prompts
        reasoning: brief explanation of why each prompt is dangerous

    For System Prompt Hardening:
        hardened_prompt: the rewritten, hardened system prompt
        changes_summary: bullet-point list of changes made
    """
    task_id: str = Field(
        ...,
        description="Which task the agent is responding to",
        examples=["pii_detection", "jailbreak_detection", "system_prompt_hardening"],
    )
    findings: Optional[List[str]] = Field(
        default=None,
        description="Detected PII strings or jailbreak prompt indices (as strings)",
    )
    reasoning: Optional[str] = Field(
        default=None,
        description="Agent's reasoning / justification for its findings",
    )
    hardened_prompt: Optional[str] = Field(
        default=None,
        description="Rewritten hardened system prompt (Hard task only)",
    )
    changes_summary: Optional[str] = Field(
        default=None,
        description="Summary of changes made to harden the system prompt (Hard task only)",
    )


# ---------------------------------------------------------------------------
# Observation — what the environment sends back to the agent
# ---------------------------------------------------------------------------

class RedTeamObservation(BaseModel):
    """
    The observation returned after reset() or step().
    """
    task_id: str = Field(..., description="Current active task ID")
    task_name: str = Field(..., description="Human-readable task name")
    difficulty: str = Field(..., description="easy | medium | hard")
    instructions: str = Field(..., description="Full task instructions for the agent")

    # Payload varies by task
    dataset_sample: Optional[str] = Field(
        default=None,
        description="The dataset / text to analyse (PII or jailbreak task)",
    )
    system_prompt_to_audit: Optional[str] = Field(
        default=None,
        description="The vulnerable system prompt to harden (Hard task only)",
    )
    red_team_report: Optional[str] = Field(
        default=None,
        description="Red-team report describing loopholes (Hard task only)",
    )

    # Feedback after a step
    feedback: Optional[str] = Field(
        default=None,
        description="Grader feedback after a step",
    )
    score: Optional[float] = Field(
        default=None,
        description="Partial score awarded for this step (0.0 – 1.0)",
    )
    done: bool = Field(default=False, description="True when the episode ends")


# ---------------------------------------------------------------------------
# State — episode metadata returned by state()
# ---------------------------------------------------------------------------

class RedTeamState(BaseModel):
    """
    Episode-level metadata (returned by state()).
    """
    episode_id: str = Field(..., description="Unique identifier for this episode")
    task_id: str = Field(..., description="Active task ID")
    step_count: int = Field(default=0, description="Number of steps taken so far")
    cumulative_reward: float = Field(default=0.0, description="Total reward accumulated")
    done: bool = Field(default=False)


# ---------------------------------------------------------------------------
# StepResult — returned by step()
# ---------------------------------------------------------------------------

class StepResult(BaseModel):
    """
    Full result returned by the environment after step().
    """
    observation: RedTeamObservation
    reward: float = Field(default=0.0, ge=0.0, le=1.0)
    done: bool = False
    info: Dict[str, Any] = Field(default_factory=dict)
