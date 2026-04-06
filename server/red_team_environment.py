"""
server/red_team_environment.py

Core OpenEnv-compliant Environment class for the Ethical Red-Teamer.
Implements:
  - reset()  → returns initial RedTeamObservation
  - step()   → grades the action, returns StepResult
  - state()  → returns current RedTeamState
"""

from __future__ import annotations

import uuid
from typing import Optional

from models import (
    RedTeamAction,
    RedTeamObservation,
    RedTeamState,
    StepResult,
)
from tasks import (
    # Task 1
    TASK_PII_INSTRUCTIONS,
    generate_pii_dataset,
    grade_pii,
    # Task 2
    TASK_JAILBREAK_INSTRUCTIONS,
    get_jailbreak_dataset,
    grade_jailbreak,
    # Task 3
    TASK_HARDENING_INSTRUCTIONS,
    VULNERABLE_SYSTEM_PROMPT,
    RED_TEAM_REPORT,
    grade_hardening,
)

TASK_ORDER = ["pii_detection", "jailbreak_detection", "system_prompt_hardening"]

TASK_META = {
    "pii_detection": {
        "name": "Easy: PII Detection",
        "difficulty": "easy",
    },
    "jailbreak_detection": {
        "name": "Medium: Jailbreak Detection",
        "difficulty": "medium",
    },
    "system_prompt_hardening": {
        "name": "Hard: System Prompt Hardening",
        "difficulty": "hard",
    },
}


class RedTeamEnvironment:
    """
    OpenEnv-compatible environment for Ethical Red-Teaming.

    Episode flow:
      reset() → step(action) → [optional more steps] → done=True
    """

    def __init__(self) -> None:
        self._state: Optional[RedTeamState] = None
        self._pii_dataset: Optional[str] = None
        self._pii_ground_truth: Optional[list] = None
        self._jailbreak_dataset: Optional[str] = None
        self._jailbreak_ground_truth: Optional[list] = None
        self._task_index: int = 0

    # ------------------------------------------------------------------
    # Public OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> RedTeamObservation:
        """Start a fresh episode from the first task."""
        self._task_index = 0
        episode_id = str(uuid.uuid4())

        # pre-generate datasets
        self._pii_dataset, self._pii_ground_truth = generate_pii_dataset(
            num_lines=10_000, num_pii=15, seed=42
        )
        self._jailbreak_dataset, self._jailbreak_ground_truth = get_jailbreak_dataset()

        self._state = RedTeamState(
            episode_id=episode_id,
            task_id=TASK_ORDER[0],
            step_count=0,
            cumulative_reward=0.0,
            done=False,
        )

        return self._make_observation(task_id=TASK_ORDER[0])

    def step(self, action: RedTeamAction) -> StepResult:
        """Grade the agent's action and advance to the next task."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        task_id = self._state.task_id
        self._state.step_count += 1

        # --- Grade ---
        reward, feedback = self._grade(task_id, action)
        self._state.cumulative_reward = round(
            self._state.cumulative_reward + reward, 4
        )

        # Advance task
        self._task_index += 1
        done = self._task_index >= len(TASK_ORDER)
        self._state.done = done

        if not done:
            next_task_id = TASK_ORDER[self._task_index]
            self._state.task_id = next_task_id
            obs = self._make_observation(task_id=next_task_id, feedback=feedback, score=reward)
        else:
            # Final observation
            obs = RedTeamObservation(
                task_id=task_id,
                task_name="All Tasks Complete",
                difficulty="done",
                instructions=(
                    f"All tasks completed! "
                    f"Total reward: {self._state.cumulative_reward:.4f}"
                ),
                feedback=feedback,
                score=reward,
                done=True,
            )

        return StepResult(
            observation=obs,
            reward=reward,
            done=done,
            info={
                "task_id": task_id,
                "step": self._state.step_count,
                "cumulative_reward": self._state.cumulative_reward,
                "feedback": feedback,
            },
        )

    def state(self) -> RedTeamState:
        """Return the current episode state metadata."""
        if self._state is None:
            raise RuntimeError("Call reset() before state().")
        return self._state

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _make_observation(
        self,
        task_id: str,
        feedback: Optional[str] = None,
        score: Optional[float] = None,
    ) -> RedTeamObservation:
        meta = TASK_META[task_id]

        if task_id == "pii_detection":
            return RedTeamObservation(
                task_id=task_id,
                task_name=meta["name"],
                difficulty=meta["difficulty"],
                instructions=TASK_PII_INSTRUCTIONS,
                dataset_sample=self._pii_dataset,
                feedback=feedback,
                score=score,
                done=False,
            )
        elif task_id == "jailbreak_detection":
            return RedTeamObservation(
                task_id=task_id,
                task_name=meta["name"],
                difficulty=meta["difficulty"],
                instructions=TASK_JAILBREAK_INSTRUCTIONS,
                dataset_sample=self._jailbreak_dataset,
                feedback=feedback,
                score=score,
                done=False,
            )
        elif task_id == "system_prompt_hardening":
            return RedTeamObservation(
                task_id=task_id,
                task_name=meta["name"],
                difficulty=meta["difficulty"],
                instructions=TASK_HARDENING_INSTRUCTIONS,
                system_prompt_to_audit=VULNERABLE_SYSTEM_PROMPT,
                red_team_report=RED_TEAM_REPORT,
                feedback=feedback,
                score=score,
                done=False,
            )
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

    def _grade(self, task_id: str, action: RedTeamAction) -> tuple[float, str]:
        """Dispatch to the correct grader."""
        if task_id == "pii_detection":
            return grade_pii(
                findings=action.findings or [],
                ground_truth=self._pii_ground_truth or [],
            )
        elif task_id == "jailbreak_detection":
            return grade_jailbreak(
                findings=action.findings or [],
                ground_truth=self._jailbreak_ground_truth or [],
            )
        elif task_id == "system_prompt_hardening":
            return grade_hardening(
                hardened_prompt=action.hardened_prompt,
                changes_summary=action.changes_summary,
            )
        else:
            return 0.0, f"Unknown task: {task_id}"
