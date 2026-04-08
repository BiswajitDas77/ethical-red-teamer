"""
server/red_team_environment.py

Core OpenEnv-compliant Environment class for the Ethical Red-Teamer.
Implements:
  - reset()  → returns initial RedTeamObservation
  - step()   → grades the action, returns StepResult
  - state()  → returns current RedTeamState

Supports multi-step tasks with partial progress rewards:
  - Agents can submit incremental findings (finalize=False)
  - Partial scores are computed and returned
  - Best score per task is tracked
"""

from __future__ import annotations

import uuid
from typing import Dict, List, Optional

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

# Maximum steps allowed per task (prevents infinite loops)
MAX_STEPS_PER_TASK = 5


class RedTeamEnvironment:
    """
    OpenEnv-compatible environment for Ethical Red-Teaming.

    Supports multi-step tasks with partial progress:
      - Agents can submit findings incrementally (finalize=False)
      - Partial scores track progress toward task completion
      - Best score is retained across attempts

    Episode flow:
      reset() → step(action, finalize=False) → ... → step(action, finalize=True) → next task
    """

    def __init__(self) -> None:
        self._state: Optional[RedTeamState] = None
        self._pii_dataset: Optional[str] = None
        self._pii_ground_truth: Optional[List[str]] = None
        self._jailbreak_dataset: Optional[str] = None
        self._jailbreak_ground_truth: Optional[List[str]] = None

        # Track cumulative findings per task
        self._cumulative_findings: Dict[str, List[str]] = {}
        self._task_step_counts: Dict[str, int] = {}

        # Track best scores per task for reward shaping
        self._task_best_scores: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Public OpenEnv API
    # ------------------------------------------------------------------

    def reset(self) -> RedTeamObservation:
        """Start a fresh episode from the first task."""
        episode_id = str(uuid.uuid4())

        # Pre-generate datasets
        self._pii_dataset, self._pii_ground_truth = generate_pii_dataset(
            num_lines=10_000, num_pii=15, seed=42
        )
        self._jailbreak_dataset, self._jailbreak_ground_truth = get_jailbreak_dataset()

        # Reset cumulative tracking
        self._cumulative_findings = {}
        self._task_step_counts = {task_id: 0 for task_id in TASK_ORDER}
        self._task_best_scores = {task_id: 0.0 for task_id in TASK_ORDER}

        self._state = RedTeamState(
            episode_id=episode_id,
            task_id=TASK_ORDER[0],
            step_count=0,
            task_step_count=0,
            cumulative_reward=0.0,
            best_scores={task_id: 0.0 for task_id in TASK_ORDER},
            done=False,
        )

        return self._make_observation(task_id=TASK_ORDER[0])

    def step(self, action: RedTeamAction) -> StepResult:
        """
        Process an action and return graded StepResult.

        If action.finalize=True (default), the task is completed and we move to next task.
        If action.finalize=False, the agent can submit incremental findings and continue.
        """
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        task_id = self._state.task_id
        self._state.step_count += 1
        self._task_step_counts[task_id] += 1

        # Accumulate findings for this task
        if action.findings:
            if task_id not in self._cumulative_findings:
                self._cumulative_findings[task_id] = []
            # Deduplicate while preserving order
            existing = set(self._cumulative_findings[task_id])
            for f in action.findings:
                if f not in existing:
                    self._cumulative_findings[task_id].append(f)
                    existing.add(f)

        # Compute partial score based on ALL accumulated findings
        reward, feedback, partial_score = self._grade_with_progress(
            task_id, action, self._cumulative_findings.get(task_id, [])
        )

        # Track best score for this task (reward shaping)
        if partial_score > self._task_best_scores.get(task_id, 0.0):
            self._task_best_scores[task_id] = partial_score

        # Update state with best scores
        self._state.best_scores[task_id] = self._task_best_scores[task_id]

        # Determine if task is complete
        task_step = self._task_step_counts[task_id]
        finalize = action.finalize or task_step >= MAX_STEPS_PER_TASK

        if finalize:
            # Use the best score achieved as final reward
            final_score = self._task_best_scores[task_id]
            self._state.cumulative_reward = round(
                self._state.cumulative_reward + final_score, 4
            )

            # Advance to next task
            current_idx = TASK_ORDER.index(task_id)
            next_idx = current_idx + 1
            done = next_idx >= len(TASK_ORDER)

            self._state.done = done

            if not done:
                next_task_id = TASK_ORDER[next_idx]
                self._state.task_id = next_task_id
                self._state.task_step_count = 0
                obs = self._make_observation(
                    task_id=next_task_id,
                    feedback=f"{feedback}\n[Task completed with score: {final_score:.2f}]",
                    score=final_score,
                )
            else:
                # All tasks complete
                obs = RedTeamObservation(
                    task_id=task_id,
                    task_name="All Tasks Complete",
                    difficulty="done",
                    instructions=(
                        f"All tasks completed! "
                        f"Total cumulative reward: {self._state.cumulative_reward:.4f}"
                    ),
                    feedback=f"{feedback}\n[Final score: {final_score:.2f}]",
                    score=final_score,
                    step_count=task_step,
                    max_steps_per_task=MAX_STEPS_PER_TASK,
                    partial_score=final_score,
                    best_score=final_score,
                    done=True,
                    task_complete=True,
                )
        else:
            # Continue with same task, return partial progress
            self._state.task_step_count = task_step
            obs = self._make_observation(
                task_id=task_id,
                feedback=f"{feedback}\n[Partial progress: score={partial_score:.2f}, step {task_step}/{MAX_STEPS_PER_TASK}]",
                score=partial_score,
            )
            obs.partial_score = partial_score
            obs.best_score = self._task_best_scores[task_id]

        return StepResult(
            observation=obs,
            reward=partial_score if not finalize else final_score,
            done=finalize and self._state.done,
            info={
                "task_id": task_id,
                "step": self._state.step_count,
                "task_step": task_step,
                "partial_score": partial_score,
                "best_score": self._task_best_scores[task_id],
                "cumulative_reward": self._state.cumulative_reward,
                "feedback": feedback,
                "finalized": finalize,
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
        task_step = self._task_step_counts.get(task_id, 0)

        if task_id == "pii_detection":
            return RedTeamObservation(
                task_id=task_id,
                task_name=meta["name"],
                difficulty=meta["difficulty"],
                instructions=TASK_PII_INSTRUCTIONS,
                dataset_sample=self._pii_dataset,
                feedback=feedback,
                score=score,
                step_count=task_step,
                max_steps_per_task=MAX_STEPS_PER_TASK,
                partial_score=score,
                best_score=self._task_best_scores.get(task_id, 0.0),
                done=False,
                task_complete=False,
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
                step_count=task_step,
                max_steps_per_task=MAX_STEPS_PER_TASK,
                partial_score=score,
                best_score=self._task_best_scores.get(task_id, 0.0),
                done=False,
                task_complete=False,
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
                step_count=task_step,
                max_steps_per_task=MAX_STEPS_PER_TASK,
                partial_score=score,
                best_score=self._task_best_scores.get(task_id, 0.0),
                done=False,
                task_complete=False,
            )
        else:
            raise ValueError(f"Unknown task_id: {task_id}")

    def _grade_with_progress(
        self,
        task_id: str,
        action: RedTeamAction,
        cumulative_findings: List[str],
    ) -> tuple[float, str, float]:
        """
        Grade the action and return:
          - step_reward: reward for this specific step
          - feedback: grader feedback string
          - partial_score: current best score for this task
        """
        if task_id == "pii_detection":
            reward, feedback = grade_pii(
                findings=cumulative_findings,
                ground_truth=self._pii_ground_truth or [],
            )
        elif task_id == "jailbreak_detection":
            reward, feedback = grade_jailbreak(
                findings=cumulative_findings,
                ground_truth=self._jailbreak_ground_truth or [],
            )
        elif task_id == "system_prompt_hardening":
            reward, feedback = grade_hardening(
                hardened_prompt=action.hardened_prompt,
                changes_summary=action.changes_summary,
            )
        else:
            reward, feedback = 0.0, f"Unknown task: {task_id}"

        # Reward shaping: give small bonus for improvement
        prev_best = self._task_best_scores.get(task_id, 0.0)
        improvement = max(0.0, reward - prev_best)
        step_reward = reward  # Use actual score as step reward

        return step_reward, feedback, reward