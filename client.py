"""
client.py — EnvClient for the Ethical Red-Teamer environment.

Usage (async):
    async with RedTeamEnv(base_url="https://your-space.hf.space") as client:
        obs = await client.reset()
        result = await client.step(action)
        state = await client.state()

Usage (sync):
    with RedTeamEnv(base_url="...").sync() as client:
        obs = client.reset()
        result = client.step(action)
"""

from __future__ import annotations

import asyncio
from contextlib import contextmanager, asynccontextmanager
from typing import Optional

import httpx

from models import (
    RedTeamAction,
    RedTeamObservation,
    RedTeamState,
    StepResult,
)


class RedTeamEnv:
    """Async client for the Ethical Red-Teamer OpenEnv environment."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._client: Optional[httpx.AsyncClient] = None

    # ------------------------------------------------------------------
    # Async context manager
    # ------------------------------------------------------------------

    async def __aenter__(self) -> "RedTeamEnv":
        self._client = httpx.AsyncClient(base_url=self.base_url, timeout=60.0)
        return self

    async def __aexit__(self, *args) -> None:
        if self._client:
            await self._client.aclose()

    # ------------------------------------------------------------------
    # OpenEnv API
    # ------------------------------------------------------------------

    async def reset(self) -> RedTeamObservation:
        """Reset the environment and return the initial observation."""
        assert self._client, "Use 'async with RedTeamEnv(...) as client:'"
        resp = await self._client.post("/reset")
        resp.raise_for_status()
        return RedTeamObservation(**resp.json())

    async def step(self, action: RedTeamAction) -> StepResult:
        """Submit an action and receive the graded StepResult."""
        assert self._client, "Use 'async with RedTeamEnv(...) as client:'"
        resp = await self._client.post("/step", content=action.model_dump_json())
        resp.raise_for_status()
        return StepResult(**resp.json())

    async def state(self) -> RedTeamState:
        """Fetch current episode state metadata."""
        assert self._client, "Use 'async with RedTeamEnv(...) as client:'"
        resp = await self._client.get("/state")
        resp.raise_for_status()
        return RedTeamState(**resp.json())

    # ------------------------------------------------------------------
    # Sync wrapper
    # ------------------------------------------------------------------

    def sync(self) -> "_SyncRedTeamEnv":
        return _SyncRedTeamEnv(self.base_url)


class _SyncRedTeamEnv:
    """Synchronous wrapper around the async RedTeamEnv client."""

    def __init__(self, base_url: str):
        self.base_url = base_url
        self._client: Optional[httpx.Client] = None

    def __enter__(self) -> "_SyncRedTeamEnv":
        self._client = httpx.Client(base_url=self.base_url, timeout=60.0)
        return self

    def __exit__(self, *args) -> None:
        if self._client:
            self._client.close()

    def reset(self) -> RedTeamObservation:
        assert self._client
        resp = self._client.post("/reset")
        resp.raise_for_status()
        return RedTeamObservation(**resp.json())

    def step(self, action: RedTeamAction) -> StepResult:
        assert self._client
        resp = self._client.post("/step", content=action.model_dump_json())
        resp.raise_for_status()
        return StepResult(**resp.json())

    def state(self) -> RedTeamState:
        assert self._client
        resp = self._client.get("/state")
        resp.raise_for_status()
        return RedTeamState(**resp.json())
