"""Single-page dashboard. Static HTML, no Node build step."""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from skillstate.fake_llm import FakeLLM
from skillstate.history_runtime import run_history
from skillstate.ollama_client import (
    OllamaClient,
    default_base_url,
    default_model,
    health_check,
)
from skillstate.policies import (
    repoops_skillstate_policy,
    warehouse_history_policy,
    warehouse_skillstate_policy,
)
from skillstate.runtime import run_skill_state
from skillstate.skills import load_skill

STATIC = Path(__file__).with_name("static")

app = FastAPI(title="SKILL.state PoC", version="0.1.0")


class StartRequest(BaseModel):
    skill: str = "warehouse"
    runtime: str = "skillstate"
    model: str | None = None
    seed: int = 42
    max_steps: int = 36
    compact: bool = False
    offline: bool = False


class ModelRequest(BaseModel):
    model: str = Field(min_length=1)


class Runner:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.pause = threading.Event()
        self.pause.set()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.model = default_model()
        self.running = False

    def start(self, req: StartRequest) -> None:
        with self.lock:
            self._stop_locked()
            self.events = queue.Queue()
            self.stop = threading.Event()
            self.pause = threading.Event()
            self.pause.set()
            self.running = True
            if req.model:
                self.model = req.model
            self.thread = threading.Thread(
                target=self._run, args=(req,), daemon=True, name="skillstate-episode"
            )
            self.thread.start()

    def _stop_locked(self) -> None:
        self.stop.set()
        self.pause.set()
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=1.5)
        self.thread = None
        self.running = False

    def reset(self) -> None:
        with self.lock:
            self._stop_locked()
            self.events = queue.Queue()
            self.events.put({"type": "reset"})
            self.events.put(None)

    def set_paused(self, paused: bool) -> None:
        if paused:
            self.pause.clear()
        else:
            self.pause.set()

    def _pause_check(self) -> None:
        while not self.pause.is_set():
            if self.stop.is_set():
                raise InterruptedError("reset")
            self.stop.wait(0.05)
            if self.stop.is_set():
                raise InterruptedError("reset")

    def _run(self, req: StartRequest) -> None:
        def on_event(event: dict[str, Any]) -> None:
            if self.stop.is_set():
                raise InterruptedError("reset")
            self.events.put(event)

        try:
            skill = load_skill(req.skill)
            model = req.model or self.model
            env_kwargs: dict[str, Any] = {"horizon": req.max_steps}
            if req.skill == "warehouse":
                env_kwargs["compact"] = req.compact
            env = skill.make_env(req.seed, **env_kwargs)

            if req.offline:
                if req.runtime == "history":
                    llm: Any = FakeLLM(warehouse_history_policy, model="fake")
                elif req.skill == "repoops":
                    llm = FakeLLM(repoops_skillstate_policy, model="fake")
                else:
                    llm = FakeLLM(warehouse_skillstate_policy, model="fake")
            else:
                llm = OllamaClient(model=model, base_url=default_base_url())

            common = dict(
                skill_name=skill.name,
                instructions=skill.instructions,
                env=env,
                llm=llm,
                parse_action=skill.parse_action,
                max_steps=req.max_steps,
                seed=req.seed,
                model=getattr(llm, "model", model),
                on_event=on_event,
                pause_check=self._pause_check,
            )
            if req.runtime == "history":
                run_history(initial_state=skill.initial_state(), **common)
            else:
                run_skill_state(
                    state_schema=skill.state_schema,
                    initial_state=skill.initial_state(),
                    **common,
                )
        except InterruptedError:
            self.events.put({"type": "reset"})
        except SystemExit as exc:
            self.events.put({"type": "error", "message": f"Ollama unavailable (exit {exc.code})"})
        except Exception as exc:  # noqa: BLE001 — surface any episode crash to the UI
            self.events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
        finally:
            self.running = False
            self.events.put(None)


runner = Runner()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/api/health")
def api_health() -> dict[str, Any]:
    try:
        info = health_check(require_model=False)
        info["selected"] = runner.model
        info["running"] = runner.running
        return info
    except SystemExit:
        from skillstate.ollama_client import INSTALL_HELP, DEFAULT_MODEL

        return {
            "ok": False,
            "models": [],
            "selected": runner.model,
            "running": runner.running,
            "install_help": INSTALL_HELP.format(
                base="http://127.0.0.1:11434", model=os.environ.get("OLLAMA_MODEL", DEFAULT_MODEL)
            ),
        }


@app.post("/api/model")
def api_model(body: ModelRequest) -> dict[str, str]:
    runner.model = body.model.strip()
    os.environ["OLLAMA_MODEL"] = runner.model
    return {"model": runner.model}


@app.post("/api/start")
def api_start(body: StartRequest) -> dict[str, Any]:
    if body.skill not in {"warehouse", "repoops"}:
        raise HTTPException(400, "skill must be warehouse or repoops")
    if body.runtime not in {"skillstate", "history"}:
        raise HTTPException(400, "runtime must be skillstate or history")
    runner.start(body)
    return {"ok": True, "model": body.model or runner.model}


@app.post("/api/pause")
def api_pause() -> dict[str, bool]:
    """Toggle pause. ``paused=true`` means the episode is holding between steps."""
    currently_running = runner.pause.is_set()
    runner.set_paused(currently_running)
    return {"paused": not runner.pause.is_set()}


@app.post("/api/pause/{wanted}")
def api_pause_set(wanted: str) -> dict[str, bool]:
    runner.set_paused(wanted.lower() in {"1", "true", "yes", "pause"})
    return {"paused": not runner.pause.is_set()}


@app.post("/api/reset")
def api_reset() -> dict[str, bool]:
    runner.reset()
    return {"ok": True}


@app.get("/api/stream")
async def api_stream() -> StreamingResponse:
    async def gen():
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, runner.events.get)
            if item is None:
                yield "data: {\"type\":\"end\"}\n\n"
                break
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")
