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
from skillstate.logging_util import estimate_tokens
from skillstate.ollama_client import (
    OllamaClient,
    default_base_url,
    default_model,
    health_check,
)
from skillstate.policies import (
    chat_skillstate_policy,
    repoops_skillstate_policy,
    warehouse_history_policy,
    warehouse_skillstate_policy,
)
from skillstate.runtime import run_skill_state
from skillstate.skills import load_skill
from skillstate.skills.chat.env import (
    ChatEnv,
    parse_chat_action,
    project_history_prompt,
)

CHAT_DEFAULT_MODEL = "qwen3.8:27b"

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


class ChatStartRequest(BaseModel):
    skill: str = "chat"
    model: str | None = None
    offline: bool = False


class ChatMessageRequest(BaseModel):
    text: str = Field(min_length=1)


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


class ChatSession:
    """One SKILL.state loop. The left column is a projection, never a second agent."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.events: queue.Queue[dict[str, Any] | None] = queue.Queue()
        self.stop = threading.Event()
        self.thread: threading.Thread | None = None
        self.model = CHAT_DEFAULT_MODEL
        self.env: ChatEnv | None = None
        self.running = False
        self.failed = False
        self.offline = False
        self.prior_lines: list[tuple[str, str]] = []
        self.state: dict[str, Any] = {}
        self.last_user_text: str | None = None
        self.instructions = ""
        self.last_validation_error: str | None = None

    def start(self, req: ChatStartRequest, *, resume_state: dict[str, Any] | None = None,
              first_obs: str | None = None) -> None:
        with self.lock:
            self._stop_locked()
            self.stop = threading.Event()
            self.running = True
            self.failed = False
            self.last_validation_error = None
            self.offline = bool(req.offline)
            if req.model:
                self.model = req.model
            if resume_state is None:
                self.prior_lines = []
                self.state = {}
            if first_obs is not None:
                self.last_user_text = first_obs
            self.thread = threading.Thread(
                target=self._run,
                args=(req, resume_state, first_obs),
                daemon=True,
                name="skillstate-chat",
            )
            self.thread.start()

    def _stop_locked(self) -> None:
        self.stop.set()
        if self.env is not None:
            try:
                self.env.feed(None)
            except Exception:
                pass
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=2.0)
        self.thread = None
        self.running = False
        self.env = None

    def reset(self) -> None:
        with self.lock:
            self._stop_locked()
            self.prior_lines = []
            self.state = {}
            self.last_user_text = None
            self.events.put({"type": "reset"})

    def message(self, text: str) -> dict[str, Any]:
        stripped = text.strip()
        if not stripped:
            raise HTTPException(400, "text is empty")
        with self.lock:
            env = self.env
            awaiting = bool(env and env.awaiting_user and not self.stop.is_set())
            failed = self.failed
            state = dict(self.state) if self.state else None
            model = self.model
            offline = self.offline
        if awaiting and env is not None:
            self.last_user_text = stripped
            env.feed(stripped)
            return {"ok": True}
        if failed and state:
            self.start(
                ChatStartRequest(skill="chat", model=model, offline=offline),
                resume_state=state,
                first_obs=stripped,
            )
            return {"ok": True, "retried": True}
        raise HTTPException(409, "chat is not waiting for a user line (send after the assistant replies, or Retry after a validator failure)")

    def _pause_check(self) -> None:
        if self.stop.is_set():
            raise InterruptedError("reset")

    def _run(
        self,
        req: ChatStartRequest,
        resume_state: dict[str, Any] | None,
        first_obs: str | None,
    ) -> None:
        def on_event(event: dict[str, Any]) -> None:
            if self.stop.is_set():
                raise InterruptedError("reset")
            self.events.put(self._enrich(event))

        try:
            skill = load_skill("chat")
            self.instructions = skill.instructions
            env = skill.make_env(0, horizon=128, reset_observation=first_obs)
            env.on_wait = lambda ev: self.events.put(ev)
            self.env = env
            model = req.model or self.model
            if req.offline:
                llm: Any = FakeLLM(chat_skillstate_policy, model="fake")
            else:
                llm = OllamaClient(model=model, base_url=default_base_url(), timeout=1800.0)
            initial = resume_state if resume_state is not None else skill.initial_state()
            self.state = dict(initial)
            result = run_skill_state(
                skill_name=skill.name,
                instructions=skill.instructions,
                state_schema=skill.state_schema,
                initial_state=initial,
                env=env,
                llm=llm,
                parse_action=skill.parse_action,
                max_steps=128,
                seed=0,
                model=getattr(llm, "model", model),
                on_event=on_event,
                pause_check=self._pause_check,
            )
            self.failed = bool(result.failed)
            if result.steps:
                self.state = dict(result.steps[-1].state_after)
        except InterruptedError:
            self.events.put({"type": "reset"})
        except SystemExit as exc:
            self.events.put({"type": "error", "message": f"Ollama unavailable (exit {exc.code})"})
            self.failed = True
        except Exception as exc:  # noqa: BLE001
            self.events.put({"type": "error", "message": f"{type(exc).__name__}: {exc}"})
            self.failed = True
        finally:
            try:
                close = getattr(llm, "close", None)
            except NameError:
                close = None
            if callable(close):
                close()
            self.running = False
            self.events.put({"type": "end"})

    def _enrich(self, event: dict[str, Any]) -> dict[str, Any]:
        kind = event.get("type")
        if kind == "action_ready":
            action = event.get("action") or ""
            parsed = parse_chat_action(action) if action else None
            event["assistant_text"] = parsed.text if parsed else None
            event["prompt_skillstate"] = event.get("prompt") or ""
            obs = event.get("observation") or ""
            projection = project_history_prompt(self.instructions, self.prior_lines, obs)
            event["prompt_history_projection"] = projection
            event["history_prompt_tokens_est"] = estimate_tokens(projection)
            return event
        if kind == "prompt":
            obs = event.get("observation") or ""
            projection = project_history_prompt(self.instructions, self.prior_lines, obs)
            event["prompt_skillstate"] = event.get("prompt") or ""
            event["prompt_history_projection"] = projection
            event["history_prompt_tokens_est"] = estimate_tokens(projection)
            event["state_before"] = event.get("state") or self.state
            return event
        if kind != "step":
            return event
        raw = event.get("step") or {}
        obs = raw.get("observation") or ""
        projection = project_history_prompt(self.instructions, self.prior_lines, obs)
        validation_error = raw.get("validation_error")
        action = raw.get("action") or ""
        parsed = parse_chat_action(action) if action and not validation_error else None
        assistant_text = parsed.text if parsed else None
        if not validation_error:
            if (
                self.last_user_text
                and obs == self.last_user_text
                and (not self.prior_lines or self.prior_lines[-1] != ("user", obs))
            ):
                self.prior_lines.append(("user", obs))
            if assistant_text:
                self.prior_lines.append(("assistant", assistant_text))
            self.state = dict(raw.get("state_after") or self.state)
            self.last_validation_error = None
        else:
            self.last_validation_error = validation_error
        event.update(
            {
                "prompt_skillstate": raw.get("prompt") or "",
                "prompt_history_projection": projection,
                "prompt_tokens": raw.get("prompt_tokens"),
                "history_prompt_tokens_est": estimate_tokens(projection),
                "state_before": raw.get("state_before") or {},
                "state_after": raw.get("state_after") or {},
                "changed_keys": raw.get("changed_keys") or [],
                "observation": obs,
                "reasoning": raw.get("reasoning") or "",
                "state_patch": raw.get("state_patch") or {},
                "action": action,
                "assistant_text": assistant_text,
                "totals": raw.get("totals") or {},
                "validation_error": validation_error,
                "done": raw.get("done"),
                "success": raw.get("success"),
                "env_error": raw.get("env_error"),
            }
        )
        return event


chat_session = ChatSession()


@app.get("/")
def index() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/episode")
def episode() -> FileResponse:
    return FileResponse(STATIC / "index.html")


@app.get("/chat")
def chat_page() -> FileResponse:
    return FileResponse(STATIC / "chat.html")


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


@app.post("/api/chat/start")
def api_chat_start(body: ChatStartRequest) -> dict[str, Any]:
    if body.skill not in {"chat", ""}:
        raise HTTPException(400, "skill must be chat")
    chat_session.start(body)
    skill = load_skill("chat")
    return {
        "ok": True,
        "model": body.model or chat_session.model,
        "offline": body.offline,
        "instructions": skill.instructions,
    }


@app.post("/api/chat/message")
def api_chat_message(body: ChatMessageRequest) -> dict[str, Any]:
    return chat_session.message(body.text)


@app.post("/api/chat/reset")
def api_chat_reset() -> dict[str, bool]:
    chat_session.reset()
    return {"ok": True}


@app.get("/api/chat/stream")
async def api_chat_stream() -> StreamingResponse:
    async def gen():
        loop = asyncio.get_running_loop()
        while True:
            item = await loop.run_in_executor(None, chat_session.events.get)
            if item is None:
                yield "data: {\"type\":\"end\"}\n\n"
                continue
            yield f"data: {json.dumps(item, default=str)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream")


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
