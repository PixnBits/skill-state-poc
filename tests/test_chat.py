"""Chat skill: grammar, env, FakeLLM episode. No Ollama."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from skillstate.fake_llm import FakeLLM, SequenceLLM
from skillstate.policies import chat_skillstate_policy
from skillstate.runtime import run_skill_state
from skillstate.skills.chat.env import (
    ChatEnv,
    parse_chat_action,
    project_history_prompt,
)
from skillstate.skills.chat.schema import ChatState, initial_chat_state
from skillstate.skills.chat import ChatSkill
from skillstate.schemas import apply_validated_patch


def test_chat_page_is_static_html():
    from fastapi.testclient import TestClient
    from skillstate.ui.app import app

    client = TestClient(app)
    page = client.get("/chat")
    assert page.status_code == 200
    assert b"History (what ReAct would keep)" in page.content
    assert b"SKILL.state (what the model actually received)" in page.content
    assert b'<textarea id="input"' in page.content
    assert b'<textarea id="input" placeholder' in page.content
    assert b'<textarea id="input" placeholder="Type a message. Enter sends, Shift+Enter newline." disabled>' not in page.content
    home = client.get("/")
    assert b'href="/chat"' in home.content
    assert client.get("/episode").status_code == 200


def test_parse_say_ask_done_with_spaces():
    say = parse_chat_action("SAY Hello there, Ada.")
    assert say is not None and say.op == "SAY" and say.text == "Hello there, Ada."
    ask = parse_chat_action("ASK What do you want to work on?")
    assert ask is not None and ask.op == "ASK"
    done = parse_chat_action("DONE Glad we could talk.")
    assert done is not None and done.op == "DONE"


def test_parse_rejects_wait_and_junk():
    assert parse_chat_action("WAIT") is None
    assert parse_chat_action("SAY") is None
    assert parse_chat_action("SAY ") is None
    assert parse_chat_action("HELLO there") is None
    assert parse_chat_action("") is None


def test_schema_extra_forbid():
    with pytest.raises(ValidationError):
        ChatState.model_validate({**initial_chat_state(), "secret": "nope"})


def test_null_facts_clear_the_list():
    state = initial_chat_state()
    filled = apply_validated_patch(
        state, {"facts": ["user_name=Ada"]}, ChatState
    )
    assert filled["facts"] == ["user_name=Ada"]
    cleared = apply_validated_patch(filled, {"facts": None}, ChatState)
    assert cleared["facts"] == []


def test_env_human_starts_then_feed():
    env = ChatEnv()
    env.feed("My name is Ada")
    assert env.reset() == "My name is Ada"
    env.feed("that's enough")
    obs, done, info = env.step("SAY Hello Ada.")
    assert info["valid"] is True
    assert info["assistant_text"] == "Hello Ada."
    assert obs == "that's enough"
    assert done is False
    obs, done, info = env.step("DONE Glad we could talk.")
    assert done is True
    assert env.success()


def test_env_rejects_wait():
    env = ChatEnv()
    env.feed("hello")
    env.reset()
    obs, done, info = env.step("WAIT")
    assert info["valid"] is False
    assert done is False


def test_history_projection_is_not_the_skillstate_prompt():
    skill = ChatSkill()
    proj = project_history_prompt(
        skill.instructions,
        [("assistant", "Hello."), ("user", "My name is Ada")],
        "What is my name?",
    )
    assert proj.startswith(skill.instructions.strip())
    assert "History:\n" in proj
    assert "User: My name is Ada" in proj
    assert "Latest Observation: What is my name?" in proj


def test_fake_llm_episode_two_user_turns_and_no_transcript_leak():
    skill = ChatSkill()
    env = ChatEnv()
    env.feed("My name is Ada. I want a warehouse demo.")
    env.feed("What is my name?")
    env.feed("that's enough")
    llm = FakeLLM(chat_skillstate_policy, model="fake")
    result = run_skill_state(
        skill_name=skill.name,
        instructions=skill.instructions,
        state_schema=skill.state_schema,
        initial_state=skill.initial_state(),
        env=env,
        llm=llm,
        parse_action=skill.parse_action,
        max_steps=8,
        seed=0,
        model="fake",
    )
    assert result.success, result.fail_reason
    actions = [s.action for s in result.steps if not s.validation_error]
    assert actions[0].startswith("SAY ")
    assert any(a.startswith("SAY ") for a in actions[1:])
    assert actions[-1].startswith("DONE ")
    for prompt in llm.prompts:
        assert "History:" not in prompt
        assert prompt.count("Latest Observation:") == 1
        assert "Reasoning & Action:" not in prompt
    # After the first user turn, the raw utterance must not re-enter A_t;
    # only Σ facts/goal may remember a short digest.
    later = llm.prompts[1:]
    assert later, "expected a later prompt after the first user line"
    for prompt in later:
        obs = prompt.split("Latest Observation:", 1)[1]
        state_json = prompt.split("Skill Execution State:", 1)[1].split("Latest Observation:", 1)[0]
        # P's example may mention this sentence; O_t and Σ must not replay it.
        assert "My name is Ada. I want a warehouse demo." not in obs
        assert "My name is Ada. I want a warehouse demo." not in state_json
    name_obs = [
        p for p in llm.prompts if "What is my name?" in p.split("Latest Observation:", 1)[-1]
    ]
    assert name_obs
    assert "user_name=Ada" in name_obs[0]
    assert result.extra.get("wall_s", 0) >= 0


def test_validator_error_does_not_execute():
    skill = ChatSkill()
    env = ChatEnv()
    env.feed("hello")
    env.feed("follow-up")
    llm = SequenceLLM(
        [
            "not json",
            'ok\n```json\n{"state_patch": {"last_action": "SAY Hi."}, "action": "SAY Hi."}\n```',
        ]
    )
    result = run_skill_state(
        skill_name=skill.name,
        instructions=skill.instructions,
        state_schema=skill.state_schema,
        initial_state=skill.initial_state(),
        env=env,
        llm=llm,
        parse_action=skill.parse_action,
        max_steps=2,
        seed=0,
        model="fake",
    )
    assert result.steps[0].validation_error
    assert result.steps[0].action == ""
    assert env.n_valid_actions == 1
    assert result.steps[1].action.startswith("SAY ")
