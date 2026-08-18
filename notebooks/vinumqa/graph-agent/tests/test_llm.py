"""Tests for the API transport's retry/escalation logic. No real API calls.

`LLMClient.client` (the OpenAI SDK instance) is replaced with a small fake
that returns scripted responses, so these are deterministic -- unlike a real
model, which may or may not reproduce a given failure mode on any given call.

Motivated by a real, measured failure: DeepSeek-V4-Flash against this
package's actual planner prompt, at the default max_tokens_planner=768, spent
728 of 768 completion tokens on reasoning_content and left 69 characters for
the plan -- one harder question away from an empty completion on every one of
`max_retries` identical attempts. `_call()`'s reasoning-budget escalation
(llm.py) is what these tests pin down.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.config import AgentConfig  # noqa: E402
from agentic.llm import (  # noqa: E402
    KNOWN_REASONING_MODELS,
    REASONING_MODEL_MIN_TOKENS,
    LLMClient,
    LLMError,
    is_reasoning_model,
)


def _usage(total: int = 20) -> SimpleNamespace:
    return SimpleNamespace(prompt_tokens=10, completion_tokens=total - 10, total_tokens=total)


def _choice(content: str | None, reasoning_content: str | None = None,
            finish_reason: str = "stop") -> SimpleNamespace:
    message = SimpleNamespace(content=content, reasoning_content=reasoning_content)
    return SimpleNamespace(message=message, finish_reason=finish_reason)


class _ScriptedCompletions:
    """Fake `client.chat.completions`: returns the next scripted response.

    Records every call's `max_tokens` so tests can assert the escalation
    actually grew the budget, not just that a later scripted response won.
    """

    def __init__(self, responses: list[SimpleNamespace]):
        self._responses = list(responses)
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if not self._responses:
            raise AssertionError("scripted responses exhausted -- test sent too many requests")
        return self._responses.pop(0)


def _client(responses: list[SimpleNamespace], **config_kwargs) -> tuple[LLMClient, _ScriptedCompletions]:
    config = AgentConfig(**config_kwargs)
    client = LLMClient(config, api_key="test", base_url="http://test.invalid")
    completions = _ScriptedCompletions(responses)
    client.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return client, completions


# ==================================================== reasoning escalation


def test_escalates_max_tokens_when_reasoning_starved_then_succeeds():
    """Empty content + non-empty reasoning_content -> retry at 2x, no sleep."""
    responses = [
        SimpleNamespace(choices=[_choice(content=None, reasoning_content="thinking...")],
                        usage=_usage()),
        SimpleNamespace(choices=[_choice(content="add(1, 2), join()")], usage=_usage()),
    ]
    client, completions = _client(responses, max_retries=5, retry_base_delay=0)

    result = client.complete(system=None, user="q", model="m", max_tokens=100)

    assert result == "add(1, 2), join()"
    assert len(completions.calls) == 2
    assert completions.calls[0]["max_tokens"] == 100
    assert completions.calls[1]["max_tokens"] == 200  # doubled once


def test_escalation_is_capped_at_two_doublings():
    """Never-satisfied reasoning starvation must still terminate, not loop forever."""
    starved = SimpleNamespace(choices=[_choice(content=None, reasoning_content="thinking...")],
                              usage=_usage())
    # 5 retries: attempts 1-3 can each escalate (cap=2 doublings -> 2 escalations
    # actually fire, on attempts 1 and 2), attempts 3-5 retry at the capped budget.
    client, completions = _client([starved] * 5, max_retries=5, retry_base_delay=0)

    with pytest.raises(LLMError, match="all 5 attempts returned empty output"):
        client.complete(system=None, user="q", model="m", max_tokens=100)

    tokens_requested = [c["max_tokens"] for c in completions.calls]
    assert tokens_requested == [100, 200, 400, 400, 400]  # caps at 4x (2 doublings)


def test_does_not_escalate_when_content_is_empty_for_a_different_reason():
    """No reasoning_content at all -> this is a different failure; do not grow the budget."""
    empty = SimpleNamespace(choices=[_choice(content=None, reasoning_content=None)], usage=_usage())
    client, completions = _client([empty] * 3, max_retries=3, retry_base_delay=0)

    with pytest.raises(LLMError, match="all 3 attempts returned empty output"):
        client.complete(system=None, user="q", model="m", max_tokens=100)

    assert [c["max_tokens"] for c in completions.calls] == [100, 100, 100]


# ============================================= known-reasoning-model floor


@pytest.mark.parametrize("model", sorted(KNOWN_REASONING_MODELS))
def test_confirmed_reasoning_models_are_detected_without_name_matching(model):
    """DeepSeek-V4-Flash and GLM-5.2 match neither r1/thinking/qwq/o1/o3 --
    this is what makes the keyword heuristic alone insufficient for them."""
    assert is_reasoning_model(model)
    assert not any(kw in model.lower() for kw in
                    ("r1", "thinking", "reasoner", "qwq", "o1", "o3", "reasoning"))


def test_keyword_heuristic_still_catches_unlisted_reasoning_models():
    assert is_reasoning_model("qwen3-4b-thinking")
    assert is_reasoning_model("some-r1-distill")
    assert not is_reasoning_model("gemma-4-31B-it")


def test_known_reasoning_model_gets_a_raised_floor_on_the_first_attempt():
    """A caller asking for 100 tokens on DeepSeek-V4-Flash should not even get
    to try at 100 -- the first real request already asks for the floor."""
    responses = [
        SimpleNamespace(choices=[_choice(content="add(1, 2), join()")], usage=_usage()),
    ]
    client, completions = _client(responses, max_retries=5, retry_base_delay=0)

    client.complete(system=None, user="q", model="DeepSeek-V4-Flash", max_tokens=100)

    assert completions.calls[0]["max_tokens"] == REASONING_MODEL_MIN_TOKENS


def test_reasoning_floor_does_not_lower_an_already_larger_request():
    responses = [SimpleNamespace(choices=[_choice(content="ok")], usage=_usage())]
    client, completions = _client(responses, max_retries=5, retry_base_delay=0)

    client.complete(system=None, user="q", model="GLM-5.2",
                     max_tokens=REASONING_MODEL_MIN_TOKENS * 4)

    assert completions.calls[0]["max_tokens"] == REASONING_MODEL_MIN_TOKENS * 4


def test_non_reasoning_model_is_not_affected_by_the_floor():
    responses = [SimpleNamespace(choices=[_choice(content="ok")], usage=_usage())]
    client, completions = _client(responses, max_retries=5, retry_base_delay=0)

    client.complete(system=None, user="q", model="gemma-4-31B-it", max_tokens=100)

    assert completions.calls[0]["max_tokens"] == 100


def test_truncated_content_is_still_preferred_over_raising():
    """finish_reason='length' with SOME content: keep it as a last resort (pre-existing behaviour)."""
    truncated = SimpleNamespace(
        choices=[_choice(content="add(1, 2", finish_reason="length")], usage=_usage()
    )
    client, completions = _client([truncated] * 2, max_retries=2, retry_base_delay=0)

    result = client.complete(system=None, user="q", model="m", max_tokens=100)

    assert result == "add(1, 2"
    # Truncated (non-empty) content is not the reasoning-starved case, so no escalation.
    assert [c["max_tokens"] for c in completions.calls] == [100, 100]
