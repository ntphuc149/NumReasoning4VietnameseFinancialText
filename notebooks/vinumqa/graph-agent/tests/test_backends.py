"""Tests for backend routing. No API calls, no GPU, no model download.

Model loading and generate() only happen inside LocalBackend._ensure_loaded()/
_generate(), never touched here -- these tests cover the parts of backends.py
that are pure functions or that only construct objects (construction is lazy;
MultiModelClient does not build a backend until complete()/sample_n() is
actually called for that model).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.backends import (  # noqa: E402
    API_MODELS,
    MODEL_REGISTRY,
    LocalBackend,
    MultiModelClient,
    build_messages,
    describe_backend,
    strip_think,
)
from agentic.config import AgentConfig  # noqa: E402

# The eleven baseline models this repo runs, exactly as named in the root
# README's comparison table. If this list and MODEL_REGISTRY/API_MODELS drift
# apart, one of the two tests below fails loudly instead of a model silently
# routing nowhere useful.
ALL_BASELINE_MODELS = [
    "Qwen3-4B", "qwen3-4b-thinking", "Gemma3-4B",
    "gemma-3-27b-it", "gemma-4-31B-it", "gpt-oss-20b", "gpt-oss-120b",
    "Llama-3.3-70B-Instruct", "DeepSeek-V4-Flash", "GLM-5.2", "gpt-5-nano",
]


# ================================================================== routing


def test_every_baseline_model_is_classified():
    assert set(MODEL_REGISTRY) | set(API_MODELS) == set(ALL_BASELINE_MODELS)


def test_registry_and_api_list_do_not_overlap():
    assert set(MODEL_REGISTRY).isdisjoint(API_MODELS)


@pytest.mark.parametrize("model", ["Qwen3-4B", "qwen3-4b-thinking", "Gemma3-4B"])
def test_local_models_route_local(model):
    assert describe_backend(model) == "local"


@pytest.mark.parametrize("model", API_MODELS)
def test_api_models_route_api(model):
    assert describe_backend(model) == "api"


def test_unknown_model_name_falls_through_to_api():
    # Not an allowlist: a model the endpoint serves under a name not in
    # API_MODELS must still work, by falling through rather than erroring.
    assert describe_backend("some-new-model-not-yet-listed") == "api"


def test_multi_model_client_builds_local_backend_lazily_without_loading_it():
    """Constructing the router, or asking it to route, must not touch a GPU.

    _backend_for only constructs LocalBackend (cheap: a dataclass and a lock);
    the actual `from_pretrained` call lives behind `complete()`/`sample_n()`,
    never exercised here.
    """
    client = MultiModelClient(AgentConfig())
    backend = client._backend_for("Qwen3-4B")
    assert isinstance(backend, LocalBackend)
    assert backend._model is None  # not loaded -- no GPU/network touched
    # Same model name must reuse the same backend instance, not reload it.
    assert client._backend_for("Qwen3-4B") is backend


def test_multi_model_client_does_not_require_api_credentials_for_local_only_use():
    """A local-only run must not need API_KEY/BASE_URL set at all."""
    client = MultiModelClient(AgentConfig())
    for model in MODEL_REGISTRY:
        client._backend_for(model)  # must not raise KeyError on os.environ
    assert client._api_backend is None  # API client never constructed


def test_local_max_batch_size_defaults_to_none():
    """Unset -> LocalBackend's own optimistic batch_size=n behaviour, unchanged."""
    client = MultiModelClient(AgentConfig())
    backend = client._backend_for("Qwen3-4B")
    assert backend.max_batch_size is None


def test_local_max_batch_size_is_threaded_from_config_to_the_backend():
    """Kaggle-T4 knob: config.local_max_batch_size must reach the LocalBackend
    that actually does the n-sampling generate() call, not just sit unused."""
    client = MultiModelClient(AgentConfig(local_max_batch_size=4))
    backend = client._backend_for("Qwen3-4B")
    assert backend.max_batch_size == 4


def test_multi_model_client_usage_aggregates_across_backends():
    client = MultiModelClient(AgentConfig())
    backend = client._backend_for("Qwen3-4B")
    backend.usage.add(prompt=10, completion=5, total=15)
    assert client.usage.total_tokens == 15
    assert client.usage.requests == 1


# =========================================================== message shape


def test_qwen_messages_use_plain_string_content():
    messages = build_messages("qwen", system="sys", user="hi")
    assert messages == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
    ]


def test_qwen_messages_omit_system_when_none():
    messages = build_messages("qwen", system=None, user="hi")
    assert messages == [{"role": "user", "content": "hi"}]


def test_gemma_messages_use_typed_content_parts():
    messages = build_messages("gemma", system="sys", user="hi")
    assert messages == [
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "text", "text": "hi"}]},
    ]


def test_gemma_messages_omit_system_when_empty_string():
    # An empty string is falsy, same as None -- no node in this package ever
    # wants a blank system turn.
    messages = build_messages("gemma", system="", user="hi")
    assert messages == [{"role": "user", "content": [{"type": "text", "text": "hi"}]}]


# ================================================================= </think>


class _FakeTokenizer:
    """Minimal stand-in: only what strip_think touches."""

    def __init__(self, think_end_id: int = 999):
        self._think_end_id = think_end_id

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._think_end_id if token == "</think>" else -1

    def decode(self, ids, skip_special_tokens: bool = True) -> str:
        return " ".join(str(i) for i in ids)


def test_strip_think_keeps_only_what_follows_the_final_marker():
    tok = _FakeTokenizer(think_end_id=999)
    # 1, 2 = "reasoning", 999 = </think>, 3, 4 = the real answer
    ids = [1, 2, 999, 3, 4]
    assert strip_think(tok, ids) == "3 4"


def test_strip_think_searches_from_the_end():
    """A </think>-like token mentioned inside the reasoning must not fool this."""
    tok = _FakeTokenizer(think_end_id=999)
    ids = [1, 999, 2, 999, 3]  # only the LAST 999 is the real close
    assert strip_think(tok, ids) == "3"


def test_strip_think_with_no_marker_returns_everything():
    """Cut off before closing </think>: nothing to slice, whole text comes back."""
    tok = _FakeTokenizer(think_end_id=999)
    ids = [1, 2, 3]
    assert strip_think(tok, ids) == "1 2 3"


def test_strip_think_falls_back_to_the_known_constant_if_lookup_fails():
    from agentic.backends import QWEN3_THINK_END_TOKEN_ID

    class _NoLookupTokenizer(_FakeTokenizer):
        def convert_tokens_to_ids(self, token: str) -> int:
            return -1  # simulates a tokenizer that cannot resolve the token

    tok = _NoLookupTokenizer()
    ids = [1, QWEN3_THINK_END_TOKEN_ID, 2]
    assert strip_think(tok, ids) == "2"
