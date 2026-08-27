import json

import pytest

from stanr.distiller import ReasoningDistiller, _is_pa_match, _is_strict_match

SAMPLE = {
    "pre_text": ["text before"],
    "table": [["", "2018"], ["revenue", "100"]],
    "post_text": ["text after"],
    "id": "s1",
    "qa": {"question": "what is revenue?", "program": "table_sum(revenue, none)", "exe_ans": "100"},
}


def test_is_pa_match_accepts_reordered_commutative():
    assert _is_pa_match("add(2, 1)", "add(1, 2)") is True


def test_is_strict_match_rejects_reordered():
    assert _is_strict_match("add(2, 1)", "add(1, 2)") is False


def test_is_strict_match_accepts_identical():
    assert _is_strict_match("add(1, 2)", "add(1, 2)") is True


def test_is_strict_match_normalizes_number_formatting():
    assert _is_strict_match("divide(100.00, 2)", "divide(100, 2)") is True


def test_invalid_filter_with_raises(tmp_path):
    input_path = tmp_path / "in.json"
    input_path.write_text(json.dumps([SAMPLE]), encoding="utf-8")
    with pytest.raises(ValueError):
        ReasoningDistiller(
            model_name="x", api_key="k",
            input_dataset=str(input_path),
            output_distill_path=str(tmp_path / "out.json"),
            filter_with="not-a-real-mode",
        )


class _FakeChoice:
    def __init__(self, content):
        self.message = type("M", (), {"content": content})()


class _FakeResponse:
    def __init__(self, content):
        self.choices = [_FakeChoice(content)]


class _FakeChatCompletions:
    def __init__(self, response_text):
        self.response_text = response_text

    def create(self, **kwargs):
        return _FakeResponse(self.response_text)


class _FakeChat:
    def __init__(self, response_text):
        self.completions = _FakeChatCompletions(response_text)


class _FakeClient:
    def __init__(self, response_text):
        self.chat = _FakeChat(response_text)


def _distiller(tmp_path, **overrides):
    input_path = tmp_path / "in.json"
    input_path.write_text(json.dumps([SAMPLE]), encoding="utf-8")
    kwargs = dict(
        model_name="x", api_key="k",
        input_dataset=str(input_path),
        output_distill_path=str(tmp_path / "out.json"),
        filter_with="pa",
    )
    kwargs.update(overrides)
    return ReasoningDistiller(**kwargs)


def test_run_keeps_only_verified_samples(tmp_path, monkeypatch):
    rd = _distiller(tmp_path)
    fake_text = "<think>because the row says so</think>table_sum(revenue, none)"
    monkeypatch.setattr(rd, "_client", lambda: _FakeClient(fake_text))

    summary = rd.run()

    assert summary["verified"] == 1
    assert summary["total"] == 1
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert out[0]["qa"]["reasoning_trace"] == "because the row says so"
    assert out[0]["qa"]["trace_source"] == "independent"
    assert out[0]["qa"]["program"] == SAMPLE["qa"]["program"]  # gold, untouched


def test_run_discards_unverified_samples(tmp_path, monkeypatch):
    rd = _distiller(tmp_path)
    fake_text = "<think>wrong reasoning</think>add(1, 2)"  # does not match gold
    monkeypatch.setattr(rd, "_client", lambda: _FakeClient(fake_text))

    summary = rd.run()

    assert summary["verified"] == 0
    out = json.loads((tmp_path / "out.json").read_text(encoding="utf-8"))
    assert out == []


def test_run_is_resumable_via_checkpoint(tmp_path, monkeypatch):
    rd = _distiller(tmp_path)
    fake_text = "<think>ok</think>table_sum(revenue, none)"
    calls = {"n": 0}

    class _CountingClient(_FakeClient):
        def __init__(self):
            super().__init__(fake_text)

    def _client():
        calls["n"] += 1
        return _CountingClient()

    monkeypatch.setattr(rd, "_client", _client)
    rd.run()
    assert calls["n"] == 1

    # A second run should find everything already verified in the checkpoint
    # and make no new client calls at all.
    rd2 = _distiller(tmp_path)
    monkeypatch.setattr(rd2, "_client", _client)
    rd2.run()
    assert calls["n"] == 1  # unchanged -- no new API calls needed
