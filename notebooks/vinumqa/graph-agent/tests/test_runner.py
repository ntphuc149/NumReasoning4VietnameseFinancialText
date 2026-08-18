"""Tests for Runner.run()'s show_progress behaviour. No API calls, no GPU.

A stub one-node graph stands in for the real four-node pipeline -- Runner
itself does not care what a node does, only that it turns an AgentState into
another AgentState, so this exercises the checkpoint/progress machinery
without ever calling a model.
"""

from __future__ import annotations

import builtins
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.agents import AgentGraph, AgentState, Node  # noqa: E402
from agentic.config import AgentConfig, RunConfig  # noqa: E402
from agentic.runner import Runner  # noqa: E402


class _StubNode(Node):
    """Deterministic, no I/O: sets a program from the sample's own id."""

    name = "stub"

    def execute(self, state: AgentState) -> AgentState:
        state.program = f"add({state.sample_id}, 1)"
        return state


def _stub_runner(tmp_path, run_name: str = "stub-run", **run_kwargs) -> Runner:
    graph = AgentGraph()
    graph.add(_StubNode())
    config = RunConfig(
        output_dir=str(tmp_path), run_name=run_name, agent=AgentConfig(), **run_kwargs,
    )
    # client=object() -- never touched, since _StubNode never calls it.
    return Runner(config, client=object(), graph=graph)


def _samples(n: int) -> list[dict]:
    return [
        {
            "id": str(i),
            "pre_text": [], "post_text": [], "table": [],
            "qa": {"question": "q", "program": "", "exe_ans": ""},
        }
        for i in range(n)
    ]


# ============================================================ show_progress


def test_run_without_progress_is_unaffected(tmp_path):
    runner = _stub_runner(tmp_path)
    df = runner.run(_samples(5))
    assert len(df) == 5
    assert (df["generated_program"] == df["id"].map(lambda i: f"add({i}, 1)")).all()


def test_run_with_progress_produces_the_same_results(tmp_path):
    runner = _stub_runner(tmp_path, run_name="progress-run")
    df = runner.run(_samples(5), show_progress=True)
    assert len(df) == 5
    assert (df["generated_program"] == df["id"].map(lambda i: f"add({i}, 1)")).all()


def test_show_progress_falls_back_gracefully_when_tqdm_is_unavailable(tmp_path, monkeypatch, capsys):
    """Simulates tqdm not being installed: must not crash, must say so once."""
    real_import = builtins.__import__

    def _no_tqdm(name, *args, **kwargs):
        if name == "tqdm":
            raise ImportError("simulated: tqdm not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", _no_tqdm)

    runner = _stub_runner(tmp_path, run_name="no-tqdm-run")
    df = runner.run(_samples(3), show_progress=True)

    assert len(df) == 3
    out = capsys.readouterr().out
    assert "tqdm is not installed" in out


def test_show_progress_has_no_effect_when_everything_is_already_checkpointed(tmp_path):
    """Resuming a fully-done run must not touch tqdm/progress at all -- todo is empty."""
    runner = _stub_runner(tmp_path, run_name="resumed-run")
    runner.run(_samples(3))  # first pass: populates the checkpoint

    resumed = _stub_runner(tmp_path, run_name="resumed-run")
    df = resumed.run(_samples(3), show_progress=True)  # nothing left to do
    assert len(df) == 3


def test_checkpoint_resume_skips_already_done_samples(tmp_path):
    runner = _stub_runner(tmp_path, run_name="partial-run")
    runner.run(_samples(2))

    resumed = _stub_runner(tmp_path, run_name="partial-run")
    df = resumed.run(_samples(5), show_progress=True)
    assert len(df) == 5
    assert (df["generated_program"] == df["id"].map(lambda i: f"add({i}, 1)")).all()
