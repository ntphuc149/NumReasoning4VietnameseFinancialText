"""Tests for the opt-in prompt patches (prompts.py). No API calls."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic import prompts  # noqa: E402


def test_placeholder_clarification_is_included_in_both_languages():
    assert "giá_trị_mới" in prompts.planner_extension("vi")
    assert "new_value" in prompts.planner_extension("en")


def test_placeholder_clarification_explicitly_forbids_the_literal_placeholder():
    # The exact failure measured on a real run: the model echoing the
    # example's placeholder name back as if it were a value.
    assert "KHÔNG được viết nguyên văn" in prompts.PLACEHOLDER_CLARIFICATION_VI
    assert "NEVER write the literal words" in prompts.PLACEHOLDER_CLARIFICATION_EN


def test_planner_extension_still_includes_the_percent_and_operator_patches():
    vi = prompts.planner_extension("vi")
    assert "PHẦN TRĂM DÙNG NHƯ TỶ LỆ" in vi
    assert "exp(a: str, b: str)" in vi

    en = prompts.planner_extension("en")
    assert "PERCENTAGES USED AS RATES" in en
    assert "Exponentiation" in en


def test_verbatim_prompts_are_unaffected_by_the_extension():
    """use_prompt_ext=False must never see the clarification -- opt-in only.

    The paper's OWN example (`subtract(a='giá_trị_mới', b='giá_trị_cũ')`) is
    expected to still be in planner_part1 unchanged -- that is the verbatim
    text this whole package is careful to preserve. What must NOT leak into
    the base prompt is the clarifying sentence that explains it is a
    placeholder, since that sentence is not the paper's.
    """
    assert "giá_trị_mới" in prompts.VI.planner_part1  # the paper's own example, untouched
    assert "CHỈ LÀ TÊN MINH HỌA" not in prompts.VI.planner_part1
    assert "CHỈ LÀ TÊN MINH HỌA" not in prompts.VI.planner_part2
    assert "ONLY placeholder names" not in prompts.EN.planner_part1
    assert "ONLY placeholder names" not in prompts.EN.planner_part2
