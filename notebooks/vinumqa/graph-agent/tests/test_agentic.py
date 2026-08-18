"""Tests for the pure core of the MPR-Agent pipeline. No API calls.

Everything the agent gets wrong silently lives between the planner's plan DSL
and the program string `scorer.py` grades, so that boundary is what is tested
hardest here. The round-trip property test at the bottom is the important one:
it exercises the transpiler against every gold program in the real splits.

Run from the repo root with:
    .venv/Scripts/python -m pytest notebooks/vinumqa/graph-agent/tests -q
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# The package lives one level up, beside the driver notebook.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agentic.agents import extract_subqueries  # noqa: E402
from agentic.program import (  # noqa: E402
    Action,
    PlanParseError,
    TranspileError,
    canonicalize,
    clean_literal,
    parallelism,
    parse_plan,
    plan_text_to_program,
    program_to_plan_text,
    transpile,
    vote,
)
from agentic.scoring import (  # noqa: E402
    equal_program,
    execute_program,
    find_project_root,
    program_tokenization,
)


# ============================================================ plan DSL parsing


def test_parses_the_papers_own_example():
    plan = parse_plan("1. subtract(a='600', b='500') 2. divide(a='$1', b='500') "
                      "3. join() <END_OF_PLAN>")
    assert [a.tool for a in plan.actions] == ["subtract", "divide", "join"]
    assert plan.by_id[2].refs() == (1,)
    assert plan.answer_action().id == 2


@pytest.mark.parametrize(
    "raw",
    [
        "1. subtract(a='600', b='500')\n2. join()\n<END_OF_PLAN>",       # newlines
        "1) subtract(a='600', b='500') 2) join() <END_OF_PLAN>",         # `1)` form
        "```\n1. subtract(a='600', b='500')\n2. join()\n```",            # code fence
        "Kế hoạch: 1. subtract(a='600', b='500') 2. join()",             # prose prefix
        "1. subtract('600', '500') 2. join()",                           # positional
        "1. subtract(a=‘600’, b=‘500’) 2. join()",   # curly quotes
        "1. subtract(a='600', b='500', context=['x']) 2. join()",        # B.7's 3rd arg
        "subtract(a='600', b='500')\njoin()",                            # unnumbered
    ],
)
def test_surface_variations_all_reach_the_same_program(raw):
    assert plan_text_to_program(raw).program == "subtract(600, 500)"


def test_table_arg_aliases_are_accepted():
    # B.7 says `row_identifier`; the paper's own Figure 1 writes `column`.
    for arg in ("row_identifier", "row", "row_name", "column", "name"):
        raw = f"1. table_max({arg}='Lãi ròng') 2. join()"
        assert plan_text_to_program(raw).program == "table_max(Lãi ròng, none)"


def test_bracketed_row_label_survives_parsing_and_emission():
    # 35 of the 497 test programs name a row whose label contains brackets.
    raw = "1. table_min(row_identifier='ROE (%)') 2. join()"
    program = plan_text_to_program(raw).program
    assert program == "table_min(ROE (%), none)"
    # And the grader's own tokeniser must still read it as one argument.
    assert program_tokenization(program) == [
        "table_min(", "ROE (%)", "none", ")", "EOF"
    ]


def test_dangling_reference_is_rejected():
    with pytest.raises(PlanParseError):
        parse_plan("1. divide(a='$9', b='2') 2. join()")


def test_cycle_is_rejected():
    with pytest.raises(PlanParseError):
        parse_plan("1. add(a='$2', b='1') 2. add(a='$1', b='1') 3. join()")


def test_empty_and_junk_plans_are_rejected():
    for raw in ("", "   ", "I cannot answer this question."):
        with pytest.raises(PlanParseError):
            parse_plan(raw)


def test_plan_with_only_join_is_rejected():
    with pytest.raises(PlanParseError):
        parse_plan("1. join() <END_OF_PLAN>")


def test_truncated_generation_keeps_the_complete_prefix():
    # max_tokens cut the plan mid-call; the complete calls before it still parse.
    plan = parse_plan("1. subtract(a='600', b='500') 2. divide(a='$1', b='5")
    assert [a.tool for a in plan.actions] == ["subtract"]


# ================================================================= plan as DAG


def test_independent_actions_share_a_topological_level():
    plan = parse_plan(
        "1. subtract(a='10', b='5') "
        "2. subtract(a='20', b='7') "
        "3. add(a='$1', b='$2') "
        "4. join()"
    )
    levels = plan.levels()
    assert [len(level) for level in levels] == [2, 1]
    assert {a.id for a in levels[0]} == {1, 2}
    assert parallelism(plan) == pytest.approx(1.5)


def test_sequential_plan_has_one_action_per_level():
    plan = parse_plan("1. subtract(a='600', b='500') 2. divide(a='$1', b='500') "
                      "3. join()")
    assert [len(level) for level in plan.levels()] == [1, 1]


# ================================================================== transpiler


def test_paper_appendix_b11_examples():
    """The four worked examples printed in Figure B.11 / B.12."""
    cases = [
        ("1. subtract(a='21', b='47') 2. join() <END_OF_PLAN>",
         "subtract(21, 47)"),
        ("1. subtract(a='600', b='500') 2. divide(a='$1', b='500') 3. join() "
         "<END_OF_PLAN>",
         "subtract(600, 500), divide(#0, 500)"),
        ("1. subtract(a='43.81', b='100.00') 2. divide(a='$1', b='100.00') "
         "3. join() <END_OF_PLAN>",
         "subtract(43.81, 100.00), divide(#0, 100.00)"),
        ("1. table_sum(row_identifier='Lợi nhuận sau thuế (tỷ đồng)') 2. join() "
         "<END_OF_PLAN>",
         "table_sum(Lợi nhuận sau thuế (tỷ đồng), none)"),
    ]
    for raw, expected in cases:
        assert plan_text_to_program(raw).program == expected


def test_reference_renumbering_is_positional_not_id_arithmetic():
    """`$k -> #(k-1)` is wrong in general; the mapping comes from emitted order."""
    raw = "5. subtract(a='600', b='500') 9. divide(a='$5', b='500') 12. join()"
    assert plan_text_to_program(raw).program == "subtract(600, 500), divide(#0, 500)"


def test_dead_branch_is_pruned():
    """A branch the answer does not use is dropped.

    It cannot change EA (the last step is the result) but it can cost PA:
    equal_program walks every step and rejects a literal absent from gold.
    """
    raw = (
        "1. multiply(a='7', b='9') "        # unused
        "2. subtract(a='600', b='500') "
        "3. divide(a='$2', b='500') "
        "4. join()"
    )
    result = plan_text_to_program(raw)
    assert result.program == "subtract(600, 500), divide(#0, 500)"
    assert any("pruned" in w for w in result.warnings)


def test_join_argument_selects_the_answer_action():
    raw = (
        "1. subtract(a='600', b='500') "
        "2. multiply(a='2', b='3') "
        "3. join($1)"
    )
    assert plan_text_to_program(raw).program == "subtract(600, 500)"


def test_reference_to_a_table_result_is_allowed():
    """B.9/B.10 forbids it, but ViNumQA gold does it and scorer.py executes it."""
    raw = "1. table_sum(row_identifier='Doanh thu') 2. divide(a='$1', b='4') 3. join()"
    assert plan_text_to_program(raw).program == (
        "table_sum(Doanh thu, none), divide(#0, 4)"
    )


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("'1,041'", "1041"),      # thousands separator would split the step
        ("'$47'", "47"),          # B.10 says strip the currency symbol
        ("' 3.14 '", "3.14"),
        ("'-54'", "-54"),
        ("'48.8%'", "48.8%"),     # kept verbatim: str_to_num reads it as 0.488
        ("'1\xa0234'", "1234"),   # non-breaking space as a thousands separator
    ],
)
def test_literal_cleaning(raw, expected):
    assert clean_literal(raw) == expected


def test_literal_100_is_not_special_cased():
    """Gold legitimately contains 100 -- e.g. a base-100 index series."""
    raw = "1. subtract(a='96.67', b='100') 2. divide(a='$1', b='100') 3. join()"
    assert plan_text_to_program(raw).program == (
        "subtract(96.67, 100), divide(#0, 100)"
    )


def test_non_numeric_argument_is_rejected():
    with pytest.raises(TranspileError):
        plan_text_to_program("1. add(a='doanh thu', b='5') 2. join()")


def test_row_label_with_comma_is_rejected():
    """A depth-0 comma in a label would split the step into three arguments."""
    with pytest.raises(TranspileError):
        plan_text_to_program("1. table_sum(row_identifier='Doanh thu, chi phí') "
                             "2. join()")


def test_emitted_program_executes_against_a_real_table():
    table = [
        ["chỉ tiêu", "2022", "2023"],
        ["ROE (%)", "12.5", "18.3"],
    ]
    program = plan_text_to_program(
        "1. table_max(row_identifier='ROE (%)') 2. join()"
    ).program
    ok, value = execute_program(program, table)
    assert ok and value == 18.3


# ====================================================================== voting


def test_commutative_arguments_are_sorted_before_grouping():
    assert canonicalize("add(1, 2)") == canonicalize("add(2, 1)")
    assert canonicalize("multiply(3, 4)") == canonicalize("multiply(4, 3)")


def test_non_commutative_arguments_are_not_sorted():
    assert canonicalize("subtract(1, 2)") != canonicalize("subtract(2, 1)")
    assert canonicalize("divide(1, 2)") != canonicalize("divide(2, 1)")


def test_numeric_formatting_does_not_split_a_cluster():
    assert canonicalize("subtract(600, 500)") == canonicalize("subtract(600.0, 500.00)")
    # str_to_num reads a percent literal as its decimal, so these agree too.
    assert canonicalize("divide(1, 48.8%)") == canonicalize("divide(1, 0.488)")


def test_majority_wins():
    result = vote(["add(1, 2)", "add(2, 1)", "subtract(1, 2)"])
    assert canonicalize(result.winner) == canonicalize("add(1, 2)")
    assert result.n_clusters == 2
    assert result.consensus == pytest.approx(2 / 3)


def test_tie_is_broken_by_fewer_steps():
    """Paper section 4.4: 'the program with fewer steps ... is chosen'."""
    short = "divide(1041, 0.292)"
    long = "divide(29.2, 100), divide(1041, #0)"
    assert vote([long, short]).winner == short


def test_shortest_surface_form_represents_a_cluster():
    result = vote(["add(2, 1)", "add(1, 2)", "add(1, 2)"])
    assert result.n_clusters == 1
    assert result.consensus == 1.0


def test_symbolic_mode_merges_algebraic_rearrangements():
    """What `equal_program` clustering actually buys -- no more, no less.

    It merges rearrangements over the *same literals*. It does not merge
    programs that introduce a new literal, because scorer.py's PA admits only
    literals present in the program compared against.
    """
    flat = [
        "add(1, 2), multiply(#0, 3)",
        "multiply(1, 3), multiply(2, 3), add(#0, #1)",
    ]
    assert vote(flat, mode="symbolic").n_clusters == 1
    assert vote(flat, mode="canonical").n_clusters == 2
    # The Example 5.1 pair introduces 29.2 and 100, so neither mode merges it.
    example_51 = ["divide(1041, 0.292)", "divide(29.2, 100), divide(1041, #0)"]
    assert vote(example_51, mode="symbolic").n_clusters == 2


def test_figure_1_candidates_are_not_merged_by_structure_voting():
    """Documents the known gap between the paper's prose and its Figure 1.

    Figure 1 draws three candidates, labels the stage "Top result voting", and
    shows 34770.78 winning 2-vs-1. Two of those candidates are structurally
    different programs with the same executed value, so structure-level voting
    -- which is what section 4.4's prose describes and what is implemented here
    -- sees 1-1-1 instead, and the depicted majority never forms.

    This test asserts the current (prose-faithful) behaviour so that adding a
    result-level mode later shows up as a deliberate change, not a regression.
    """
    c1 = "add(19038.80, 9445.09), add(#0, 6286.89)"
    c2 = "add(19038.80, 6286.89), add(#0, 9445.09)"
    c3 = "add(19038.80, 31434.47), add(#0, 6286.89)"

    assert execute_program(c1, None)[1] == execute_program(c2, None)[1] == 34770.78
    assert execute_program(c3, None)[1] == 56760.16
    assert vote([c1, c2, c3], mode="canonical").n_clusters == 3


def test_empty_candidate_list_yields_no_winner():
    result = vote([])
    assert result.winner is None and result.n_clusters == 0


# ================================================ subquery JSON recovery (G_sq)


@pytest.mark.parametrize(
    "raw",
    [
        '{"subqueries": ["Doanh thu 2023?", "Doanh thu 2022?"]}',
        '```json\n{"subqueries": ["Doanh thu 2023?", "Doanh thu 2022?"]}\n```',
        'Đây là kết quả:\n{"subqueries": ["Doanh thu 2023?", "Doanh thu 2022?"]}\nHết.',
        '["Doanh thu 2023?", "Doanh thu 2022?"]',
    ],
)
def test_subqueries_are_recovered_from_wrapped_json(raw):
    assert extract_subqueries(raw) == ["Doanh thu 2023?", "Doanh thu 2022?"]


def test_unrecoverable_subquery_output_returns_empty():
    assert extract_subqueries("Tôi không thể trả lời.") == []
    assert extract_subqueries("") == []


# ============================================ round-trip over every gold program


def _gold_programs(split: str) -> list[str]:
    path = find_project_root() / "datasets" / "ViNumQA" / "origin" / f"{split}.json"
    with open(path, encoding="utf-8") as handle:
        return [s["qa"]["program"] for s in json.load(handle)]


@pytest.mark.parametrize("split", ["train", "valid", "test"])
def test_round_trip_over_all_gold_programs(split):
    """`transpile(parse_plan(to_plan(g)))` must be equivalent to `g`, for all g.

    Thousands of real cases -- step renumbering, `table_*(row, none)` arity,
    bracketed row labels, multi-step chains -- for the cost of no API calls.
    Golds that scorer.py itself cannot tokenise are pre-existing data noise
    (5 rows in train) and are skipped rather than counted as failures.
    """
    failures = []
    skipped = 0
    for gold in _gold_programs(split):
        try:
            gold_tokens = program_tokenization(gold)
        except ValueError:
            skipped += 1
            continue
        try:
            program = plan_text_to_program(program_to_plan_text(gold)).program
            if not equal_program(gold_tokens, program_tokenization(program)):
                failures.append((gold, program))
        except Exception as exc:  # noqa: BLE001
            failures.append((gold, f"{type(exc).__name__}: {exc}"))

    assert not failures, (
        f"{len(failures)} round-trip failure(s) in {split} "
        f"(skipped {skipped} unparseable gold): {failures[:5]}"
    )


# =========================================================== graph construction


def test_pipeline_graph_orders_nodes_by_dependency():
    from agentic.agents import AgentGraph, GraphError, Node

    class Stub(Node):
        def __init__(self, name):
            self.name = name

        def execute(self, state):
            return state

    graph = AgentGraph()
    graph.add(Stub("a"))
    graph.add(Stub("b"), depends_on=("a",))
    graph.add(Stub("c"), depends_on=("b",))
    assert [n.name for n in graph.order()] == ["a", "b", "c"]

    with pytest.raises(GraphError):
        graph.add(Stub("d"), depends_on=("missing",))


def test_node_failure_is_traced_not_raised():
    from agentic.agents import AgentState, Node

    class Exploding(Node):
        name = "boom"

        def execute(self, state):
            raise RuntimeError("kaboom")

    state = Exploding().run(AgentState(sample_id="1", question="q", context="c"))
    assert state.traces[-1].ok is False
    assert "kaboom" in state.errors[0]


# ================================================================ plan renderer


def test_action_str_round_trips_through_the_parser():
    action = Action(id=1, tool="subtract", args=("600", "500"))
    assert str(action) == "1. subtract(a='600', b='500')"
    assert transpile(parse_plan(str(action) + " 2. join()")).program == (
        "subtract(600, 500)"
    )
