from stanr import data
from stanr.prompts import SFT_USER_FRAME

SAMPLE = {
    "pre_text": ["some text before"],
    "table": [["", "2018", "2017"], ["revenue", "100", "90"]],
    "post_text": ["some text after"],
    "id": "abc-1",
    "qa": {"question": "what is revenue in 2018?", "program": "table_sum(revenue, none)", "exe_ans": "100"},
}


def test_format_pre_text():
    assert data.format_pre_text(SAMPLE) == "some text before"


def test_format_table_uses_first_row_as_header():
    table = data.format_table(SAMPLE)
    assert "revenue" in table
    assert "2018" in table


def test_build_user_message_fills_all_fields():
    msg = data.build_user_message(SFT_USER_FRAME, SAMPLE)
    assert "what is revenue in 2018?" in msg
    assert "some text before" in msg


def test_ids_of():
    assert data.ids_of([SAMPLE]) == {"abc-1"}


def test_samples_missing_from():
    full = [SAMPLE, {**SAMPLE, "id": "abc-2"}]
    subset = [SAMPLE]
    missing = data.samples_missing_from(full, subset)
    assert [s["id"] for s in missing] == ["abc-2"]


def test_with_reasoning_trace_does_not_mutate_original():
    out = data.with_reasoning_trace(SAMPLE, "because X", "independent")
    assert out["qa"]["reasoning_trace"] == "because X"
    assert out["qa"]["trace_source"] == "independent"
    assert "reasoning_trace" not in SAMPLE["qa"]  # original untouched
    assert out["qa"]["program"] == SAMPLE["qa"]["program"]  # gold program preserved


def test_extract_think_and_program_finds_last_marker():
    raw = "<think>reasoning here</think>divide(1, 2)"
    trace, program = data.extract_think_and_program(raw)
    assert trace == "reasoning here"
    assert program == "divide(1, 2)"


def test_extract_think_and_program_uses_last_marker_if_reopened():
    raw = "<think>first</think>still thinking</think>divide(1, 2)"
    trace, program = data.extract_think_and_program(raw)
    assert trace == "first</think>still thinking"
    assert program == "divide(1, 2)"


def test_extract_think_and_program_no_marker_falls_back_to_whole_text_as_program():
    raw = "divide(1, 2)"
    trace, program = data.extract_think_and_program(raw)
    assert trace == ""
    assert program == "divide(1, 2)"
