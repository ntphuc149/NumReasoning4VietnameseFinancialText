from stanr import scoring


def test_score_one_exact_match():
    pa, ea = scoring.score_one("divide(317, 1830)", "divide(317, 1830)", "0.17322")
    assert pa == 1.0
    assert ea == 1.0


def test_score_one_reordered_still_pa_via_commutative_add():
    pa, _ = scoring.score_one("add(9091, 6851)", "add(6851, 9091)", "15942")
    assert pa == 1.0


def test_score_one_wrong_program_scores_zero():
    pa, ea = scoring.score_one("add(1, 2)", "divide(317, 1830)", "0.17322")
    assert pa == 0.0
    assert ea == 0.0


def test_score_one_truncated_program_scores_zero_not_raises():
    # No complete call anywhere in the string -- extract_program finds
    # nothing to salvage, program_tokenization then rejects the raw
    # (unbalanced) fallback text, and score_one catches that as 0.0/0.0
    # rather than raising.
    pa, ea = scoring.score_one("divide(#0, 5", "subtract(100, 50)", "50")
    assert pa == 0.0
    assert ea == 0.0


def test_score_one_extracts_leading_complete_call_before_a_truncated_one():
    # extract_program only keeps *complete, balanced* operator calls -- a
    # truncated second step is dropped rather than corrupting the whole
    # program, so a generation cut off after finishing its real answer still
    # scores correctly against a single-step gold.
    pa, ea = scoring.score_one("subtract(100, 50), divide(#0, 5", "subtract(100, 50)", "50")
    assert pa == 1.0
    assert ea == 1.0


def test_table_sum_row_lookup():
    table = [["", "2018", "2017"], ["intrinsic value", "9", "18"]]
    _, ea = scoring.score_one(
        "table_sum(intrinsic value, none)", "table_sum(intrinsic value, none)", "27", table=table,
    )
    assert ea == 1.0
