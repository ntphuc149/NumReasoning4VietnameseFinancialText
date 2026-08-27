import json

import pytest

from stanr.trainer import STaNRTrainer, _strip_chat_end_markers

TRAIN_SAMPLE = {
    "pre_text": ["t"], "table": [["", "2018"], ["revenue", "100"]], "post_text": ["t"],
    "id": "s1", "qa": {"question": "q", "program": "table_sum(revenue, none)", "exe_ans": "100"},
}
DISTILLED_SAMPLE = {
    **TRAIN_SAMPLE,
    "qa": {**TRAIN_SAMPLE["qa"], "reasoning_trace": "trace", "trace_source": "independent"},
}


def _write(tmp_path, name, obj):
    path = tmp_path / name
    path.write_text(json.dumps(obj), encoding="utf-8")
    return str(path)


def _make_trainer(tmp_path, **overrides):
    kwargs = dict(
        input_train_raw_dataset=_write(tmp_path, "train.json", [TRAIN_SAMPLE]),
        input_distilled_train_set=_write(tmp_path, "train_d.json", [DISTILLED_SAMPLE]),
        input_val_raw_dataset=_write(tmp_path, "val.json", [TRAIN_SAMPLE]),
        input_distilled_val_set=_write(tmp_path, "val_d.json", [DISTILLED_SAMPLE]),
        input_test_raw_dataset=_write(tmp_path, "test.json", [TRAIN_SAMPLE]),
        pretrained_model_name="unsloth/Qwen3-4B",
        model_checkpoint_path=str(tmp_path / "out"),
    )
    kwargs.update(overrides)
    return STaNRTrainer(**kwargs)


def test_invalid_precision_raises(tmp_path):
    with pytest.raises(ValueError):
        _make_trainer(tmp_path, precision="8-bit")


def test_negative_loop_raises(tmp_path):
    with pytest.raises(ValueError):
        _make_trainer(tmp_path, loop=-1)


def test_valid_precisions_accepted(tmp_path):
    for precision in ("4-bit", "16-bit", "full"):
        trainer = _make_trainer(tmp_path, precision=precision)
        assert trainer.precision == precision


def test_loop_zero_is_allowed_plain_sft_only(tmp_path):
    trainer = _make_trainer(tmp_path, loop=0)
    assert trainer.loop == 0


def test_strip_chat_end_markers():
    assert _strip_chat_end_markers("divide(1, 2)<|im_end|>") == "divide(1, 2)"
    assert _strip_chat_end_markers("divide(1, 2)<end_of_turn>") == "divide(1, 2)"
    assert _strip_chat_end_markers("divide(1, 2)") == "divide(1, 2)"


def test_filter_and_build_keeps_only_pa_matches(tmp_path):
    trainer = _make_trainer(tmp_path)
    unresolved = [
        {**TRAIN_SAMPLE, "id": "u1", "qa": {**TRAIN_SAMPLE["qa"], "program": "add(1, 2)"}},
        {**TRAIN_SAMPLE, "id": "u2", "qa": {**TRAIN_SAMPLE["qa"], "program": "add(1, 2)"}},
    ]
    raw_outputs = [
        "<think>right</think>add(1, 2)",   # matches -- kept
        "<think>wrong</think>add(9, 9)",   # doesn't match -- dropped
    ]
    passed = trainer._filter_and_build(unresolved, raw_outputs)
    assert len(passed) == 1
    assert passed[0]["id"] == "u1"
    assert passed[0]["qa"]["reasoning_trace"] == "right"
    assert passed[0]["qa"]["trace_source"] == "self_distilled"


def test_filter_and_build_accepts_commutative_reorder(tmp_path):
    """PA (equal_program), not strict order -- matches STaNR's own filter."""
    trainer = _make_trainer(tmp_path)
    unresolved = [{**TRAIN_SAMPLE, "id": "u1", "qa": {**TRAIN_SAMPLE["qa"], "program": "add(1, 2)"}}]
    raw_outputs = ["<think>reordered</think>add(2, 1)"]
    passed = trainer._filter_and_build(unresolved, raw_outputs)
    assert len(passed) == 1
