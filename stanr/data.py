"""Dataset I/O, context formatting, and sample-set operations shared by
`ReasoningDistiller` and `STaNRTrainer`.

Every sample in this repo's ViNumQA/FinQA-style JSON files has the same
shape::

    {"pre_text": [...], "table": [[...], ...], "post_text": [...],
     "id": "...", "qa": {"question": ..., "program": ..., "exe_ans": ...,
     # added by distillation/self-teaching:
     "reasoning_trace": ..., "trace_source": "independent" | "self_distilled"}}

Pure Python + `tabulate` only -- no ML framework import here, so this module
(and `ReasoningDistiller`, which only needs this + `openai`) stays usable on
a machine with no GPU/Unsloth installed.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable, Sequence

from tabulate import tabulate


def load_json(path) -> list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def save_json(obj, path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def format_pre_text(sample: dict) -> str:
    return "\n".join(sample["pre_text"])


def format_table(sample: dict) -> str:
    return tabulate(sample["table"][1:], headers=sample["table"][0], tablefmt="github")


def format_post_text(sample: dict) -> str:
    return "\n".join(sample["post_text"])


def format_context(sample: dict) -> tuple[str, str, str]:
    """(pre_text, table, post_text), formatted the same way every notebook in
    this repo does, so prompts built from this stay comparable across
    experiments."""
    return format_pre_text(sample), format_table(sample), format_post_text(sample)


def build_user_message(frame: str, sample: dict) -> str:
    """Fill a `{pre_text}/{table}/{post_text}/{question}`-shaped template --
    both `CONR_USER_FRAME` and `SFT_USER_FRAME` in `stanr.prompts` take
    exactly these four fields."""
    pre_text, table, post_text = format_context(sample)
    return frame.format(
        pre_text=pre_text, table=table, post_text=post_text,
        question=sample["qa"]["question"],
    )


def ids_of(dataset: Iterable[dict]) -> set:
    return {s["id"] for s in dataset}


def samples_missing_from(full_dataset: Sequence[dict], subset_dataset: Sequence[dict]) -> list:
    """Samples in `full_dataset` whose id does not appear in `subset_dataset`.

    This is the "still unresolved" set STaNR's self-teaching step attempts
    each round: `train.json` minus the current distilled/passed training set,
    matched by id -- exactly what the self-distillation notebooks compute
    before each round of generation.
    """
    have = ids_of(subset_dataset)
    return [s for s in full_dataset if s["id"] not in have]


def with_reasoning_trace(sample: dict, reasoning_trace: str, trace_source: str) -> dict:
    """A copy of `sample` with `qa.reasoning_trace`/`qa.trace_source` set.

    `qa.program`/`qa.exe_ans` are left untouched -- always the gold values,
    never the model's own derivation. A generated program only ever serves as
    evidence that its trace can be trusted, then is discarded, matching every
    distillation/self-teaching notebook in this repo.
    """
    out = dict(sample)
    out["qa"] = dict(sample["qa"])
    out["qa"]["reasoning_trace"] = reasoning_trace
    out["qa"]["trace_source"] = trace_source
    return out


_OPEN_THINK_RE = re.compile(r"^\s*<think>\s*")


def extract_think_and_program(raw_text: str, think_end_marker: str = "</think>") -> tuple[str, str]:
    """Split raw model output into `(reasoning_trace, program_text)`.

    Searches from the END (`rfind`), not the start: a completion can contain
    more than one `</think>` if the model re-opens a second think block
    mid-generation, and taking the first occurrence then leaves prose mixed
    into what should be the program half -- this exact bug was found and
    fixed in this repo's GRPO reward function (see
    `notebooks/vinumqa/sft-grpo/qwen3-4b-grpo-continue-modal-a100-80gb.ipynb`).

    If no closing marker is found at all (generation cut off before closing
    it), the reasoning trace is empty and the entire text is treated as the
    program -- matching the `idx = 0` fallback used by the dual-GPU
    self-distillation notebook, not a guess at where the split "should" be.
    """
    close_idx = raw_text.rfind(think_end_marker)
    if close_idx == -1:
        return "", raw_text.strip()
    trace = _OPEN_THINK_RE.sub("", raw_text[:close_idx]).strip()
    program = raw_text[close_idx + len(think_end_marker):].strip()
    return trace, program
