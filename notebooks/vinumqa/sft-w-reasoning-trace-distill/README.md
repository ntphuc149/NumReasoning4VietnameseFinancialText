# QLoRA SFT — with distilled reasoning trace

Fine-tunes a student model to produce `{reasoning_trace}\n</think>\n\n{program}`
(or, for bare-program rows, an empty think block) instead of the program
alone. Compares against `../sft-wo-reasoning-trace-distill/` to measure
whether reasoning-augmented training helps.

## Models covered (parity across all three)

| Model | Train notebook(s) | Eval notebook(s) |
|---|---|---|
| Qwen3-4B | `qwen3-4b-stf-w-reasoning-trace.ipynb` | `qwen3-4b-eval-only.ipynb` |
| Qwen3-4B-Thinking-2507 | `qwen3-4b-thinking-2507-stf-w-reasoning-trace.ipynb`, `-modal.ipynb` | `qwen3-4b-thinking-2507-eval-only.ipynb`, `-modal.ipynb` |
| Gemma3-4B | `gemma3-4b-stf-w-reasoning-trace.ipynb`, `-modal.ipynb` | `gemma3-4b-eval-only.ipynb` |

Train and eval are split into separate notebooks (rather than one combined
run) because a single Kaggle session hit the 12h limit mid-eval and lost the
just-finished adapter — training now always finishes and saves the adapter
before evaluation starts in a fresh session. `-modal.ipynb` variants are the
same notebook adapted to run single-session on Modal instead of Kaggle
(train + eval in one run, since Modal has no 12h wall clock).

## `DATASET_VARIANT` switch

Every train/eval notebook here is parametrized by a single `DATASET_VARIANT`
constant near the top of the data-prep cell — change it and re-run, no other
edit needed; adapter/output/checkpoint paths auto-tag with the variant so
re-running a different one never collides with a prior run's artifacts.

| `DATASET_VARIANT` | Language | Filter | Merged with full split? |
|---|---|---|---|
| `v6` | Vietnamese | union (strict OR PA) | yes — every train sample kept, unverified rows train as bare-program |
| `v6_en` | English | union (strict OR PA) | yes — same merge, English trace |
| `pa` | Vietnamese | PA-scorer only (strictest) | no — only verified samples are in the file |
| `pa_en` | English | PA-scorer only (strictest) | no — same, English trace |

**Note on file naming**: these notebooks read `train_mixed_reasoning_v6*.json`
/ `train_with_reasoning_trace_pa*.json` from their own dataset path (built
from the same `gemma-4-31b-conr-trace-gen-independent-solve*.ipynb` pipeline
described in `../distill-reasoning-trace/README.md`). This repo's
`datasets/ViNumQA/` ships the equivalent data under different filenames —
`*_pa_distiil_gemma.json` (PA-only, ≈ the `pa`/`pa_en` variant) and
`*_distill_gemma_*_pa_partial_match.json` (union filter, ≈ the `v6`/`v6_en`
variant, gold-merged). See `datasets/ViNumQA/README.md` for the exact file
mapping and sample counts; adjust the `_VARIANT_FILES` dict in the data-prep
cell if pointing at those files instead.

## Bug fixed here

Bare-program (no-trace) rows were previously trained on the **literal string
`"nan"`** as their `<think>` content (a `pandas`/`float`-to-`str` artifact on
missing values), instead of an empty think block. `build_conversation` now
checks `isinstance(trace, str) and trace.strip()` before treating a value as
a real trace.

## `MAX_SEQ_LENGTH`

Measured empirically per model rather than copied across models — training
measures the fully-formatted sample (system + context + question + trace +
program), but inference only has the prompt and must fit
`prompt_tokens + max_new_tokens` in the same budget. Qwen3-4B-Thinking-2507
uses `MAX_SEQ_LENGTH=8192` (longer than plain Qwen3-4B's 5,678) because it is
a "thinking-length-extended" variant, not because its training traces are
longer — always re-measure rather than assume this transfers to a new model.
