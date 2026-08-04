# ViNumQA dataset variants

All files share the FinQA-style schema:
`{pre_text, table, post_text, id, qa: {question, program, exe_ans[, reasoning_trace, trace_source]}}`.

## Core splits (do not modify)

| File | Samples | Notes |
|---|---|---|
| `train.json` | 2,993 | original ViNumQA training set, gold `program`/`exe_ans`, **no reasoning trace** |
| `valid.json` | 584 | original ViNumQA validation set, gold `program`/`exe_ans`, **no reasoning trace** |
| `test.json` | 497 | public test set, gold provided — **the fixed evaluation set for every experiment in this repo; never edited** |
| `private_test.json` | 1,625 | leaderboard held-out set, **no gold** — not used for local evaluation |

## Reasoning-trace distillation variants

Every `*_with_reasoning_trace*` file adds a `qa.reasoning_trace` field (and
`qa.trace_source`) on top of `train.json`/`valid.json`; `program`/`exe_ans`
are always the untouched original gold — the student is trained to reproduce
`program`, never the teacher's own program.

Two independent axes distinguish the variants:

- **Distillation method** — how the trace was produced.
- **Language** — Vietnamese (no suffix) or English (`_en` suffix, machine-translated
  or distilled directly in English).

### 1. Backward rationalization (Qwen3-Next-80B teacher) — superseded

The teacher is shown the **gold program and answer** and asked to write a
trace that arrives at exactly that program (CoNR prompting,
`notebooks/vinumqa/distill-reasoning-trace/qwen3-next-80b-conr-trace-gen.ipynb`).
**Known issue**: reading the full trace set end-to-end found ~1% of samples
where the teacher's reasoning acknowledged it was following a program it had
been handed, rather than deriving it independently — i.e. real leakage from
being shown the answer. Kept for comparison, not used for new experiments.

| File | Samples | Language |
|---|---|---|
| `train_with_reasoning_trace.json` | 2,905 | Vietnamese |
| `valid_with_reasoning_trace.json` | 571 | Vietnamese |
| `train_with_reasoning_trace_en.json` | 2,905 | English (MT via `envit5-translation`) |
| `valid_with_reasoning_trace_en.json` | 571 | English (MT via `envit5-translation`) |

### 2. Independent-solve, PA-match only (teacher: gemma-4-31B-it) — `pa_distiil_gemma`

The teacher sees **only context + question** (no gold shown) and derives the
program itself. A sample is kept only if the teacher's program matches gold
under the Program Accuracy check. This is the strictest filter, so it has the
fewest samples of the independent-solve variants.

Used in the results table as **`SFT (w VIE/ENG reasoning trace distill — PA match only)`**.

| File | Samples | Language |
|---|---|---|
| `train_with_reasoning_trace_pa_distiil_gemma.json` | 2,290 | Vietnamese |
| `valid_with_reasoning_trace_pa_distiil_gemma.json` | 455 | Vietnamese |
| `train_with_reasoning_trace_en_pa_distiil_gemma.json` | 2,123 | English |
| `valid_with_reasoning_trace_en_pa_distiil_gemma.json` | 450 | English |

### 3. Independent-solve, PA + partial match (teacher: gemma-4-31B-it) — `pa_partial_match`

Same independent-solve pipeline as above, with the match condition **relaxed**:
a program that differs from gold only in numeric surface form (e.g.
`add(100, 50)` vs `add(100.00, 50.00)`) is still accepted, on top of the PA
check. This recovers extra verified samples the stricter PA-only filter
rejects for cosmetic formatting reasons — every `pa_distiil_gemma` sample is
also in this set, plus the additional partial-match recoveries.

Used in the results table as **`SFT (w VIE/ENG reasoning trace distill — PA/partial match)`**.

| File | Samples | Language |
|---|---|---|
| `train_with_reasoning_trace_distill_gemma_pa_partial_match.json` | 2,364 | Vietnamese |
| `valid_with_reasoning_trace_distill_gemma_pa_partial_match.json` | 470 | Vietnamese |
| `train_with_reasoning_trace_distill_gemma_en_pa_partial_match.json` | 2,993 | English — gold-merged: every original sample kept, unverified ones carry `reasoning_trace: null` and train as bare-program (no trace) |
| `valid_with_reasoning_trace_distill_gemma_en_pa_partial_match.json` | 584 | English — same gold-merged shape |

## Generating source

The independent-solve variants (2 and 3 above) are produced by
`notebooks/vinumqa/distill-reasoning-trace/gemma-4-31b-conr-trace-gen-independent-solve.ipynb`
(Vietnamese) and `gemma-4-31b-conr-trace-gen-independent-solve-en.ipynb`
(English). See `notebooks/vinumqa/distill-reasoning-trace/README.md` for the
full distillation pipeline and prompt design.
