# Reasoning-trace distillation (CoNR)

Generates the `qa.reasoning_trace` field used by the SFT-with-reasoning
notebooks (`../sft-w-reasoning-trace-distill/`). Two pipelines exist, in the
order they were tried:

## 1. Backward rationalization — `qwen3-next-80b-conr-trace-gen*.ipynb`

**Superseded — do not use for new experiments.** Teacher: `Qwen3-Next-80B-A3B-Thinking`,
served locally via vLLM. The teacher is given the context/question **plus the
verified gold program and answer**, and asked to write a trace that arrives
at exactly that program. Reading the full output end-to-end found ~1% of
traces where the teacher's reasoning explicitly acknowledged following a
program it had been handed, rather than deriving it — genuine leakage from
being shown the answer, not just a denylist near-miss. Produces
`datasets/ViNumQA/train_with_reasoning_trace.json` /
`valid_with_reasoning_trace.json` (kept for comparison).

- `qwen3-next-80b-conr-trace-gen.ipynb` — the original pipeline.
- `qwen3-next-80b-conr-trace-gen-independent-solve-reflect.ipynb` — an
  independent-solve + self-reflection redesign of this pipeline (teacher
  drafts a program, reflects against format rules, then commits a final
  program), written after the leakage was found. Not yet run against the
  teacher model; superseded in practice by the gemma-4-31B-it pipeline below,
  which reached a verified pipeline first.

## 2. Independent-solve — `gemma-4-31b-conr-trace-gen-independent-solve*.ipynb`

**Current pipeline.** Teacher: `gemma-4-31B-it`, called via the same
OpenAI-compatible API used by the ICL baseline notebooks (cheaper/faster than
a local vLLM load, and the strongest model on the ICL leaderboard besides —
see the root README's results table). The teacher sees **only context +
question**, exactly what the student sees at inference time, and must derive
the program itself — no gold shown.

- `gemma-4-31b-conr-trace-gen-independent-solve.ipynb` — Vietnamese traces.
- `gemma-4-31b-conr-trace-gen-independent-solve-en.ipynb` — English traces.

### Prompt design (v2, convention-hardened)

A first pass with a plain independent-solve prompt reached only 14.4%
exact-match. An error analysis of all 2,993 attempts found the misses were
overwhelmingly *formatting conventions*, not reasoning errors:

| Failure mode | Share of v1 errors |
|---|---|
| Nested calls inside arguments (e.g. `divide(add(#0,#1), #2)`) | 37.6% |
| Unjustified trailing `multiply(#N, 100)` to force a percentage | 27.3% |
| Wrapping a single already-known value in `table_sum(...)` | 9.3% |
| Invented operator outside the 10 valid ones | 9.0% |
| No program emitted at all | 9.2% |

Each became an explicit rule in the v2 prompt, stated with a wrong/right
example and grounded in counts over the gold programs (e.g. nested calls
appear in only 6 of 2,993 gold programs; `table_*` uses a row label + `none`
454 times against an explicit value list twice). This alone raised exact-match
to 62.8% (train) / 63.4% (valid).

### Filtering — two match tests, kept separate

A generated trace is only trusted if its program agrees with gold under
**at least one** of two complementary checks — chosen because measurement
showed neither one dominates the other:

- **Strict match** (`is_exact_program_match`) — normalizes numeric surface
  form (`100` == `100.00`, `20%` == `0.2`) but requires identical step
  structure.
- **PA match** (`is_pa_match`, via `notebooks/evaluate/scorer.py`'s
  `equal_program`) — sympy symbolic equivalence, so commuted arguments or
  reordered steps pass, but literals are opaque strings, so it wrongly
  rejects `100` vs `100.00`.

Accepting either lifted verified coverage from 2,334 (PA-only) to 2,362
(union) on the measured run, with zero extra API calls — every recovery was
manually spot-checked as a genuine match, not a false positive. This union is
what `datasets/ViNumQA/*_distill_gemma_*_pa_partial_match.json` contains;
the stricter PA-only variant (`*_pa_distiil_gemma.json`) is a subset of it.
See `datasets/ViNumQA/README.md` for exact file-to-filter mapping and sample
counts.

### Rationalization pass (STaR-style) for residual failures

For samples still unverified after independent-solve, a second pass supplies
the teacher with the gold **answer only** (never the program) as a hint to
resolve genuine labelling ambiguities the question's wording cannot settle
(e.g. "X tăng bao nhiêu" — absolute or relative change — splits roughly
54/46 in gold with no wording cue). Traces from this pass are tagged
`qa.trace_source = "answer_conditioned"` (vs. `"independent"`), have their
own calibrated leak-phrase and answer-echo detectors, and a leaking or
still-unverified trace is rejected outright rather than kept with a caveat.

**Note**: as measured in the datasets currently in this repo
(`datasets/ViNumQA/README.md`), the shipped `pa_distiil_gemma` /
`pa_partial_match` files are 100% `trace_source = "independent"` — the
rationalization pass exists in the pipeline but its output is not yet part
of the checked-in dataset variants.

## Output format

Both pipelines produce `<think>...</think>` + `<program>...</program>`
(the gemma pipeline additionally uses `<draft_program>`/`<reflection>` in the
reflect variant). `qa.program`/`qa.exe_ans` in the final dataset are always
the original gold — only `qa.reasoning_trace` (and `qa.trace_source`, for the
gemma pipeline) are new fields.
