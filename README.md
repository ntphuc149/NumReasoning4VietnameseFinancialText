# NumReasoning4VietnameseFinancialText

Numerical reasoning question answering over Vietnamese financial documents,
built for the **VLSP 2025 Numerical Reasoning QA (NumQA)** shared task.

Given a financial document (text passages + a table) and a natural-language
question, the goal is to generate an executable **computation program** (a
sequence of arithmetic/table operators) that derives the correct numerical
answer — not just the answer itself. This makes every prediction auditable:
a wrong-but-lucky answer with a broken program is worth less than a
transparent, verifiable reasoning path.

## Task definition

![Task formulation](img/task-formulation.png)

- **Input**: `pre_text` (paragraphs before a table), `table`, `post_text`
  (paragraphs after), and a Vietnamese `question`.
- **Output**: a `program` string using 10 operators —
  `add`, `subtract`, `multiply`, `divide`, `exp`, `greater`,
  `table_sum`, `table_average`, `table_max`, `table_min` — where later steps
  can reference earlier results via `#0`, `#1`, etc.
  (e.g. `subtract(9829, 642), divide(#0, 642)`).
- **Metrics** (following the FinQA/VLSP 2025 protocol):
  - **Execution Accuracy (EA)** — does executing the program produce the
    gold numeric answer?
  - **Program Accuracy (PA)** — does the program structurally match the gold
    program (same operators/args/order, after normalization)? This is the
    stricter, primary metric for the task, since a correct-by-coincidence
    program is not actually trustworthy reasoning.

See `papers/` for the shared task description, the original FinQA paper this
task's format is based on, and a program-centric policy optimization paper
used as a reference for the SFT+GRPO stage.

## Dataset — ViNumQA

`datasets/ViNumQA/` holds the core splits plus several reasoning-trace
distillation variants used across the SFT experiments below — **see
[`datasets/ViNumQA/README.md`](datasets/ViNumQA/README.md) for the full
file-by-file breakdown** (sample counts, distillation method, language,
filter). Quick orientation:

| File                                     | Samples   | Notes                                                                                              |
| ---------------------------------------- | --------- | -------------------------------------------------------------------------------------------------- |
| `train.json`                             | 2,993     | gold `program` + `exe_ans`, no reasoning trace                                                     |
| `valid.json`                             | 584       | gold `program` + `exe_ans`, no reasoning trace                                                     |
| `test.json`                              | 497       | public test set, gold provided — **fixed evaluation set for every result below**                   |
| `private_test.json`                      | 1,625     | **no gold** — held out for leaderboard scoring                                                     |
| `*_with_reasoning_trace*.json` (8 files) | 450–2,993 | distilled `qa.reasoning_trace` variants — see the dataset README for which one backs which SFT row |

Each entry: `{pre_text, table, post_text, id, qa: {question, program, exe_ans}}`.
Unlike the original English FinQA, this dataset does **not** include a
`program_re` field (alternative valid programs for the same question), so no
program-diversity augmentation from that source is available here.

## Repository layout

```
notebooks/vinumqa/
├── 0-shot/                        0-shot prompting baselines
├── 1-shot/                        1-shot prompting baselines
├── few-shot/                      3-shot prompting baselines
├── distill-reasoning-trace/       teacher → reasoning-trace distillation (CoNR)
├── translate-reasoning-trace/     (abandoned) MT-based trace translation
├── sft-wo-reasoning-trace-distill/  QLoRA SFT, program-only labels (baseline)
├── sft-w-reasoning-trace-distill/   QLoRA SFT, distilled-reasoning-trace labels
└── sft-grpo/                      (planned) SFT + GRPO policy optimization stage
notebooks/evaluate/                shared PA/EA scorer (scorer.py)
notebooks/translated-finqa/        (placeholder) planned baselines on the translated-FinQA subset
datasets/ViNumQA/                  dataset splits (see table above)
datasets/FinQA/                    original English FinQA (reference only)
papers/                            reference papers (task description, FinQA, PCPO)
```

**Every subdirectory above has its own `README.md`** with the per-notebook
breakdown — `notebooks/README.md` is the entry point into those; start there
for anything beyond a quick orientation. `datasets/README.md` /
`datasets/ViNumQA/README.md` cover the dataset variants in the same way.

Every prompting/SFT notebook shares the same `SYSTEM_MESSAGE` (operator list +
formatting rules), the same `pre_text`/`table`/`post_text`/`question` context
formatting (table rendered as GitHub-flavored markdown via `tabulate`), and
the same program parser/PA/EA scorer (`notebooks/evaluate/scorer.py`), so
results are directly comparable across models and methods.

## Methodology

### 1. Prompting baselines (0-shot / 1-shot / few-shot)

Zero, one, and three in-context examples, evaluated across multiple models
(local Qwen3-4B, API models via an OpenAI-compatible endpoint, and
`gpt-5-nano` via the OpenAI Batch API — 50% cheaper than sync calls, used
since these are offline full-test-set scoring runs rather than live serving).
The 1-shot exemplar is drawn from the VLSP 2025 paper's own Figure 1 example;
the 3 few-shot exemplars are hand-picked from `train.json`, one per evidence
category (Table Only / Text Only / Table & Text) matching the paper's own
question-type breakdown.

### 2. QLoRA SFT — no reasoning trace

`sft-wo-reasoning-trace-distill/qwen3-4b-stf-wo-reasoning-trace.ipynb`: fine-tunes
Qwen3-4B (4-bit + LoRA via Unsloth) to map context+question directly to the
gold program string, with loss masked to the assistant turn only
(`train_on_responses_only`). No synthesized or distilled reasoning trace —
this is the plain-label baseline to compare reasoning-augmented training
against.

### 3. Reasoning-trace distillation (CoNR) + SFT with reasoning

Rather than hoping the student model discovers good reasoning on its own, we
distill it from a stronger teacher. Two pipelines were tried, in this order —
full detail (prompt design, filters, sample counts) is in
[`notebooks/vinumqa/distill-reasoning-trace/README.md`](notebooks/vinumqa/distill-reasoning-trace/README.md)
and [`datasets/ViNumQA/README.md`](datasets/ViNumQA/README.md):

**a. Backward rationalization (superseded).** Teacher `Qwen3-Next-80B-A3B-Thinking`
is given the context/question **plus the verified gold program and answer**,
and asked to write a trace that arrives at exactly that program. Validated at
98.1%/99.0% exact-match (train/valid) via denylist-checked leakage screening —
but reading the full trace set end-to-end later found ~1% of samples where
the teacher's reasoning admitted it was following a program it had been
handed, i.e. real leakage despite the denylist. Superseded for new experiments;
produced `train_with_reasoning_trace.json` / `valid_with_reasoning_trace.json`
(kept for comparison).

**b. Independent-solve (current).** Teacher `gemma-4-31B-it` (the strongest
model on the ICL leaderboard below) sees **only context + question** — no
gold shown — and must derive the program itself. A convention-hardened v2
prompt (explicit rules against nested calls, unjustified `*100` rescaling,
wrapping a single value in `table_sum`, invented operators — each grounded in
counts over the gold programs) raised exact-match from 14.4% to 62.8%/63.4%
(train/valid). Samples are kept if the teacher's program matches gold under
either a strict surface-normalized check or sympy-based PA equivalence
(`notebooks/evaluate/scorer.py`) — two complementary, non-redundant tests.
Two filter strictness levels are shipped:

- **PA match only** — the stricter filter, fewest samples.
- **PA + partial match** — also accepts numeric-surface-form differences
  (`100` vs `100.00`) that PA alone rejects; a superset of PA-match-only.

Both are available in Vietnamese and English (distilled directly in English,
not machine-translated), giving 4 dataset variants — see the dataset README
for exact files/counts.

**SFT with reasoning** (`sft-w-reasoning-trace-distill/`, notebooks for
Qwen3-4B / Qwen3-4B-Thinking-2507 / Gemma3-4B): the assistant turn is
`{reasoning_trace}\n</think>\n\n{program}` (empty think block for bare-program
rows in the gold-merged variants). See that folder's README for the
`DATASET_VARIANT` switch used to pick which of the 4 variants to train on.

### 4. SFT + GRPO (planned)

`notebooks/vinumqa/sft-grpo/` — reward function combining program validity,
execution correctness, and conciseness (following the program-centric policy
optimization approach in `papers/`), to further refine the SFT model beyond
supervised imitation. Not yet implemented.

## Results (test.json, 497 samples)

### In-context learning results

![In-context learning performance](img/in-context-learning.png)

Best in-context result so far: **gemma-4-31B-it, few-shot (3)** — PA 0.5674 / EA 0.6137.

### Fine-tuned

<img src="img/sft.png" alt="With supervised fine-tuning" width="480">

PA/EA are computed with the shared evaluator in `notebooks/evaluate/scorer.py`
(sympy-based symbolic Program Accuracy, table-row-lookup-aware Execution
Accuracy, ported from the official FinQA evaluation protocol), so all rows
are directly comparable.

## Running the notebooks

Most notebooks are self-contained: they resolve the project root by walking
up to the nearest `.git` directory, so they can be run from either the repo
root or the notebook's own folder without path edits.

- **API-based prompting notebooks** (0/1/few-shot, non-batch): need `API_KEY`
  and `BASE_URL` in a `.env` file at the project root (OpenAI-compatible
  endpoint).
- **Batch API notebooks** (`*-gpt5nano-batch.ipynb`): need `OPENAI_API_KEY`
  (a real OpenAI account key — Batch API is not available on third-party
  proxy endpoints).
- **Local-model notebooks** (Qwen3-4B, Qwen3-4B-Thinking-2507, Gemma3-4B,
  Qwen3-Next-80B): designed to run on Kaggle (free T4/P100 tier, 12h session
  limit — train/eval are split into separate notebooks where a combined run
  would exceed it) or Modal (`-modal.ipynb` variants; Qwen3-Next-80B needs
  ≥160GB VRAM — see the trace-gen notebook's intro cell for the specific GPU
  config and vLLM flags needed on Blackwell-generation GPUs).

`.env` and all `outputs/`/intermediate-artifact directories are gitignored;
see `.gitignore` for the exact excluded paths.

## Acknowledgments

Built as part of the VLSP 2025 NumQA shared task submission. See `papers/`
for the shared task paper (dataset construction, evaluation protocol, and a
survey of top-performing teams' methods) and the FinQA/PCPO papers this
work's format and SFT+GRPO design are based on.
