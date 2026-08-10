# STaNR self-distillation — ViNumQA-only training

**These files contain only the base teacher-verified set (`../distill-from-gemma-teacher/pa-match-only/eng/`)
plus newly self-distilled samples** — not a separate standalone dataset.
Each file is the full training/validation set for its STaNR row in the root
README's Table 2 (`SFT (wo reasoning/distill)` trained purely on ViNumQA).

Pipeline: fine-tune a model on `../distill-from-gemma-teacher/pa-match-only/eng/`
(the "SFT w ENG trace; PA match only" checkpoint), then use that
**same checkpoint** as a second-round teacher on exactly the ViNumQA train/valid
samples the *original* teacher (gemma-4-31B-it) failed to solve. Keep only
self-generated traces whose program matches gold under PA, tag them
`qa.trace_source = "self_distilled"` (vs. `"independent"` for the original
teacher-verified rows), and merge into the base set.
See `notebooks/vinumqa/sft-w-reasoning-trace-distill/qwen3-4b-self-distill-teacher-failed.ipynb`.

| File | Self-distilled from |
|---|---|
| `train_qwen3-4b.json` / `valid_qwen3-4b.json` | Qwen3-4B's own checkpoint, single generation per sample |
| `train_qwen3-4b-thinking.json` / `valid_qwen3-4b-thinking.json` | qwen3-4b-thinking's own checkpoint, **majority-of-5 voting** (5 independent generations per teacher-failed sample, keep the modal program, first-generation as tiebreak) |

Gemma3-4B's self-distilled pair is still running (teammate).

**Why the two collection strategies differ**: an early single-generation run
looked like a regression for one model, which raised a "self-loop
confirmation bias" concern — traced instead to an evaluation bug (a
`</think>` split failing on ~39% of that checkpoint's outputs). Once fixed,
majority-of-5 voting was found to help further, plausibly because it filters
out "right answer via unsound/coincidental reasoning" cases that a bare
PA-match filter alone doesn't catch — found by manually reading a sample of
self-distilled traces (see `report/progress-log.md`, 5-6/8/2026).

**Not to be confused with `datasets/FinQA/self-distill/`**, whose files are
built the same way but from a checkpoint trained on ViNumQA **+ FinQA**
combined (root README Table 3), and whose `train_*` file therefore is *not*
ViNumQA-only.
