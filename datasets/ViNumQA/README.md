# ViNumQA dataset

All files share the FinQA-style schema:
`{pre_text, table, post_text, id, qa: {question, program, exe_ans[, reasoning_trace, trace_source]}}`.

Organized by pipeline stage — **each subdirectory has its own `README.md`**
with exact file/sample-count breakdowns; this file is just the map.

```
origin/                      original ViNumQA splits (train/valid/test/private_test) -- ground truth, never edited
backward-rationale-distill/  superseded distillation pipeline (teacher shown the gold answer) -- kept for comparison only
distill-from-gemma-teacher/  current distillation pipeline (independent-solve, teacher: gemma-4-31B-it)
self-distill/                STaNR: self-distillation from a model's own fine-tuned checkpoint (ViNumQA-only training)
```

`test.json` (in `origin/`) is the **fixed evaluation set for every result in
the root README** — never used for training, validation, or any filtering.
`valid.json` (also in `origin/`) is the validation set for every SFT run,
whether the training set is ViNumQA alone or ViNumQA+FinQA combined (see
`datasets/FinQA/self-distill/README.md`).

For the ViNumQA+FinQA combined training-set variant (root README Table 3),
see `datasets/FinQA/README.md` and `datasets/FinQA/self-distill/README.md`.
