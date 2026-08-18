# ViNumQA — original splits

Unmodified ViNumQA (VLSP 2025 NumQA shared task). No reasoning trace, no
filtering — the ground truth every other file in `datasets/` derives from.

| File | Samples | Notes |
|---|---|---|
| `train.json` | 2,993 | gold `program` + `exe_ans` |
| `valid.json` | 584 | gold `program` + `exe_ans` — **the validation set used for every SFT run in this repo**, regardless of which training-set variant is used (ViNumQA-only or ViNumQA+FinQA combined) |
| `test.json` | 497 | gold provided — **the fixed evaluation set for every result in the root README**, never used for training or validation |
| `private_test.json` | 1,625 | **no gold** — leaderboard-only, not used locally |

Schema: `{pre_text, table, post_text, id, qa: {question, program, exe_ans}}`.
