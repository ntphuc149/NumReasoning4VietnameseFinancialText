# Notebooks

```
vinumqa/
├── 0-shot/                          0-shot prompting baselines
├── 1-shot/                          1-shot prompting baselines
├── few-shot/                        3-shot prompting baselines
├── distill-reasoning-trace/         teacher → reasoning-trace distillation (CoNR)
├── translate-reasoning-trace/       (abandoned) MT-based trace translation
├── sft-wo-reasoning-trace-distill/  QLoRA SFT, program-only labels (baseline)
├── sft-w-reasoning-trace-distill/   QLoRA SFT, distilled-reasoning-trace labels
├── sft-grpo/                        (planned) SFT + GRPO policy optimization
└── graph-agent/                     MPR-Agent: inference-only multi-agent pipeline (no training)
evaluate/                            shared PA/EA scorer (scorer.py) — read this first
translated-finqa/                    (placeholder) planned baselines on translated-FinQA
```

Every prompting/SFT notebook shares the same `SYSTEM_MESSAGE` (operator list +
formatting rules), the same `pre_text`/`table`/`post_text`/`question` context
formatting (table rendered as GitHub-flavored markdown via `tabulate`), and
the same scorer (`evaluate/scorer.py`), so results across all of them are
directly comparable. Each subdirectory above has its own `README.md` with the
per-notebook breakdown — start there for anything beyond a quick orientation.

See the root `README.md` for the task definition, results table, and how to
run these notebooks (API keys, local vs. Kaggle vs. Modal).
