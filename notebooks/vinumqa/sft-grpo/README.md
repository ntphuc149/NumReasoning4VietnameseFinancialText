# SFT + GRPO (planned)

`template-sft-grpo-qwen3-4b-base.ipynb` is Unsloth's stock GSM8K GRPO
template, kept as a starting point — **not yet rewritten for ViNumQA**. Still
needs: a reward function combining program validity, execution correctness
(via `notebooks/evaluate/scorer.py`), and conciseness, following the
program-centric policy optimization approach in `papers/`; and the dataset/
prompt wiring swapped from GSM8K to ViNumQA's context+question+program
format. Intended to further refine the best SFT checkpoint from
`../sft-w-reasoning-trace-distill/` beyond supervised imitation.
