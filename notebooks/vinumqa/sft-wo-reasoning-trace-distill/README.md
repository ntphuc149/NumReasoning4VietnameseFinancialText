# QLoRA SFT — no reasoning trace (baseline)

Fine-tunes Qwen3-4B (4-bit + LoRA via Unsloth) to map context+question
directly to the gold program string, with loss masked to the assistant turn
only (`train_on_responses_only`). No synthesized or distilled reasoning
trace — this is the plain-label baseline that `../sft-w-reasoning-trace-distill/`
is compared against.

| Notebook | Purpose |
|---|---|
| `qwen3-4b-stf-wo-reasoning-trace.ipynb` | Train + inline eval. |
| `qwen3-4b-wo-reasoning-eval-only.ipynb` | Eval-only, for loading a previously-trained adapter without retraining. |

Trained on `datasets/ViNumQA/train.json` / `valid.json` directly (no
`reasoning_trace` field involved).
