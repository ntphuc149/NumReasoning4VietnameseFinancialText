# 0-shot prompting baselines

Zero in-context examples — the model gets only the shared `SYSTEM_MESSAGE`
(operator list + formatting rules) plus the context/question, no worked
example. Every notebook here shares the same context formatting
(`pre_text`/`table`/`post_text`/`question`, table rendered as GitHub-flavored
markdown) and scores against `notebooks/evaluate/scorer.py`, so results are
directly comparable across models.

| Notebook | Coverage |
|---|---|
| `vsf-0-shot-vinumqa.ipynb` | Shared template for API models via an OpenAI-compatible endpoint — change the `MODEL` constant and re-run to add a new model's row to the results table. |
| `vsf-0-shot-vinumqa-gpt5nano-batch.ipynb` | `gpt-5-nano` via the OpenAI Batch API (50% cheaper than sync calls, needs a real `OPENAI_API_KEY`, not a third-party proxy). |
| `vsf-vinumqa-0-shot-qwen3-4b.ipynb` | Local Qwen3-4B, designed to run on Kaggle's free T4/P100 tier. |
| `vsf-0-shot-qwen3-4b-thinking-2507-modal.ipynb` | Qwen3-4B-Thinking-2507, run on Modal (thinking-only model — always emits a `<think>` block, no `enable_thinking` flag). |

See the root README's results table for the full per-model PA/EA numbers this
setting has produced so far.
