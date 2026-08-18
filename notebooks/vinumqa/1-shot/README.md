# 1-shot prompting baselines

One in-context example — the shared `SYSTEM_MESSAGE` plus a single
demonstration (drawn from the VLSP 2025 paper's own Figure 1 example),
followed by the real context/question. Same shared formatting/scoring as the
0-shot and few-shot notebooks.

| Notebook | Coverage |
|---|---|
| `vsf-1-shot-vinumqa.ipynb` | Shared template for API models via an OpenAI-compatible endpoint — change the `MODEL` constant and re-run to add a new model's row. |
| `vsf-1-shot-vinumqa-gpt5nano-batch.ipynb` | `gpt-5-nano` via the OpenAI Batch API. |
| `vsf-vinumqa-1-shot-qwen3-4b.ipynb` | Local Qwen3-4B (Kaggle). |
| `vsf-1-shot-qwen3-4b-thinking-2507-modal.ipynb` | Qwen3-4B-Thinking-2507 (Modal). |
| `vsf-1-shot-gemma-4-31b-it.ipynb` | gemma-4-31B-it — the strongest ICL model measured so far (see root README). |
| `vsf-1-shot-gpt-oss-120b.ipynb` | gpt-oss-120b. |
| `vsf-1-shot-glm5.1.ipynb` | GLM-5.1, via FPT AI Factory. |
| `vsf-1-shot-glm5.2.ipynb` | GLM-5.2, via FPT AI Factory — rate-limited (RPM=50/TPM=100,000), see `../few-shot/README.md` for the rate-limiting design shared with the few-shot GLM-5.2 notebook. |

See the root README's results table for the full per-model PA/EA numbers.
