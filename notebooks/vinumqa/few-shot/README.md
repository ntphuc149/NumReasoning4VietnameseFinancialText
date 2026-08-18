# Few-shot prompting baselines (3 examples)

Three in-context examples — the shared `SYSTEM_MESSAGE` plus one hand-picked
demonstration per evidence category (Table Only / Text Only / Table & Text,
matching the VLSP 2025 paper's own question-type breakdown), followed by the
real context/question.

| Notebook | Coverage |
|---|---|
| `vsf-few-shot-vinumqa.ipynb` | Shared template for API models via an OpenAI-compatible endpoint — change the `MODEL` constant and re-run to add a new model's row. |
| `vsf-few-shot-vinumqa-gpt5nano-batch.ipynb` | `gpt-5-nano` via the OpenAI Batch API. |
| `vsf-vinumqa-few-shot-qwen3-4b.ipynb` | Local Qwen3-4B (Kaggle). |
| `vsf-few-shot-qwen3-4b-thinking-2507-modal.ipynb` | Qwen3-4B-Thinking-2507 (Modal). |
| `vsf-few-shot-vinumqa-glm5.2.ipynb` | GLM-5.2, via FPT AI Factory. |
| `vsf-few-shot-vinumqa-glm5.2-rate-check.ipynb` | Not a scored run — measures GLM-5.2's real per-request token cost via `usage.prompt_tokens`/`total_tokens` and confirms it returns reasoning in a separate `message.reasoning_content` field. Read this first if extending the GLM-5.2 notebook to a new rate-limited endpoint. |

## GLM-5.2 rate limiting

FPT AI Factory enforces RPM=50/TPM=100,000 on GLM-5.2. Measured over 15 probe
requests (`*-rate-check.ipynb`): mean ~2,721 tokens/request, so **TPM — not
RPM — is the binding constraint** (~36.7 req/min safe ceiling). The scored
notebook (`vsf-few-shot-vinumqa-glm5.2.ipynb`) implements:

- A sliding-window `RateLimiter` that throttles before each request and is
  corrected afterward with the real `usage.total_tokens`.
- The OpenAI SDK's own retry-on-429 disabled (`max_retries=0`) — it was
  issuing retries invisible to the limiter, silently under-counting what was
  actually sent and letting RPM get exceeded from underneath it. All retry
  logic is explicit instead (parses `Retry-After`, resets the limiter window,
  retries up to 5 attempts).
- Per-sample checkpointing to a JSON file, so an interruption doesn't require
  regenerating from sample 0.

See the root README's results table for the full per-model PA/EA numbers.
