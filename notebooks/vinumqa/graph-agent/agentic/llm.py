"""Everything that goes to the model: how the context is rendered, and transport.

Two sections:

1. **Context formatting** -- builds `C` in the paper's notation. The per-field
   formatting is copied from this repo's prompting notebooks
   (`notebooks/vinumqa/0-shot/vsf-0-shot-vinumqa-gemma-4-31B-it.ipynb`, cell 2):
   paragraphs joined with newlines, table rendered as a GitHub-flavoured
   markdown table via `tabulate`. That matters -- if the agent saw a
   differently-rendered table than the 0-shot/1-shot/few-shot baselines did, its
   PA/EA would no longer be comparable to the rows already in the root README.
   Only the wrapper around those fields is new, and it follows the labelling in
   the paper's Figure 1.

2. **Transport** -- an OpenAI-compatible client. Three things here are not
   generic boilerplate but lessons already paid for in this repo:

   * `max_retries=0` on the client. The OpenAI SDK's built-in retry-on-429
     issues requests the local RateLimiter never sees, so the limiter's window
     under-counts what was actually sent and the server-side cap gets exceeded
     from underneath it. All retry logic is explicit here instead.
     (notebooks/vinumqa/few-shot/vsf-few-shot-vinumqa-glm5.2.ipynb, cell 6)
   * The `RateLimiter` itself -- sliding 60s window over both RPM and TPM,
     pre-throttled on an estimate and corrected afterwards with the real
     `usage.total_tokens`. Made thread-safe here, which the notebook version did
     not need to be: this package fans out across subqueries, across n-samples,
     and across dataset rows, so several threads share one limiter.
   * Capability probing instead of name-guessing. The repo already learned that
     `DeepSeek-V4-Flash` returns `reasoning_content` despite matching no name
     heuristic. The same applies to server-side `n>1`: many proxies accept the
     parameter and return a single choice anyway, so it is probed once with a
     real request and cached.
"""

from __future__ import annotations

import math
import os
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from openai import OpenAI
from tabulate import tabulate

from agentic.config import AgentConfig

# =============================================================================
# 1. CONTEXT FORMATTING -- `C` in the paper's notation
# =============================================================================

_CONTEXT_FRAME_VI = """- Văn bản trước bảng:
{pre_text}

- Bảng:
{table}

- Văn bản sau bảng:
{post_text}"""

_CONTEXT_FRAME_EN = """- Context before table:
{pre_text}

- Table:
{table}

- Context after table:
{post_text}"""


def format_pre_text(sample: Mapping[str, Any]) -> str:
    return "\n".join(sample["pre_text"])


def format_post_text(sample: Mapping[str, Any]) -> str:
    return "\n".join(sample["post_text"])


def format_table(sample: Mapping[str, Any]) -> str:
    """GitHub-flavoured markdown, first row treated as the header."""
    table = sample["table"]
    if not table:
        return ""
    return tabulate(table[1:], headers=table[0], tablefmt="github")


def format_context(sample: Mapping[str, Any], lang: str = "vi") -> str:
    """Build `C` -- the single {context} string every Appendix B prompt takes."""
    frame = _CONTEXT_FRAME_VI if lang == "vi" else _CONTEXT_FRAME_EN
    empty = "(không có)" if lang == "vi" else "(none)"
    return frame.format(
        pre_text=format_pre_text(sample) or empty,
        table=format_table(sample) or empty,
        post_text=format_post_text(sample) or empty,
    )


def table_row_labels(table: Sequence[Sequence[str]] | None) -> set[str]:
    """First cell of every row -- the exact strings `table_*` may name.

    `eval_program` looks a row up as `{row[0]: row[1:] for row in table}`, so a
    label that is not in this set makes the program execute to "n/a".
    """
    if not table:
        return set()
    return {str(row[0]) for row in table if len(row) > 0}


# =============================================================================
# 2. TRANSPORT
# =============================================================================

# Reasoning models emit a chain-of-thought before their answer and need a far
# larger token budget or they get cut off mid-output. Same heuristic as the
# prompting notebooks.
_REASONING_KEYWORDS = ("r1", "thinking", "reasoner", "qwq", "o1", "o3", "reasoning")

# Confirmed by this repo's own baseline notebooks against the REAL endpoint --
# not name-guessed. Both are on this package's own 11-model baseline list.
#   * DeepSeek-V4-Flash: notebooks/vinumqa/0-shot/
#     vsf-0-shot-vinumqa-deepseek-v4-flash.ipynb forces IS_REASONING=True for
#     it specifically, discovered after a real run crashed with content=None
#     at sample 483 -- its name matches none of _REASONING_KEYWORDS above.
#   * GLM-5.2: notebooks/vinumqa/few-shot/vsf-few-shot-vinumqa-glm5.2-rate-
#     check.ipynb dumped a raw response and found a populated
#     reasoning_content field.
# is_reasoning_model() checks this set FIRST, so a model matching neither this
# set nor the keyword heuristic (the common case) costs one dict lookup, not a
# live probe.
KNOWN_REASONING_MODELS = frozenset({"DeepSeek-V4-Flash", "GLM-5.2"})


def is_reasoning_model(model_name: str) -> bool:
    if model_name in KNOWN_REASONING_MODELS:
        return True
    name = model_name.lower()
    return any(kw in name for kw in _REASONING_KEYWORDS)


# Floor for a reasoning model's FIRST attempt, so it does not have to pay for
# one guaranteed-too-small round trip before _call()'s escalation kicks in.
# 2048 rather than the 8192 the 0-shot baseline notebooks use flat: those
# notebooks send one long direct-prompt per sample and never revisit the
# number; this package's four nodes have much shorter prompts (see
# backends.py's LocalBackend docstring for the measured planner-prompt
# lengths), and escalation still reaches 8192 (2048 x 2 x 2) if a harder
# sample's reasoning genuinely needs it -- this floor only removes the
# common-case first miss, not the ceiling.
REASONING_MODEL_MIN_TOKENS = 2048


class LLMError(RuntimeError):
    """Every retry for a request was exhausted."""


class RateLimiter:
    """Thread-safe sliding-window limiter over requests/min and tokens/min.

    Call `wait_if_needed(estimate)` before a request and `record(actual)` after
    it. A limiter constructed with both limits None is inert.
    """

    def __init__(
        self,
        rpm_limit: Optional[int] = None,
        tpm_limit: Optional[int] = None,
        window_s: float = 60.0,
    ):
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit
        self.window_s = window_s
        self.request_times: deque[float] = deque()
        self.token_events: deque[tuple[float, int]] = deque()
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.rpm_limit is not None or self.tpm_limit is not None

    def _prune(self, now: float) -> None:
        while self.request_times and now - self.request_times[0] > self.window_s:
            self.request_times.popleft()
        while self.token_events and now - self.token_events[0][0] > self.window_s:
            self.token_events.popleft()

    def wait_if_needed(self, estimated_tokens: int = 0) -> None:
        if not self.enabled:
            return
        while True:
            with self._lock:
                now = time.monotonic()
                self._prune(now)
                over_rpm = (
                    self.rpm_limit is not None
                    and len(self.request_times) >= self.rpm_limit
                )
                over_tpm = (
                    self.tpm_limit is not None
                    and sum(t for _, t in self.token_events) + estimated_tokens
                    > self.tpm_limit
                )
                if not over_rpm and not over_tpm:
                    return
                oldest = min(
                    self.request_times[0] if self.request_times else math.inf,
                    self.token_events[0][0] if self.token_events else math.inf,
                )
                sleep_for = max(0.05, self.window_s - (now - oldest))
            # Sleep outside the lock so other threads can still account.
            time.sleep(sleep_for)

    def record(self, actual_tokens: int) -> None:
        if not self.enabled:
            return
        with self._lock:
            now = time.monotonic()
            self.request_times.append(now)
            self.token_events.append((now, actual_tokens))

    def on_rate_limit_error(self, retry_after: Optional[float] = None) -> None:
        """Treat the window as saturated for `retry_after` seconds.

        A 429 is proof that local accounting has drifted from the server's, so
        correcting towards "full" is safer than trusting the (evidently wrong)
        count.
        """
        if not self.enabled:
            return
        with self._lock:
            now = time.monotonic()
            self.request_times.clear()
            self.token_events.clear()
            pause = retry_after if retry_after is not None else self.window_s
            self.request_times.append(now + pause - self.window_s)
            self.token_events.append(
                (now + pause - self.window_s, self.tpm_limit or 0)
            )


@dataclass
class Usage:
    requests: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def add(self, prompt: int, completion: int, total: int) -> None:
        with self._lock:
            self.requests += 1
            self.prompt_tokens += prompt
            self.completion_tokens += completion
            self.total_tokens += total

    def as_dict(self) -> dict:
        return {
            "requests": self.requests,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.total_tokens,
        }


def _retry_after_seconds(exc: Exception) -> Optional[float]:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    for key in ("retry-after", "Retry-After"):
        value = headers.get(key)
        if value:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    return None


class LLMClient:
    """One shared client for every node. Safe to call from many threads."""

    def __init__(self, config: AgentConfig, api_key: str | None = None,
                 base_url: str | None = None):
        self.config = config
        self.api_key = api_key or os.environ["API_KEY"]
        self.base_url = base_url or os.environ["BASE_URL"]
        # See module docstring, point 1.
        self.client = OpenAI(
            api_key=self.api_key, base_url=self.base_url, max_retries=0
        )
        self.limiter = RateLimiter(config.rpm_limit, config.tpm_limit)
        self.usage = Usage()
        self._server_side_n: Optional[bool] = config.use_server_side_n
        self._send_top_k = config.send_top_k
        self._probe_lock = threading.Lock()

    # ------------------------------------------------------------ internals --
    def _build_kwargs(self, max_tokens: int, temperature: Optional[float],
                      n: Optional[int]) -> dict:
        cfg = self.config
        kwargs: dict = {
            "max_tokens": max_tokens,
            "temperature": cfg.temperature if temperature is None else temperature,
            "top_p": cfg.top_p,
            "stream": False,
        }
        if self._send_top_k:
            kwargs["extra_body"] = {"top_k": cfg.top_k}
        if n is not None and n > 1:
            kwargs["n"] = n
        return kwargs

    def _record_usage(self, response) -> int:
        usage = getattr(response, "usage", None)
        if usage is None:
            return 0
        prompt = getattr(usage, "prompt_tokens", 0) or 0
        completion = getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", 0) or (prompt + completion)
        self.usage.add(prompt, completion, total)
        return total

    @staticmethod
    def _content(choice) -> str:
        """Final answer text only.

        A reasoning model keeps its chain-of-thought in
        `message.reasoning_content`, which the API already separates from
        `message.content` -- verified for GLM-5.2 and DeepSeek-V4-Flash in this
        repo's rate-check notebook. So the final program/answer is `content`,
        with no stripping needed.
        """
        message = getattr(choice, "message", None)
        return (getattr(message, "content", None) or "").strip()

    def _call(self, messages: list[dict], model: str, max_tokens: int,
              temperature: Optional[float], n: Optional[int]) -> list[str]:
        cfg = self.config
        # Grows, not fixed: see the reasoning-starvation handling below. Kept
        # local to this call, not written back to cfg, so a node's normal
        # budget for every OTHER call is untouched -- only a call that
        # actually hits the wall pays for a bigger one.
        effective_max_tokens = max_tokens
        if is_reasoning_model(model) and effective_max_tokens < REASONING_MODEL_MIN_TOKENS:
            effective_max_tokens = REASONING_MODEL_MIN_TOKENS
        # A reasoning model can spend its whole token budget on hidden
        # chain-of-thought and never reach the answer. Measured directly on
        # this exact planner prompt against DeepSeek-V4-Flash at the default
        # max_tokens_planner=768: 728 of 768 completion tokens went to
        # reasoning_content, leaving 69 characters for the plan -- one bad
        # question away from empty. Retrying at the SAME max_tokens would hit
        # the identical wall every time (this is not a transient failure),
        # burning cfg.max_retries attempts -- and their reasoning tokens --
        # for a guaranteed LLMError. Doubling the budget when that exact
        # signature (empty content, non-empty reasoning_content) appears is
        # cheaper and actually has a chance of succeeding.
        reasoning_escalations = 0
        MAX_REASONING_ESCALATIONS = 2  # budget may grow to at most 4x the caller's request
        last_error: Exception | None = None
        # Content that arrived but was cut off at max_tokens. Preferred over
        # nothing: an instruct model that overruns its budget usually emits the
        # program first and then keeps talking, so `extract_program` can still
        # recover it. Only used if no untruncated response is ever obtained.
        truncated: list[str] = []

        for attempt in range(1, cfg.max_retries + 1):
            estimate = (
                sum(len(m.get("content", "")) for m in messages) // 4
                + effective_max_tokens
            )
            self.limiter.wait_if_needed(estimate)
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=messages,
                    **self._build_kwargs(effective_max_tokens, temperature, n),
                )
            except Exception as exc:  # noqa: BLE001 - transport errors are varied
                last_error = exc
                name = type(exc).__name__
                text = str(exc)
                # An endpoint that rejects top_k (or n) will do so
                # deterministically; retrying the same body forever is
                # pointless. Drop the parameter and try again immediately.
                if self._send_top_k and "top_k" in text:
                    self._send_top_k = False
                    continue
                if n and n > 1 and "'n'" in text.lower():
                    self._server_side_n = False
                    n = None
                    continue
                if "RateLimit" in name or "429" in text:
                    delay = _retry_after_seconds(exc) or 65.0
                    self.limiter.on_rate_limit_error(delay)
                    time.sleep(delay)
                    continue
                if attempt < cfg.max_retries:
                    time.sleep(cfg.retry_base_delay * attempt)
                continue

            total = self._record_usage(response)
            self.limiter.record(total or estimate)

            outputs = []
            reasoning_starved = False
            for choice in response.choices:
                content = self._content(choice)
                if not content:
                    # Distinguish "cut off mid chain-of-thought" (worth a
                    # bigger budget) from "the model just returned nothing"
                    # (a bigger budget would not help): the former leaves a
                    # non-empty reasoning_content behind, the API's own
                    # signal that generation was still inside the hidden
                    # trace when max_tokens ran out.
                    message = getattr(choice, "message", None)
                    if getattr(message, "reasoning_content", None):
                        reasoning_starved = True
                    continue
                if getattr(choice, "finish_reason", None) == "length":
                    truncated.append(content)
                else:
                    outputs.append(content)
            if outputs:
                return outputs

            if reasoning_starved and reasoning_escalations < MAX_REASONING_ESCALATIONS:
                reasoning_escalations += 1
                effective_max_tokens *= 2
                continue  # not a rate limit or transient error -- no sleep needed

            if attempt < cfg.max_retries:
                time.sleep(cfg.retry_base_delay * attempt)

        if truncated:
            return truncated

        if last_error is not None:
            raise LLMError(
                f"{model}: all {cfg.max_retries} attempts failed"
            ) from last_error
        raise LLMError(f"{model}: all {cfg.max_retries} attempts returned empty output")

    # --------------------------------------------------------------- public --
    def complete(self, system: Optional[str], user: str, model: str,
                 max_tokens: int, temperature: Optional[float] = None) -> str:
        """One completion. Raises LLMError if every attempt fails."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})
        return self._call(messages, model, max_tokens, temperature, n=None)[0]

    def supports_server_side_n(self, model: str) -> bool:
        """Probe once with a real 2-choice request, then cache.

        Cheaper than assuming: if the endpoint honours `n`, an n=15 plan costs
        one prompt instead of fifteen.
        """
        if self._server_side_n is not None:
            return self._server_side_n
        with self._probe_lock:
            if self._server_side_n is not None:
                return self._server_side_n
            try:
                response = self.client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": "Say OK."}],
                    max_tokens=8,
                    temperature=1.0,
                    n=2,
                )
                self._record_usage(response)
                self._server_side_n = len(response.choices) >= 2
            except Exception:  # noqa: BLE001 - any failure means "assume no"
                self._server_side_n = False
            return self._server_side_n

    def sample_n(self, system: Optional[str], user: str, model: str,
                 n: int, max_tokens: int,
                 temperature: Optional[float] = None,
                 max_workers: int = 15) -> list[str]:
        """`P_n-sample` from paper eq. (3): n independent generations.

        Uses one server-side `n=n` request when the endpoint supports it, and
        tops up with extra requests if fewer choices come back than asked for.
        Falls back to `n` concurrent single requests otherwise.
        """
        if n <= 1:
            return [self.complete(system, user, model, max_tokens, temperature)]

        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": user})

        outputs: list[str] = []
        if self.supports_server_side_n(model):
            try:
                outputs = self._call(messages, model, max_tokens, temperature, n=n)
            except LLMError:
                outputs = []
            if len(outputs) >= n:
                return outputs[:n]

        # Top up whatever the server did not return, concurrently.
        from concurrent.futures import ThreadPoolExecutor

        remaining = n - len(outputs)
        if remaining > 0:
            def one(_: int) -> Optional[str]:
                try:
                    return self._call(messages, model, max_tokens, temperature,
                                      n=None)[0]
                except LLMError:
                    return None

            with ThreadPoolExecutor(max_workers=min(max_workers, remaining)) as pool:
                for result in pool.map(one, range(remaining)):
                    if result:
                        outputs.append(result)

        if not outputs:
            raise LLMError(f"{model}: n-sampling produced no usable output")
        return outputs
