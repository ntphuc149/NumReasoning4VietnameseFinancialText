"""Backend routing: let every node run on any of this repo's baseline models.

`llm.LLMClient` only ever talks to one place: the OpenAI-compatible endpoint in
`.env`. That covers eight of this repo's eleven baseline models fine
(gemma-3-27b-it, gemma-4-31B-it, gpt-oss-20b, gpt-oss-120b,
Llama-3.3-70B-Instruct, DeepSeek-V4-Flash, GLM-5.2, gpt-5-nano) -- they are
only ever run through that endpoint anywhere in this repo. The other three
(Qwen3-4B, qwen3-4b-thinking, Gemma3-4B) are the models this repo actually
fine-tunes, and every other notebook that runs them
(`notebooks/vinumqa/0-shot/vsf-vinumqa-0-shot-{qwen3-4b,gemma3-4b}.ipynb`)
loads them locally with plain `transformers` -- there is no API entry for
them, and inventing one (standing up a local OpenAI-compatible server) would
be a second, untested transport for something this repo already knows how to
run directly.

So there are two backends, not one:

    APIBackend    llm.LLMClient, unchanged -- large hosted models
    LocalBackend  transformers AutoModelForCausalLM.generate() -- the 3 models
                  this repo trains, loaded the same way the 0-shot notebooks do

`MultiModelClient` is what `Runner` actually constructs. It looks at the
`model` argument on every `complete()`/`sample_n()` call and dispatches to
whichever backend serves that name, lazily constructing and caching each
backend the first time its model is actually used -- so a run that only uses
API models never needs a GPU, and a run that only uses local models never
needs API_KEY/BASE_URL. Every node in `agents.py` calls `self.client.complete(
..., model=self.config.model_planner, ...)` already; nothing there changes.
Point `MODEL` at any of the eleven and the right backend is chosen for you.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Optional, Protocol, Sequence, runtime_checkable

from agentic.config import AgentConfig
from agentic.llm import LLMClient, LLMError, Usage

# =============================================================================
# THE BACKEND PROTOCOL -- LLMClient already satisfies this; LocalBackend below
# implements it fresh. Structural typing, so LLMClient did not need to change.
# =============================================================================


@runtime_checkable
class Backend(Protocol):
    usage: Usage

    def complete(self, system: Optional[str], user: str, model: str,
                 max_tokens: int, temperature: Optional[float] = None) -> str: ...

    def sample_n(self, system: Optional[str], user: str, model: str, n: int,
                 max_tokens: int, temperature: Optional[float] = None,
                 max_workers: int = 15) -> list[str]: ...


# =============================================================================
# THE REGISTRY -- the three models this repo runs locally
# =============================================================================


@dataclass(frozen=True)
class LocalModelSpec:
    repo: str            # HF repo id, exactly what the 0-shot notebooks load
    chat_style: str       # "qwen" (string content) | "gemma" (typed parts)
    thinking: bool = False  # strip a </think> span out of the decoded output


# Keys are exactly the names in this repo's baseline model list, so `MODEL =
# "Qwen3-4B"` in the notebook is both the display name and the registry key.
MODEL_REGISTRY: dict[str, LocalModelSpec] = {
    "Qwen3-4B": LocalModelSpec(repo="Qwen/Qwen3-4B", chat_style="qwen"),
    "qwen3-4b-thinking": LocalModelSpec(
        repo="Qwen/Qwen3-4B-Thinking-2507", chat_style="qwen", thinking=True,
    ),
    "Gemma3-4B": LocalModelSpec(repo="unsloth/gemma-3-4b-it", chat_style="gemma"),
}

# Documentation only -- NOT an allowlist. Any name absent from MODEL_REGISTRY
# falls through to the API backend, so a model the endpoint serves under a
# name not listed here still works; this is just what `describe_backend()` and
# the notebook's startup print check against for a friendlier message.
API_MODELS = [
    "gemma-3-27b-it", "gemma-4-31B-it", "gpt-oss-20b", "gpt-oss-120b",
    "Llama-3.3-70B-Instruct", "DeepSeek-V4-Flash", "GLM-5.2", "gpt-5-nano",
]


def describe_backend(model: str) -> str:
    """'local' or 'api' -- which backend a model name would route to."""
    return "local" if model in MODEL_REGISTRY else "api"


# =============================================================================
# CHAT-MESSAGE SHAPE -- one function, unit-testable without a model loaded
# =============================================================================


def build_messages(chat_style: str, system: Optional[str], user: str) -> list[dict]:
    """Chat messages in the shape each model family's own template expects.

    Gemma3's template wants typed content parts -- copied from
    notebooks/vinumqa/0-shot/vsf-vinumqa-0-shot-gemma3-4b.ipynb, cell 6, the
    proven-working shape for this exact repo/model pair. Qwen3 takes plain
    string content, matching vsf-vinumqa-0-shot-qwen3-4b.ipynb. Both put
    `system` first when one is given; every node in this package that wants no
    system turn already passes `system=None`.
    """
    if chat_style == "gemma":
        def part(text: str) -> list[dict]:
            return [{"type": "text", "text": text}]

        messages = []
        if system:
            messages.append({"role": "system", "content": part(system)})
        messages.append({"role": "user", "content": part(user)})
        return messages

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": user})
    return messages


# =============================================================================
# THINKING-MODEL OUTPUT -- also a pure function; needs only token ids
# =============================================================================

# Same constant this repo's Qwen3-thinking eval notebooks hardcode
# (notebooks/vinumqa/sft-w-reasoning-trace-distill/
#  qwen3-4b-thinking-2507-eval-only.ipynb): id of the literal token "</think>"
# in Qwen3's vocabulary. Cross-checked against the loaded tokenizer's own id
# in `strip_think` below rather than trusted blindly, so a future tokenizer
# change is caught instead of silently mis-slicing every completion.
QWEN3_THINK_END_TOKEN_ID = 151668


def strip_think(tokenizer, output_ids: Sequence[int]) -> str:
    """Keep only what follows the model's own final </think>.

    Same slicing this repo's Qwen3-thinking eval notebooks use: search from
    the END for the </think> token id, so an output that mentions the literal
    string earlier (inside its own reasoning) is not mistaken for the close.
    A generation that never closes </think> was almost certainly cut off by
    `max_tokens` -- there is nothing to slice, so the whole (truncated,
    still-thinking) text comes back and will most likely fail to parse
    downstream, which is the honest outcome: a `parse:` error in the trace,
    not a silently wrong answer.
    """
    ids = list(output_ids)
    looked_up = tokenizer.convert_tokens_to_ids("</think>")
    end_id = looked_up if isinstance(looked_up, int) and looked_up >= 0 else QWEN3_THINK_END_TOKEN_ID
    try:
        cut = len(ids) - ids[::-1].index(end_id)
    except ValueError:
        cut = 0
    return tokenizer.decode(ids[cut:], skip_special_tokens=True).strip()


# =============================================================================
# LOCAL BACKEND
# =============================================================================


class LocalBackend:
    """One locally-loaded model, served through plain `transformers.generate()`.

    Deliberately not Unsloth: Unsloth's `FastLanguageModel`/`FastModel` exist
    for training (LoRA, 4-bit, `fast_inference`'s vLLM engine); this package
    never trains anything, and the 0-shot notebooks this mirrors do not use it
    either -- `AutoModelForCausalLM.from_pretrained(..., torch_dtype="auto",
    device_map="cuda:0")` is the whole load.

    n-sampling uses `num_return_sequences=n` in ONE generate() call rather than
    n separate ones, the same choice the eval notebooks make and for the same
    reason: one prompt's KV cache is shared instead of paid for n times.

    A single GPU cannot run two generate() calls at once. `agents.py` fans out
    over subqueries and over n-samples with a `ThreadPoolExecutor`, which is
    correct for an API backend (each call is an independent HTTP request) and
    would corrupt a local model's forward pass if concurrent calls were let
    through here -- hence `_lock` wraps every actual `generate()`. Concurrency
    configured via `AgentConfig.max_workers_*` has no effect when every
    `model_*` field names a local model; that is the expected, correct
    tradeoff for a single GPU, not a bug -- the same "one prompt at a time,
    deliberately" this repo's own eval notebooks already document.
    """

    DEFAULT_MAX_SEQ_LENGTH = 8192

    def __init__(self, name: str, spec: LocalModelSpec, config: AgentConfig,
                 max_seq_length: int = DEFAULT_MAX_SEQ_LENGTH,
                 max_batch_size: Optional[int] = None):
        self.name = name
        self.spec = spec
        self.config = config
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        self.usage = Usage()
        self._lock = threading.Lock()
        self._model = None
        self._tokenizer = None

    # ------------------------------------------------------------- loading --
    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        with self._lock:
            if self._model is not None:
                return
            # Imported here, not at module level: the offline test suite and
            # any run that only uses API models must not require torch/
            # transformers (or a GPU) to be installed at all.
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer

            # NOT torch_dtype="auto": that reads the checkpoint's own preferred
            # dtype, which for Qwen3 is bfloat16 -- fine on Ampere+ (Modal's
            # A100), but a Turing-generation GPU (Kaggle's T4) has no native
            # bf16 tensor cores, so bf16 ops fall back to a much slower path.
            # Measured cause of a real run being ~20-30 min/sample instead of
            # the expected tens of seconds.
            #
            # NOT torch.cuda.is_bf16_supported() either -- measured on a real
            # T4: it returns True there too (it only checks that bf16 ops run
            # without erroring, via CUDA's software fallback when there is no
            # tensor-core support, not that they run fast). Compute capability
            # is the real signal: bf16 tensor cores start at Ampere (SM 8.0).
            major, _minor = torch.cuda.get_device_capability()
            dtype = torch.bfloat16 if major >= 8 else torch.float16
            print(f"LocalBackend: loading {self.spec.repo} for {self.name!r} (dtype={dtype}) ...")
            self._tokenizer = AutoTokenizer.from_pretrained(self.spec.repo)
            self._model = AutoModelForCausalLM.from_pretrained(
                self.spec.repo, torch_dtype=dtype, device_map="cuda:0",
            )
            print(f"LocalBackend: {self.name!r} ready.")

    # -------------------------------------------------------------- decode --
    def _generate(self, system: Optional[str], user: str, max_tokens: int,
                  temperature: Optional[float], n: int) -> list[str]:
        """The one place that calls generate(). Caller must hold `_lock`.

        `num_return_sequences=n` shares one prompt's KV cache across all n
        completions -- cheap when it fits, but EVERY sequence's KV cache has
        to live in VRAM at once for the whole call, and that scales with n
        AND with prompt length. Measured cause of a real failure: n=15 at a
        ~4-5k token planner prompt OOM'd on a 14.56 GB Kaggle T4 (all 10
        samples in the run, ~800 MiB short each time) -- there was no retry,
        so every sample lost its entire planner stage. Retrying at a SMALLER
        n on OOM, rather than assuming one fixed "safe" batch size, matches
        this file's other adaptive-retry (llm.py's reasoning-budget
        escalation): the safe batch size depends on prompt length, which
        varies per sample, so it has to be discovered per call, not guessed
        once for every model/GPU/prompt combination.
        """
        import torch

        tokenizer, model = self._tokenizer, self._model
        messages = build_messages(self.spec.chat_style, system, user)
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True,
            return_tensors="pt", return_dict=True,
        ).to(model.device)

        prompt_len = inputs["input_ids"].shape[-1]
        budget = self.max_seq_length - prompt_len
        if budget <= 0:
            raise LLMError(
                f"{self.name}: prompt ({prompt_len} tokens) already exceeds "
                f"max_seq_length={self.max_seq_length}; nothing left to "
                f"generate. Raise LocalBackend.max_seq_length if this "
                f"pipeline's prompts genuinely need more room."
            )
        new_tokens = min(max_tokens, budget)

        cfg = self.config
        temp = cfg.temperature if temperature is None else temperature
        base_kwargs: dict = {"max_new_tokens": new_tokens}
        if temp is not None and temp > 0:
            base_kwargs.update(do_sample=True, temperature=temp, top_p=cfg.top_p)
            if cfg.send_top_k:
                base_kwargs["top_k"] = cfg.top_k
        else:
            base_kwargs["do_sample"] = False

        texts: list[str] = []
        remaining = n
        # Optimistic (try the whole thing in one call) unless the caller has
        # already told us a smaller size is known-safer for this GPU -- see
        # AgentConfig.local_max_batch_size.
        batch_size = n if self.max_batch_size is None else min(n, self.max_batch_size)
        while remaining > 0:
            this_batch = min(batch_size, remaining)
            gen_kwargs = dict(base_kwargs)
            if this_batch > 1:
                gen_kwargs["num_return_sequences"] = this_batch

            try:
                with torch.no_grad():
                    out = model.generate(**inputs, **gen_kwargs)
            except torch.cuda.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                if this_batch <= 1:
                    # Nothing smaller left to try -- a single sequence at
                    # this prompt length genuinely does not fit.
                    raise LLMError(
                        f"{self.name}: out of memory generating a single "
                        f"sequence at prompt_len={prompt_len} tokens, "
                        f"max_new_tokens={new_tokens} -- lower "
                        f"max_seq_length or max_tokens_*, or use a smaller "
                        f"model on this GPU."
                    ) from exc
                batch_size = max(1, this_batch // 2)
                print(f"LocalBackend: OOM at batch={this_batch} for {self.name!r} "
                      f"(prompt_len={prompt_len}); retrying at batch={batch_size}.")
                continue  # retry this same remaining chunk, not the whole n

            gen_only = [row[prompt_len:].tolist() for row in out]
            completion_tokens = sum(len(g) for g in gen_only)
            self.usage.add(prompt_len, completion_tokens, prompt_len + completion_tokens)
            if self.spec.thinking:
                texts.extend(strip_think(tokenizer, g) for g in gen_only)
            else:
                texts.extend(tokenizer.decode(g, skip_special_tokens=True).strip() for g in gen_only)
            remaining -= this_batch

        return texts

    # --------------------------------------------------------------- public --
    def complete(self, system: Optional[str], user: str, model: str,
                 max_tokens: int, temperature: Optional[float] = None) -> str:
        self._ensure_loaded()
        with self._lock:
            outputs = self._generate(system, user, max_tokens, temperature, n=1)
        if not outputs or not outputs[0]:
            raise LLMError(f"{self.name}: generation returned empty output")
        return outputs[0]

    def sample_n(self, system: Optional[str], user: str, model: str, n: int,
                 max_tokens: int, temperature: Optional[float] = None,
                 max_workers: int = 15) -> list[str]:
        if n <= 1:
            return [self.complete(system, user, model, max_tokens, temperature)]
        self._ensure_loaded()
        with self._lock:
            outputs = self._generate(system, user, max_tokens, temperature, n=n)
        usable = [o for o in outputs if o]
        if not usable:
            raise LLMError(f"{self.name}: n-sampling produced no usable output")
        return usable


# =============================================================================
# THE ROUTER -- what Runner actually constructs
# =============================================================================


class MultiModelClient:
    """Dispatches complete()/sample_n() to the backend that serves `model`.

    Drop-in replacement for `LLMClient` from `Runner`'s point of view: same two
    public methods, same `.usage`. Backends are built lazily and cached, keyed
    by model name -- a run that names four different models in the four
    `AgentConfig.model_*` fields (nothing stops a run from putting a small
    local model on the cheap extraction nodes and a large API model on the
    planner, or vice versa) ends up with exactly the backends it actually used,
    no more.
    """

    def __init__(self, config: AgentConfig, api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.config = config
        self._api_key = api_key
        self._base_url = base_url
        self._api_backend: Optional[LLMClient] = None
        self._local_backends: dict[str, LocalBackend] = {}
        self._lock = threading.Lock()

    def _backend_for(self, model: str) -> Backend:
        spec = MODEL_REGISTRY.get(model)
        if spec is None:
            return self._get_api_backend()
        return self._get_local_backend(model, spec)

    def _get_api_backend(self) -> LLMClient:
        if self._api_backend is None:
            with self._lock:
                if self._api_backend is None:
                    # Constructed here, not eagerly in __init__: a run that
                    # only ever names local models must not require
                    # API_KEY/BASE_URL to be set at all.
                    self._api_backend = LLMClient(
                        self.config, api_key=self._api_key, base_url=self._base_url,
                    )
        return self._api_backend

    def _get_local_backend(self, model: str, spec: LocalModelSpec) -> LocalBackend:
        backend = self._local_backends.get(model)
        if backend is None:
            with self._lock:
                backend = self._local_backends.get(model)
                if backend is None:
                    backend = LocalBackend(
                        model, spec, self.config,
                        max_batch_size=self.config.local_max_batch_size,
                    )
                    self._local_backends[model] = backend
        return backend

    # --------------------------------------------------------------- public --
    def complete(self, system: Optional[str], user: str, model: str,
                 max_tokens: int, temperature: Optional[float] = None) -> str:
        return self._backend_for(model).complete(
            system, user, model, max_tokens, temperature
        )

    def sample_n(self, system: Optional[str], user: str, model: str, n: int,
                 max_tokens: int, temperature: Optional[float] = None,
                 max_workers: int = 15) -> list[str]:
        return self._backend_for(model).sample_n(
            system, user, model, n, max_tokens, temperature, max_workers
        )

    @property
    def usage(self) -> Usage:
        """Combined usage across every backend this client has ever used."""
        total = Usage()
        backends: list[Backend] = list(self._local_backends.values())
        if self._api_backend is not None:
            backends.append(self._api_backend)
        for backend in backends:
            u = backend.usage
            total.requests += u.requests
            total.prompt_tokens += u.prompt_tokens
            total.completion_tokens += u.completion_tokens
            total.total_tokens += u.total_tokens
        return total


__all__ = [
    "API_MODELS",
    "Backend",
    "LocalBackend",
    "LocalModelSpec",
    "MODEL_REGISTRY",
    "MultiModelClient",
    "QWEN3_THINK_END_TOKEN_ID",
    "build_messages",
    "describe_backend",
    "strip_think",
]
