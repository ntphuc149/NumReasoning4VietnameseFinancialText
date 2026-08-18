"""Configuration for the MPR-Agent pipeline.

Every value the paper states explicitly is defaulted to the paper's value and
annotated with where it comes from. Every value the paper does *not* state
(concurrency, retries, fallback behaviour) is defaulted conservatively and
annotated as such, so a reader can tell at a glance which knobs are "as
published" and which are this implementation's own.
"""

from dataclasses import dataclass, field
from typing import Literal, Optional

# Model names as served by the OpenAI-compatible endpoint in `.env` (BASE_URL).
# gemma-4-31B-it is this repo's strongest in-context baseline (few-shot(3):
# PA 0.5674 / EA 0.6137 on test.json), so running the agent on the same model
# measures the architecture's contribution rather than a model swap -- the same
# comparison the paper draws in its Table 2.
DEFAULT_MODEL = "gemma-4-31B-it"

VoteMode = Literal["canonical", "symbolic"]
PromptLang = Literal["vi", "en"]


@dataclass
class AgentConfig:
    # ------------------------------------------------------------- models --
    # The paper uses one model for all four nodes (Qwen3-8B / Qwen3-32B). These
    # are split per node so the cheap extraction nodes can be pointed at a
    # smaller model without touching the planner.
    model_subquery_gen: str = DEFAULT_MODEL
    model_subquery_ans: str = DEFAULT_MODEL
    model_planner: str = DEFAULT_MODEL
    model_fallback: str = DEFAULT_MODEL

    # ----------------------------------------- sampling (paper section 5.1) --
    n_samples: int = 15          # "n-sampling parameter of n = 15"
    temperature: float = 0.6     # "a sampling temperature of 0.6"
    top_p: float = 0.95          # "top_p ... set to 0.95"
    top_k: int = 20              # "top_k ... set to 20" -- sent via extra_body
    send_top_k: bool = True      # auto-disabled if the endpoint rejects it

    # Planner-only temperature override; None means "use `temperature` above",
    # which is the paper's configuration.
    #
    # Worth knowing before leaving this at None: measured on gemma-4-31B-it over
    # 30 test samples, n=15 at temperature 0.6 produced a *single distinct plan*
    # in 26 of them, so the multi-path stage had nothing to choose between and
    # oracle@15 equalled plain accuracy exactly. The endpoint is not at fault --
    # a control probe returned 9 distinct completions out of 15 on an open-ended
    # prompt, and the same 1-distinct result comes back from a raw server-side
    # n=15 call and from 15 separate requests alike. The planner task is simply
    # near-deterministic once decomposition has pinned the numbers down.
    temperature_planner: Optional[float] = None

    max_tokens_subquery_gen: int = 1024
    max_tokens_subquery_ans: int = 512
    max_tokens_planner: int = 768
    max_tokens_fallback: int = 512

    # -------------------------------------------------------- prompt choice --
    # Appendix B.1/B.3/B.5/B.7/B.9/B.11 are Vietnamese; B.2/B.4/B.6/B.8/B.10/B.12
    # are the paper's own English translations of the same prompts. The task is
    # Vietnamese, so "vi" is the default.
    prompt_lang: PromptLang = "vi"

    # Opt-in prompt patches that are NOT in the paper. Off by default so the
    # headline run is prompt-faithful; see the "opt-in extensions" section of
    # prompts.py for what each one changes and which failure mode it targets.
    use_prompt_ext: bool = False

    # --------------------------------------- ablation flags (paper Table 4) --
    # "Multi-Path Only"    -> use_decomposition = False
    # "Decomposition Only" -> n_samples = 1
    use_decomposition: bool = True

    # The prompt asks for 3 to 5 subqueries (B.1/B.2). This is a cost guard for
    # when a model ignores that, not a target: each subquery is one extra call.
    max_subqueries: int = 8

    # ------------------------------------------------------------- voting ---
    # "canonical" is the paper's section 4.4 prose: canonicalise, group, take the
    # most frequent, break ties on fewer steps.
    # "symbolic" clusters by scorer.equal_program (sympy) instead. Not the
    # paper's method; measure, don't assume -- and see the note in program.py
    # about how little it actually merges.
    #
    # KNOWN GAP: the paper's Figure 1 labels this stage "Top result voting" and
    # its worked example only reaches the depicted 2-vs-1 majority if candidates
    # are grouped by *executed value* rather than by program structure. Neither
    # mode here does that yet. See the "Pending" section of the README.
    vote_mode: VoteMode = "canonical"

    # Candidates that cannot be parsed, transpiled, or executed are dropped
    # before the vote. Not stated in the paper, but voting over programs that
    # cannot run is strictly worse than voting over the ones that can.
    drop_invalid_candidates: bool = True

    # Drop candidates whose table_*(row, none) names a row that is not in this
    # sample's table -- such a program always executes to "n/a".
    validate_row_labels: bool = True

    # ---------------------------------------------------------- fallbacks ---
    # Not in the paper. An empty prediction scores 0 on both metrics with
    # certainty, so every degradation path still emits something.
    use_direct_prompt_fallback: bool = True

    # --------------------------------------------------------- local GPU only --
    # LocalBackend's n-sampling starts optimistically at batch_size=n and only
    # backs off after an actual CUDA OOM (see backends.py's adaptive retry).
    # That is correct but wasteful on a GPU already known to be tight (a
    # Kaggle T4 OOM'd at n=15 on a ~4-5k token planner prompt, confirmed by a
    # real run) -- every sample pays for at least one failed generate() call
    # before finding a size that fits. Set this to skip straight to a smaller
    # starting batch instead of discovering it by OOMing every time. None (the
    # default) keeps the optimistic behaviour, correct for a GPU with more
    # headroom (e.g. Modal's A100-80GB).
    local_max_batch_size: Optional[int] = None

    # ------------------------------------------------ concurrency / retries --
    max_workers_subquery: int = 5    # fan-out within one sample (A_sq)
    max_workers_sample: int = 15     # client-side n-sampling fan-out
    max_workers_dataset: int = 4     # samples processed concurrently
    max_retries: int = 5
    retry_base_delay: float = 2.0

    # Server-side `n>1` in a single request is much cheaper (prompt billed once)
    # but many proxy endpoints silently return one choice. None = probe once at
    # startup and cache the answer; True/False forces it.
    use_server_side_n: Optional[bool] = None

    # --------------------------------------------------------- rate limits --
    # None disables throttling. Set both for a rate-limited endpoint, e.g. FPT
    # AI Factory's GLM-5.2 (rpm=50, tpm=100_000) -- see
    # notebooks/vinumqa/few-shot/vsf-few-shot-vinumqa-glm5.2.ipynb.
    rpm_limit: Optional[int] = None
    tpm_limit: Optional[int] = None

    # ------------------------------------------------------------ tracing ---
    # Keep every candidate, not just the winner. Required to compute oracle@n,
    # which is what separates the paper's two failure modes (section 5.5.2):
    # oracle@n - PA is voting loss, 1 - oracle@n is base-model reasoning loss.
    # It is also what makes re-voting under a different `vote_mode` possible
    # offline, with no further API calls.
    keep_all_candidates: bool = True


@dataclass
class RunConfig:
    """Batch-run settings: what to score, where to checkpoint."""

    dataset_path: str = "datasets/ViNumQA/origin/test.json"
    output_dir: str = "notebooks/vinumqa/graph-agent/outputs"
    run_name: str = "mpr-agent"
    limit: Optional[int] = None      # smoke run: set to 30
    checkpoint_every: int = 1        # per-sample; a run is expensive to redo
    resume: bool = True
    agent: AgentConfig = field(default_factory=AgentConfig)
