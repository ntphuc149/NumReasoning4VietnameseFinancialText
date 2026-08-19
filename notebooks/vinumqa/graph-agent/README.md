# MPR-Agent — graph-based multi-agent pipeline

An implementation of Nguyen, Ha, Le & Vu, *"A Graph-Based Agent Approach to
Numerical Reasoning Question Answering"* (VLSP 2025,
[aclanthology.org/2025.vlsp-1.29](https://aclanthology.org/2025.vlsp-1.29/)) —
the system that **won Subtask 2 of this shared task with EA 84.00%**.

```
graph-agent/
├── agentic/          the pipeline (9 modules)
├── tests/            97 tests, no API calls, no GPU
├── mpr-agent.ipynb   driver: config -> full 497-sample run -> PA/EA + error analysis
└── outputs/          traces, per-sample CSV, summaries (gitignored)
```

The notebook is deliberately minimal — one config cell, one cell that runs the
full `test.json` and prints PA/EA unrounded, one error-analysis cell. The
smoke-test / baseline-comparison / ablation / prompt-fidelity-A/B cells that
produced the numbers in this README were run once, by hand, and are not kept
as live notebook cells — see "Ablation & prompt-fidelity results" below for
what they found and how to reproduce them if needed.

## Why this sits apart from the rest of `notebooks/vinumqa/`

Every other method in this repo is single-pass: one model, one prompt, one
program. 0/1/few-shot prompt it, SFT fine-tunes it, CoNR/STaNR improve its
training labels, GRPO optimises its policy.

This one **trains nothing**. It is inference-only and changes only *how the
model is asked*. On the paper's own baseline comparison, that took a Qwen3-32B
from EA 47.89 → 81.29 and PA 40.24 → 75.25. It is therefore orthogonal to
everything else here, and stackable on it — nothing stops the planner node
pointing at an SFT or STaNR checkpoint instead of an API model.

---

## The two graphs

### 1. The pipeline DAG — four agent nodes

```
q, C ──▶ [1] SubqueryGenerator   G_sq(q, C)            SQ = {sq_1..sq_k}, k∈[3,5]
                    │ fan-out, one independent call per subquery
         [2] SubqueryAnswerer    A_sq(sq_j, C)         V  = {v_1..v_k}
                    │ fan-in
         [3] Planner             P_n-sample(V,C,q,T)   n = 15 candidate plans
                    │
         [4] EquationExtractor   canonicalise → vote → p* → Execute → a*
```

| Node | Paper |
|---|---|
| Subquery Generator | §4.1, eq. (1), prompt B.1/B.2 |
| Subquery Answerer | §4.2, eq. (2), prompt B.3/B.4 |
| Plan and Scheduler | §4.3, eq. (3), prompts B.5–B.12 |
| Equation Extractor | §4.4, eqs. (4), (5), Figure 1 |

`agents.AgentGraph` resolves execution order from declared dependencies rather
than hard-wiring the chain, so the paper's ablations are configuration changes
rather than a second code path.

### 2. The plan DAG — inside each candidate

The planner prompt asks for *"maximum parallelization"*, so a plan is not a list
but a computation DAG: independent actions share a topological level.
`program.PlanGraph.levels()` recovers that structure and
`program.parallelism()` reports the mean actions per level — i.e. whether the
model actually complied.

## The three ideas that make it work

1. **Separate *finding numbers* from *doing arithmetic*** (§4.1). Long financial
   documents cause attention drift; asking one model to locate figures *and*
   chain multi-step arithmetic fails at both. The prompt's three `DO NOT` rules
   (no comparisons, no calculations, no final answers) enforce the boundary.
2. **Never commit to one reasoning path** (§4.3). Sample n=15 plans
   independently at temperature 0.6 instead of greedily decoding one.
3. **Consensus instead of a critic** (§4.4). No judge model, no labels:
   canonicalise, cluster, take the largest cluster, break ties on fewer steps.

The paper's own ablation (Table 4, public test):

| Configuration | 8B EA / PA | 32B EA / PA |
|---|---|---|
| Full MPR-Agent | 78.47 / 71.83 | 81.29 / 75.25 |
| Decomposition only (n=1) | 72.84 / 62.58 | 79.48 / 66.80 |
| Multi-path only (no decomposition) | 78.38 / 70.91 | 80.91 / 74.65 |
| Direct-prompt baseline | 41.05 / 32.60 | 47.89 / 40.24 |

Read carefully: **n-sampling carries most of the gain** in their setup.
Removing decomposition costs ~0.1–0.4 EA; removing n-sampling costs 5.6 EA /
9.3 PA at 8B. But the largest gap by far is *having the pipeline at all*
(+30–37 points over direct prompting). Appendix A: EA peaks at n=10, PA at
n=15 — hence n=15, since PA is the primary metric.

---

## The module map

| Module | Job |
|---|---|
| `config.py` | `AgentConfig` / `RunConfig`; every knob annotated "paper's" or "ours" |
| `scoring.py` | The one bridge to `notebooks/evaluate/scorer.py` — loaded by path, never copied |
| `prompts.py` | Appendix B verbatim (VI + EN), opt-in patches, fallback prompt |
| `llm.py` | Context formatting (`C`) + OpenAI-compatible client, rate limiter, `n` probe |
| `backends.py` | Routes each `model_*` name to local `transformers.generate()` or the API client |
| `program.py` | plan DSL → `PlanGraph` → ViNumQA program → vote |
| `agents.py` | `AgentState`, the four nodes, the pipeline graph |
| `runner.py` | Batch run, checkpoint/resume, scoring, oracle@n, offline re-vote |

---

## Running on any of this repo's eleven baseline models

The paper runs one model (Qwen3-8B / Qwen3-32B) through one transport it
controls end to end. This repo's own baseline suite is eleven models across
two very different transports — three it fine-tunes and runs locally
(`Qwen3-4B`, `qwen3-4b-thinking`, `Gemma3-4B`, loaded the same way
`notebooks/vinumqa/0-shot/vsf-vinumqa-0-shot-{qwen3-4b,gemma3-4b}.ipynb`
already do, with plain `transformers`, no Unsloth — that wrapper is for
training, and this package never trains anything) and eight it only ever
reaches through the hosted OpenAI-compatible endpoint already configured in
`.env` (`gemma-3-27b-it`, `gemma-4-31B-it`, `gpt-oss-20b`, `gpt-oss-120b`,
`Llama-3.3-70B-Instruct`, `DeepSeek-V4-Flash`, `GLM-5.2`, `gpt-5-nano`).

`backends.MultiModelClient` is what `Runner` actually constructs. It looks at
the `model` argument on every call and routes to whichever backend serves that
name — `backends.LocalBackend` (local) or `llm.LLMClient` (API), both built
lazily so a local-only run never needs `API_KEY`/`BASE_URL` and an API-only
run never touches a GPU. Every node in `agents.py` still just calls
`self.client.complete(...)` / `self.client.sample_n(...)`; nothing there, or
in `runner.py` beyond the one line that constructs the client, changed.

```python
MODEL = "Qwen3-4B"                    # -> LocalBackend, needs a GPU in this kernel
MODEL = "gemma-4-31B-it"              # -> LLMClient (API), needs .env
agent_config = AgentConfig(
    model_subquery_gen=MODEL, model_subquery_ans=MODEL,
    model_planner=MODEL, model_fallback=MODEL,
)
```

Nothing stops the four `model_*` fields naming different models — a small
local model on the two cheap extraction nodes and a large API model on the
planner, say. `describe_backend(name)` reports which way any given name
routes; `backends.MODEL_REGISTRY` is the three local names, `backends.
API_MODELS` documents (but does not gate) the other eight — a model the
endpoint serves under a name not in that list still falls through to the API
backend rather than erroring.

**Reasoning-model token budgets (API path)**: `DeepSeek-V4-Flash` and `GLM-5.2`
are confirmed reasoning models -- not name-guessed, established by this repo's
own baseline notebooks against the real endpoint
(`notebooks/vinumqa/0-shot/vsf-0-shot-vinumqa-deepseek-v4-flash.ipynb` forces
this for DeepSeek-V4-Flash after a real run crashed with `content=None` at
sample 483; `notebooks/vinumqa/few-shot/vsf-few-shot-vinumqa-glm5.2-rate-
check.ipynb` dumped a raw response and found a populated `reasoning_content`
field for GLM-5.2). Measured directly against this package's own planner
prompt: DeepSeek-V4-Flash spent 728 of a 768-token budget on hidden
`reasoning_content`, leaving 69 characters for the actual plan -- one harder
question away from empty. `llm.py` handles this two ways, so nothing in
`agentic/config.py`'s per-node `max_tokens_*` needs to change to use either
model: `is_reasoning_model()` checks `KNOWN_REASONING_MODELS` (this confirmed
set) before falling back to a name-keyword heuristic for unlisted models, and
`_call()` (a) starts a reasoning model's first request at
`REASONING_MODEL_MIN_TOKENS=2048` regardless of the caller's smaller default,
and (b) doubles the budget (capped at 4x) if a response still comes back with
non-empty `reasoning_content` but empty `content` -- the exact signature of
"ran out of room mid-thought," distinct from an empty response for some other
reason, which is not retried at a larger budget.

GLM-5.2 also has a **known rate limit** on this endpoint (measured in the
rate-check notebook above): RPM=50, TPM=100,000, with TPM binding first. Not
auto-applied here (`AgentConfig.rpm_limit`/`tpm_limit` default to `None` --
this package's own request pattern and concurrency differ from that
notebook's, so the same numbers are a starting point to set explicitly, not
assumed correct unmeasured for this pipeline):
`AgentConfig(rpm_limit=50, tpm_limit=100_000)`.

**LocalBackend specifics** (see the class docstring for the full reasoning):
`num_return_sequences=n` does the paper's n-sampling in one `generate()` call
rather than n separate ones, matching the eval notebooks' own choice; a
`qwen3-4b-thinking` output is sliced at its own final `</think>` token, the
same slicing and the same `QWEN3_THINK_END_TOKEN_ID` constant those notebooks
already use; and every actual `generate()` call is serialised behind a lock,
because a single GPU cannot run two at once and `agents.py` fans out over
subqueries and over n-samples with a `ThreadPoolExecutor` — correct for an API
backend, and would corrupt a local model's forward pass if not serialised
here. `AgentConfig.max_workers_*` accordingly has no effect when every
`model_*` names a local model; that is the expected tradeoff for one GPU, not
a bug.

Loads in whichever dtype this GPU actually has bf16 tensor cores for
(compute capability >= 8.0, i.e. Ampere+), not `torch_dtype="auto"` (the
checkpoint's own preferred dtype) and not `torch.cuda.is_bf16_supported()`
(measured on a real T4: returns `True` there too -- it only checks that bf16
ops run without erroring via a software fallback, not that they run fast).
Qwen3 checkpoints default to bfloat16, fine on Ampere+ (Modal's A100) but
Turing-generation GPUs (Kaggle's T4, compute capability 7.5) have no native
bf16 tensor cores -- measured cause of a real run being ~20-30 min/sample
instead of the expected tens of seconds.

`_generate()` frees the raw `generate()` output (`del out, gen_only;
torch.cuda.empty_cache()`) after every single call, not only on the OOM
path. A 497-sample run makes hundreds of these calls back to back in one
long-lived process (n=15 per sample at `batch_size=1` is 15 of them per
sample alone) -- measured on a real run: without this, GPU memory climbed
call over call until a *later* sample OOM'd at `batch_size=1` (previously
assumed always safe, since it is the smallest size there is) even though
earlier samples at that same batch size had generated fine.

`num_return_sequences=n` needs every sequence's KV cache in VRAM at once, and
that scales with both `n` and prompt length — a real Kaggle T4 (14.56 GB) run
OOM'd at `n=15` on the planner's ~3.9k token prompt, and follow-up runs showed
`n=4` and `n=2` OOM'ing too — only `n=1` actually fits at that prompt length
on this GPU. `LocalBackend` retries automatically at half the batch size on
OOM (down to 1), so a run always finishes, but each halving costs a wasted
`generate()` call first. On a GPU already known to be tight, set
`AgentConfig.local_max_batch_size` to skip straight to a smaller starting
batch instead — `mpr-agent.ipynb`'s Kaggle cells set this to `1`, matching
what was actually measured to fit. Leave it `None` (the default) on a roomier
GPU, e.g. Modal's A100-80GB.

**Dual-GPU (Kaggle "GPU T4 x2")**: `mpr-agent.ipynb` has an optional section
that runs two independent subprocesses, one pinned to each physical GPU via
`CUDA_VISIBLE_DEVICES`, each processing half of `test.json` through its own
`Runner` (so each half also gets its own checkpoint/resume, independently of
the other GPU), then merges both halves' scored CSVs before printing PA/EA.
Same subprocess-per-GPU pattern this repo's
`sft-w-reasoning-trace-distill/qwen3-4b-eval-only-dual-gpu.ipynb` already
uses. Roughly halves wall-clock. Controlled by a single `RUN_DUAL_GPU` flag in
the config cell, not by skipping cells by hand — the single-GPU cell and the
parallel section each check it and no-op if it's not their turn, so "Run All"
picks exactly one path instead of running the full single-GPU pass on 1 idle
GPU first and only then reaching the parallel section (the failure mode this
guards against, found on a real run).

**`qwen3-4b-thinking` on Kaggle: use `mpr-agent-qwen3-4b-thinking-kaggle.ipynb`
instead, not `mpr-agent.ipynb`'s local/dual-GPU path above.** Same `agentic/`
package underneath (unmodified) and a `Backend`-protocol client passed to
`Runner(client=...)`, but the transport is deliberately different for this one
model:

- **Loads with Unsloth in 4-bit** (`FastLanguageModel.from_pretrained(...,
  load_in_4bit=True)`), the same call this repo's own
  `qwen3-4b-thinking-2507-eval-only.ipynb`/`...-stf-w-reasoning-trace.ipynb`
  already use for this exact checkpoint on Kaggle -- not `LocalBackend`'s
  plain `transformers` fp16 load. ~3.2 GiB of weights instead of ~8 GiB frees
  enough VRAM to sample in chunks (`SAMPLE_CHUNK`, starts at 3, steps down by
  one -- not by half -- on OOM and stays there) instead of `LocalBackend`'s
  forced `batch_size=1`.
- **`THINK_HEADROOM` added on top of every one of the paper's four token
  budgets**, not just the planner's. A thinking model spends its first
  1-2k tokens inside `<think>` before ever writing an answer; a paper-sized
  budget alone is spent entirely on the trace and the generation is cut off
  before the plan exists, which scores 0 exactly like a wrong answer would
  but is actually a budget bug.
- **Clears the OOM's own traceback before reclaiming VRAM**
  (`exc.__traceback__ = None`) -- a live traceback holds one frame per level
  of `generate()`, and each frame holds its tensors, so `empty_cache()` taken
  while the exception is still bound frees nothing.
- **Single GPU only** (no dual-GPU section) -- the four points above address
  the actual bottleneck (sequential, unbatched generation from being forced to
  `batch_size=1`), which running two of the same slow thing in parallel does
  not.

Ships with its own smoke-run cell that measures real s/sample and projects the
full 497-sample wall-clock **before** committing a session to it, and
session-to-session resume for the runs this pipeline's token cost genuinely
needs (multiple 12h Kaggle sessions even after the fixes above). Was written
and reasoned through without GPU access, so — same as every other config on
this page — treat its first real run as the actual verification, not this
description.

**Comparability**: this notebook's own default was `use_decomposition=True`
(paper-faithful); changed to `use_decomposition=False` to match every other
model's row in the table below and in the root README, since that is this
repo's measured-best, actually-used default, not the paper's own choice.
Flip it back only to deliberately reproduce the paper-faithful row, and label
that result as such if you report it.

---

## Ablation & prompt-fidelity results

Measured on `DeepSeek-V4-Flash`, full `test.json` (497 samples), 2026-08-18.
Each row is a separate ~2-3h full run (~8h for all four); reproduce with:

```python
from dataclasses import replace
paper_faithful = replace(agent_config, use_decomposition=True)       # "full MPR-Agent"
decomposition_only = replace(paper_faithful, n_samples=1)             # nodes [1]+[2], single plan
multi_path_only = replace(agent_config, use_decomposition=False)      # nodes [3]+[4] -- this repo's default
with_prompt_patch = replace(paper_faithful, use_prompt_ext=True)      # paper-faithful nodes + prompt fix
# Runner(RunConfig(..., run_name=f"mpr-agent-{MODEL}-<name>", agent=<one of the above>)).run(show_progress=True)
```

| Configuration | PA | EA | s/sample |
|---|---:|---:|---:|
| full MPR-Agent (paper-faithful: decompose + n=15 vote) | 0.7505 | 0.8189 | 22.9 |
| decomposition-only (decompose, n=1, no vote) | 0.7485 | 0.8330 | 13.7 |
| **multi-path-only (no decompose, n=15 vote) — this repo's default** | **0.7787** | **0.8370** | **12.4** |
| + percent-as-decimal + placeholder-clarification patch (paper-faithful nodes, `use_prompt_ext=True`) | 0.7686 | 0.8370 | 20.0 |

**Reading it**: the paper's own ablation (Table 4, above) found decomposition
helps a *little* on Qwen3 (removing it costs ~0.1–0.4 EA). Here, on
DeepSeek-V4-Flash, it's the opposite and the effect is not small: dropping
decomposition entirely *gains* +2.8 PA / +1.8 EA over the paper-faithful
pipeline, while also being the fastest and cheapest row (skips 2 of 4 nodes'
API calls). The prompt patch, tested against the paper-faithful (decomposing)
pipeline, does help there (+1.8 PA) — but not enough to catch up to simply
dropping decomposition, so it is not layered on top of `multi-path-only` here;
that combination has not been measured.

**Working hypothesis for the reversal** (not independently verified): the
subquery-answerer node can extract a wrong fact from the table/text and hand
it to the planner as confident "additional context," and the pipeline's own
high `mean_consensus` (~0.91–0.94 throughout) suggests the model commits to
whatever it is given rather than double-checking it — so a bad extraction
doesn't just fail to help, it actively steers all 15 planner samples toward
the same wrong answer. Without decomposition, the planner reasons over the
raw context directly and reasoning models may simply not need the "divide and
conquer" scaffolding a weaker instruct model would.

**Error breakdown on the kept `multi-path-only` result** (`oracle@15`
PA=0.8410, EA=0.8732 — see "oracle@n" below for what this diagnostic means):
6.24% of samples had a correct candidate that voting picked wrong (*heuristic
selection error*); 15.90% never generated a correct candidate in any of the 15
samples (*systematic reasoning error*, voting cannot fix this). Of the wrong
final answers, 48.18% were **unanimous** — all 15 candidates agreed on the
same wrong answer — the signature of a systematic model limitation rather
than voting noise. `fallback_rate` and `empty_rate` were both 0.0000.

**Scope of this finding**: measured on one model (DeepSeek-V4-Flash) only.
Whether it holds for the other ten baseline models — including the three this
repo actually fine-tunes — is untested.

---

## The gap the paper does not have to solve: plan DSL → ViNumQA program

The planner speaks the paper's plan DSL. `scorer.py` grades ViNumQA program
strings. These are not the same language, the paper never documents its own
conversion, and `program.py` PART 2 is the reconstruction — the single
highest-risk code here.

| | Paper plan DSL | ViNumQA program |
|---|---|---|
| Syntax | `1. subtract(a='21', b='47')` | `subtract(21, 47)` |
| References | `$1` — action id, 1-based | `#0` — step position, 0-based |
| Table ops | `table_max(row_identifier='X')` — **1 arg** | `table_max(X, none)` — **2 args** |
| Terminator | `join()` + `<END_OF_PLAN>` | none |
| Operators | 8 + `join` | 10 (adds `exp`, `greater`) |

Seven rules, each justified in the `program.py` PART 2 header. The three that
matter most:

- **`table_*(row, none)`.** Verified on the data: all 63 `table_*` calls in the
  gold test programs pass exactly `none` as the second argument. Row labels are
  emitted **unquoted** — real labels contain brackets (`ROE (%)`, `P/E (x)`,
  `EPS (VND)`) which `scorer.py`'s bracket-aware tokeniser handles but quotes
  would break.
- **Literals pass through verbatim.** `scorer.py`'s PA admits only literals that
  occur in the gold program, so any helpful-looking renormalisation is a direct
  PA loss. Gold legitimately contains `100` (e.g. `subtract(96.67, 100),
  divide(#0, 100)` for a base-100 index), so `100` cannot be special-cased.
- **Dead branches are pruned.** A branch the answer does not use cannot change
  EA (the last step is the result) but *can* cost PA, because `equal_program`
  walks every step of the prediction and rejects one whose literal is absent
  from gold.

### On `exp` and `greater`

Counted over the real splits: `exp` appears **0 times** in train and test;
`greater` appears **once** in test, **never** in train. The paper's eight tools
already cover 496/497 test samples, so they stay as-is; the two extras live
behind `use_prompt_ext`.

---

## Fidelity to the paper — what matches and what does not

### Matches

4-node architecture and node roles; all Appendix B prompts verbatim (including
the paper's own typo); `n=15`, `T=0.6`, `top_p=0.95`, `top_k=20`; planner
receives `(V, C, q, T)` per eq. (3); tie-break on fewer steps; inference-only;
both graphs explicit.

One field is **not** verbatim and is marked so in `prompts.py`:
`planner_query_block`. The paper prints B.7/B.9/B.11 as static instruction text
but never shows the block carrying the question, context, and subquery answers.
It is reconstructed from B.9's reference to a section headed "BỐI CẢNH BỔ SUNG
TỪ TRUY VẤN CON" and the `Truy vấn:` / `Bối cảnh:` / `Kế hoạch:` shape of
B.11's worked examples.

### Known gap — the level at which votes are counted

§4.4's prose describes canonicalising **programs** and grouping them. That is
what is implemented.

But Figure 1 labels the stage **"Top result voting"** and draws each candidate
beside its executed value:

| Candidate in Figure 1 | Value |
|---|---|
| `add(19038.80, 9445.09), add(#0, 6286.89)` | 34770.78 |
| `add(19038.80, 6286.89), add(#0, 9445.09)` | 34770.78 |
| `add(19038.80, 31434.47), add(#0, 6286.89)` | 56760.16 |

The first two are **structurally different programs with the same value**, and
the figure shows 34770.78 winning **2–1**. Under program-structure voting they
land in different clusters, so the count is 1–1–1 and the depicted majority
never forms — the paper's own example is not reproducible under its own prose.
The mechanism it draws is voting over **executed results**.

Circumstantial support: the paper's EA (84.00) exceeds its PA (74.07) by ten
points, and §5.5 says *"our agent prioritizes functional correctness over
stylistic conformity"* — the signature of result-level selection.

`test_figure_1_candidates_are_not_merged_by_structure_voting` pins the current
behaviour so that adding a result-level mode later reads as a deliberate change.
**Pending**: `vote_mode="result"`. Because every candidate is kept on disk with
its executed value, this can be evaluated on existing runs via
`runner.revote()` with no further API calls.

### On `vote_mode="symbolic"`

Less than it sounds, and the code says so. `equal_program` admits only literals
present in the program it is compared against, so it merges **algebraic
rearrangements over the same literals** (`multiply(add(a,b), c)` with
`add(multiply(a,c), multiply(b,c))`) — nothing more. It does **not** merge the
Example 5.1 pair, which introduces new literals. That is a prompt problem
(`use_prompt_ext`), not a voting problem.

---

## Decisions that are ours, not the paper's

Each is a flag, so its effect is measurable rather than baked in.

| Flag | Default | Why |
|---|---|---|
| `drop_invalid_candidates` | `True` | Voting over programs that cannot execute lets a malformed cluster outvote a working one |
| `validate_row_labels` | `True` | A `table_*` naming an absent row always executes to `n/a`; catching it names the failure instead of hiding it in a zero |
| `use_direct_prompt_fallback` | `True` | An empty prediction is a guaranteed zero on both metrics. Every use is recorded, so `fallback_rate` is reported next to PA/EA |
| `keep_all_candidates` | `True` | Required for oracle@n, and for offline re-voting |
| `use_prompt_ext` | `False` | Keeps the headline run prompt-faithful |
| `temperature_planner` | `None` | `None` = the paper's shared 0.6 |

### oracle@n — the diagnostic the paper needs but does not report

`Runner.oracle()` scores the *best* of the n candidates per sample, splitting
the paper's two failure modes (§5.5.2) apart:

- `oracle_pa − program_accuracy` → what majority voting **threw away**
  (their "heuristic selection error")
- `1 − oracle_pa` → what the base model **never generated**
  (their "systematic reasoning error")

---

## Running

`mpr-agent.ipynb` is the driver: a config cell (pick `MODEL`, everything else
defaults to the best-measured setup) and one cell that runs the full
497-sample `test.json` and prints PA/EA, followed by the error-analysis cell.
Needs `API_KEY`/`BASE_URL` in `.env` (API model) or a GPU in the kernel (local
model). Run the tests first — they cover the transpiler against every gold
program in all three splits and cost nothing:

```bash
.venv/Scripts/python -m pytest notebooks/vinumqa/graph-agent/tests -q
```

Equivalent, outside the notebook:

```python
import sys; sys.path.insert(0, "notebooks/vinumqa/graph-agent")
from dotenv import load_dotenv; load_dotenv()
from agentic import AgentConfig, RunConfig, Runner

runner = Runner(RunConfig(run_name="mpr-agent", agent=AgentConfig(use_decomposition=False)))
df = runner.run()                       # checkpoints per sample, resumable
scored, summary = runner.score(df)      # PA/EA + oracle@n
runner.save(scored, summary)
```

To reproduce the paper's Table 4 ablations (the direct-prompt baseline row
already exists in this repo's 0-shot/few-shot notebooks, so it does not need
re-running) or the "Ablation & prompt-fidelity results" table above:

```python
AgentConfig(n_samples=1)              # "Decomposition Only"
AgentConfig(use_decomposition=False)  # "Multi-Path Only" -- this repo's default
AgentConfig(use_prompt_ext=True)      # + percent-as-decimal + placeholder-clarification patch
```

Cost is dominated by the planner: with server-side `n` supported by the
endpoint, roughly **4.5 requests and ~9k tokens per sample**; without it, about
20 requests per sample. `llm.py` probes for that once at startup and caches it.

---

## Two things to know before reading any number

**n-sampling may be doing nothing on your model.** Measured on
`gemma-4-31B-it` at the paper's `temperature = 0.6`: 26 of 30 smoke samples
produced a *single distinct plan* across all 15 generations, and `oracle@15`
equalled plain PA exactly. This was checked three ways on a real planner prompt
— this package's client, a raw server-side `n=15` call, and 15 separate
requests — all returned 1 distinct plan, while a control probe on an open-ended
prompt returned 9 distinct out of 15. Raising planner temperature to 0.9 and
1.2 changed the distinct-*plan* count slightly but not the distinct-*program*
count. Once decomposition has pinned the numbers down, the plan is effectively
determined. Check `trace["candidates"]`'s distinct `program` count per sample
(in a saved `*_traces.json`) before attributing any gain to multi-path
reasoning on a new model.

**Prompt fidelity costs PA.** Appendix B.9/B.10 says to strip a percent sign and
use `48.8`; ViNumQA gold writes such a rate as `0.488`. 34 of the 497 gold test
programs contain a `0.xx` literal of this kind, and following the prompt
literally passes EA but fails PA on them — the paper's own Example 5.1. See
"Ablation & prompt-fidelity results" above for the measured cost of the
one-bullet fix (`use_prompt_ext=True`).
