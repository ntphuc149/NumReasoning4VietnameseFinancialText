# stanr

Reasoning-trace distillation (CoNR) + iterative self-teaching fine-tuning
(STaNR) for Vietnamese financial numerical reasoning, as a runnable package
instead of one notebook per model. Clone the repo, install, run the script
below -- no notebook cells to edit by hand.

## Install

```bash
# distillation + data utilities only (no GPU needed)
pip install -e .

# + training (Unsloth/torch/trl -- needs a GPU)
pip install -e ".[train]"
```

## Usage (no code needed)

```bash
python scripts/run.py
```

Runs with defaults: Qwen3-4B, reuses the distilled data already shipped in
this repo, one self-teaching round. Every setting is a command-line flag
with a sensible default -- override only what you need, nothing to edit:

```bash
python scripts/run.py \
    --teacher_model_name "some-other-teacher" \
    --dataset datasets/ViNumQA/origin \
    --pretrained_model_name "unsloth/gemma-3-4b-it" \
    --run_distill \
    --loop 5
```

Run `python scripts/run.py --help` for the full flag list (data paths,
precision, batch size, learning rate, ...).

`--run_distill` is off by default, which skips distillation and trains
directly on `--train_distilled`/`--valid_distilled` (already present in this
repo). Passing `--run_distill` calls `--teacher_model_name` first and
regenerates those two files -- that needs `API_KEY` (and optionally
`BASE_URL`) set in a `.env` file at the repo root, the same convention every
API notebook in this repo already uses:

```
API_KEY=...
BASE_URL=https://mkp-api.fptcloud.com/v1
```

## Usage (as a library, for a custom script)

`scripts/run.py` is a thin wrapper around two classes you can also call
directly if you want a custom pipeline `config.yaml` doesn't cover:

```python
from stanr import ReasoningDistiller, STaNRTrainer

rd = ReasoningDistiller(
    model_name="gemma-4-31B-it", api_key="...",
    base_url="https://mkp-api.fptcloud.com/v1",   # any OpenAI-compatible endpoint
    input_dataset="datasets/train.json",
    output_distill_path="datasets/train_distilled.json",
    filter_with="pa",   # "pa" | "strict" | "union" -- see ReasoningDistiller's docstring
)
rd.run()

trainer = STaNRTrainer(
    input_train_raw_dataset="datasets/train.json",
    input_distilled_train_set="datasets/train_distilled.json",
    input_val_raw_dataset="datasets/valid.json",
    input_distilled_val_set="datasets/valid_distilled.json",
    input_test_raw_dataset="datasets/test.json",
    pretrained_model_name="unsloth/Qwen3-4B",   # or Gemma3-4B / qwen3-4b-thinking -- same class
    model_checkpoint_path="output/",
    precision="4-bit",   # "4-bit" (QLoRA) | "16-bit" (LoRA) | "full" (full fine-tune)
    loop=1,               # number of self-teaching rounds after round 0
)
history = trainer.train()
print(history)   # per-round {"round", "adapter_dir", "train_size", "newly_passed", "pa", "ea"}
```

## What each piece is responsible for

- **`stanr.prompts`** -- the CoNR prompt (distillation) and the SFT
  system/user prompt (training + inference), copied verbatim from the
  notebooks that validated them. Only the English-trace CoNR variant ships
  by default (the paper's own default); pass `conr_system_prompt=`/
  `conr_user_frame=` to `ReasoningDistiller` for a different one (e.g. the
  Vietnamese-trace variant).
- **`stanr.scoring`** -- the corrected FinQA/ViNumQA PA/EA protocol, vendored
  from `notebooks/evaluate/scorer.py` (this is the same file every notebook
  in the repo inlines, for the same reason: remote sessions can't import
  from the repo).
- **`stanr.data`** -- dataset I/O, context formatting, and the id-based
  set operations STaNR's self-teaching loop needs (`samples_missing_from`,
  `with_reasoning_trace`, `extract_think_and_program`).
- **`stanr.distiller.ReasoningDistiller`** -- one API call per sample
  (parallel, retried, checkpointed/resumable), asking the teacher to derive
  `<think>` + `<program>` independently, no gold program in the prompt.
- **`stanr.trainer.STaNRTrainer`** -- Unsloth `FastLanguageModel` + TRL
  `SFTTrainer` under the hood, generic over `pretrained_model_name` (works
  for any of the three student models used in the paper). GPU-heavy imports
  are deferred to inside its methods, so importing `stanr` and constructing
  either class works on a machine with no GPU/Unsloth installed -- only
  calling `.train()` needs them.

## Notes / current limitations of this first version

- `STaNRTrainer._generate` runs samples one at a time (`model.generate()`
  per sample), matching the self-distillation notebooks' correctness but not
  their Kaggle-specific dual-GPU/batched speed optimizations. Fine for
  correctness; a power user chasing wall-clock time on a specific rig should
  subclass and override `_generate`.
- The self-teaching filter (round 1+) is always PA-only (`scoring.equal_program`),
  independent of whatever `filter_with` round 0's `ReasoningDistiller` used --
  this matches every self-distillation notebook in this repo, not a
  configurable choice.
- `.train()` stops early (before `loop` rounds are exhausted) if a round
  finds no unresolved questions left, or if a round's self-teaching attempt
  yields zero newly-passed samples -- both are safeguards against wasted GPU
  time on a round that provably cannot change anything, not something the
  original notebooks needed to handle (they were run round-by-round by hand).
