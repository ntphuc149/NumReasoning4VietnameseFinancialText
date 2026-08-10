# STaNR self-distillation — ViNumQA + FinQA combined training

Root README Table 3 (`STaNR_v1` row). Same self-distillation idea as
`datasets/ViNumQA/self-distill/`, but the base training set the checkpoint
is fine-tuned on first is ViNumQA **+ FinQA** combined, not ViNumQA alone —
**read `datasets/ViNumQA/self-distill/README.md` first** if you haven't;
this doc only covers what's different here.

## Pipeline, in order

1. **Base training set**: `datasets/ViNumQA/origin/train.json` +
   `datasets/FinQA/origin/finqa.json` (FinQA's train+dev+test merged into
   ViNumQA's schema). Teacher `gemma-4-31B-it` attempts every sample
   (context+question only); PA-match-verified samples from *both* sources
   become the base of `train_qwen3-4b.json`
   (64.64% of the combined pool — root README Table 3's `train (%)` for the
   `SFT (w ENG trace; PA match only)` row).
2. **Validation** for this training run is `datasets/ViNumQA/origin/valid.json`
   only — FinQA has no held-out valid side in this setup (its train+dev+test
   are all folded into the *training* pool in step 1). Same teacher, same
   PA-match filter, applied only to ViNumQA's valid split
   (`datasets/ViNumQA/distill-from-gemma-teacher/pa-match-only/eng/valid.json` —
   this is the exact same file/count as Table 2's `SFT (w ENG trace; PA match
   only)` row's `val (%)`, since it's the same teacher run on the same
   ViNumQA valid split, independent of whether FinQA is added to training).
3. **Fine-tune** Qwen3-4B on the combined training set from step 1, eval
   against the valid set from step 2 → this is the `SFT (w ENG reasoning
   trace; distill - PA match only)` checkpoint in Table 3.
4. **Self-distillation (STaNR)**: use that checkpoint as a second-round
   teacher on the samples the *original* teacher failed in steps 1 and 2:
   - **Train side** — teacher-failed samples from **both** ViNumQA train and
     FinQA (since both fed into the same base training pool in step 1). Keep
     PA-verified self-distilled traces, merge with the base set →
     **`train_{model}.json`**.
   - **Valid side** — teacher-failed samples from **ViNumQA valid only**
     (FinQA has no valid side to fail on, per step 2). Keep PA-verified
     self-distilled traces, merge with the base valid set →
     **`valid_{model}.json`**.

## Files here

Naming convention matches `datasets/ViNumQA/self-distill/`:
`train_{model}.json` / `valid_{model}.json`, one pair per model that's been
through this pipeline.

| File | Contents |
|---|---|
| `train_qwen3-4b.json` | Training set: PA-verified teacher solutions (ViNumQA + FinQA) + self-distilled additions from Qwen3-4B's combined-trained checkpoint |
| `valid_qwen3-4b.json` | ViNumQA-only validation set, enriched the same way, using the checkpoint trained on the **combined** (ViNumQA+FinQA) data |

**Despite the identical filenames, this is not the same content as
`datasets/ViNumQA/self-distill/train_qwen3-4b.json` /
`valid_qwen3-4b.json`** — those are self-distilled from a checkpoint trained
on ViNumQA **alone**; the files here are from a checkpoint trained on
ViNumQA **+ FinQA combined**. The directory (`datasets/FinQA/self-distill/`
vs. `datasets/ViNumQA/self-distill/`) is what disambiguates them — always
check which directory a `train_{model}.json`/`valid_{model}.json` came from
before using it.

When adding a new model's pair here (e.g. Gemma3-4B, qwen3-4b-thinking),
follow the same steps 1–4 above with that model in place of Qwen3-4B, and
name the files `train_{model}.json` / `valid_{model}.json` to match.
