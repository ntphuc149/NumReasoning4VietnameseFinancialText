# FinQA distilled from gemma-4-31B-it

`finqa.json`: `../origin/finqa.json` run through the same independent-solve
distillation as ViNumQA (teacher `gemma-4-31B-it`, context+question only, no
gold shown), English trace, PA-match-only filter — the FinQA-side
counterpart of `datasets/ViNumQA/distill-from-gemma-teacher/pa-match-only/eng/train.json`.

No Vietnamese variant and no PA-partial-match variant here (unlike the
ViNumQA side) — combined-training experiments (root README Table 3) only
use the English, strict-PA-match setup, since prior ViNumQA-only results
already showed English trace beats Vietnamese and partial-match risks
training on a trace that doesn't match its own program's exact surface form.

No separate valid/test file: FinQA is folded entirely into the *training*
side of the combined experiments (see `../origin/README.md`), so there is
nothing to distill on the validation/test side here.
