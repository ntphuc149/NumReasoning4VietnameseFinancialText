# FinQA dataset

Original English FinQA (Chen et al., 2021) — used here purely as a
**training-set augmentation** for ViNumQA (root README Table 3,
`SFT (... ; distill - PA match only)` and `STaNR_v1` rows trained on
ViNumQA+FinQA combined). Never used for validation or test — ViNumQA's own
`valid.json`/`test.json` (see `../ViNumQA/origin/`) are unchanged across
every experiment.

Organized the same way as `../ViNumQA/` — **each subdirectory has its own
`README.md`**:

```
origin/                      FinQA's own splits, plus a ViNumQA-schema reformat (finqa.json = train+dev+test merged)
distill-from-gemma-teacher/  finqa.json run through the same independent-solve distillation as ViNumQA (English, PA-match-only)
self-distill/                STaNR self-distillation using a checkpoint trained on ViNumQA+FinQA combined
```

`formating_datasets.ipynb` is the notebook that produces `origin/finqa.json`
from FinQA's raw splits (schema mapping, dropped fields, and why —
documented in the notebook itself).
