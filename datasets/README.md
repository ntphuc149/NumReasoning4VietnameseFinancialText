# Datasets

| Directory | Contents |
|---|---|
| `ViNumQA/` | The dataset this project is built on — VLSP 2025 NumQA shared task. Core splits plus every reasoning-trace-distillation variant. **See `ViNumQA/README.md`.** |
| `FinQA/` | Original English FinQA dataset (Chen et al., 2021), the source dataset ViNumQA's format is based on — used as a **training-set augmentation** for ViNumQA (root README Table 3), reformatted to ViNumQA's schema. Never used for validation/test. **See `FinQA/README.md`.** |

Both directories are organized by pipeline stage (`origin/` →
`distill-from-gemma-teacher/` → `self-distill/`), each with its own
`README.md` — start with the two top-level READMEs above, they map out
which subdirectory to read next for a given experiment.
