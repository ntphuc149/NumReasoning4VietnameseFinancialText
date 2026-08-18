# FinQA — original splits + ViNumQA-schema reformat

Original English FinQA (Chen et al., 2021) — the source dataset ViNumQA's
format is based on. **Used here purely as a training-set augmentation for
ViNumQA** (root README Table 3), never for validation or test — ViNumQA's
own `valid.json`/`test.json` are used unchanged for every experiment,
whether the training set is ViNumQA-only or ViNumQA+FinQA combined.

| File | Contents |
|---|---|
| `train.json`, `dev.json`, `test.json`, `private_test.json` | FinQA's own splits, unmodified, English schema (`table_ori`, `table_retrieved*`, etc. — see `../formating_datasets.ipynb`) |
| `finqa.json` | `train.json` + `dev.json` + `test.json` **concatenated** and reformatted to ViNumQA's schema (`{pre_text, table, post_text, id, qa: {question, program, exe_ans}}`) — the retriever-stage fields, `qa.program_re`, and other FinQA-only fields are dropped (see `../formating_datasets.ipynb` for why). This is the file that gets folded into ViNumQA's train set for the combined-training experiments; `private_test.json` is excluded (no gold). |

FinQA's `train`+`dev`+`test` are all merged into `finqa.json` because this
project only borrows FinQA as extra *training* signal — there is no separate
FinQA-only eval anywhere downstream.
