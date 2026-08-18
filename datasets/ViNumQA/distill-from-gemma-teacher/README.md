# Independent-solve distillation from gemma-4-31B-it — current pipeline

Teacher `gemma-4-31B-it` sees **only context + question** (no gold shown)
and derives the program itself. Kept if the teacher's program matches gold
under PA. See `notebooks/vinumqa/distill-reasoning-trace/README.md` for the
prompt design and `notebooks/vinumqa/sft-w-reasoning-trace-distill/README.md`
for which SFT row (root README Table 2) each variant backs.

Two independent axes, giving 4 variants:

| | `vie/` | `eng/` |
|---|---|---|
| **`pa-match-only/`** | Vietnamese trace, strict PA filter | English trace, strict PA filter — used for `SFT (w ENG trace; PA match only)` |
| **`pa-and-partial/`** | Vietnamese trace, PA + numeric-surface-form partial match | English trace, PA + partial match |

Each leaf has `train.json` / `valid.json`. `pa-match-only/eng/` is the base
this repo's SFT-with-reasoning and STaNR experiments build on (see
`../self-distill/README.md` and `../../FinQA/self-distill/README.md`).
