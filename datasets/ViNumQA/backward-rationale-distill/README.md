# Backward-rationalization distillation — superseded

**Not used in current experiments.** Kept for comparison only. See
`notebooks/vinumqa/distill-reasoning-trace/README.md` for the full pipeline
writeup and why it was replaced.

Teacher (`Qwen3-Next-80B-A3B-Thinking`) was shown the **gold program and
answer** and asked to write a trace that arrives at exactly that program.
Reached 98–99% exact-match, but reading the traces end-to-end later found
~1% leaked awareness of the answer despite denylist screening — the teacher
sometimes acknowledged following a program it had been handed rather than
deriving it. Superseded by the independent-solve pipeline
(`../distill-from-gemma-teacher/`).

| Directory | Contents |
|---|---|
| `vie-trace/` | `train.json` (2,905) / `valid.json` (571) — original Vietnamese traces |
| `vie2eng-translate/` | Same samples, trace machine-translated to English (`VietAI/envit5-translation`) — also abandoned in favor of distilling directly in English with the independent-solve pipeline |
