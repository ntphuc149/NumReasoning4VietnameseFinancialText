# Reasoning-trace machine translation — abandoned

`translate-reasoning-trace-envit5.ipynb`: translates the Vietnamese
backward-rationalization traces
(`datasets/ViNumQA/train_with_reasoning_trace.json`) to English via
`VietAI/envit5-translation`, with sentence-level translation and a
number-recovery step (splicing original numeric tokens back into the
translation) to avoid corrupting numbers embedded in each trace. Produced
`train_with_reasoning_trace_en.json` / `valid_with_reasoning_trace_en.json`.

**Superseded** by direct English distillation from the gemma-4-31B-it
teacher (`../distill-reasoning-trace/gemma-4-31b-conr-trace-gen-independent-solve-en.ipynb`),
which distills the trace in English directly rather than translating a
Vietnamese trace after the fact — cleaner, and avoids compounding any MT
translation errors on top of the trace's own reasoning. Kept for reference;
not used for new experiments.
