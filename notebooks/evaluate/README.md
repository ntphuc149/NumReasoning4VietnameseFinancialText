# Shared evaluator

`scorer.py` is the **one** evaluator used by every prompting/SFT notebook in
this repo — a port of the FinQA official evaluation protocol (Chen et al.,
2021), adapted to reproduce ViNumQA's own gold labels exactly (five
corrections, documented in the module docstring: bracket-depth-aware
tokenization for row labels like `ROE (%)`, the `(3344)` accounting-negative
form, skipping unparseable table cells instead of voiding the whole row,
coercing `exe_ans` to float, and dropping a debug assert that aborted scoring
on rounding disagreements).

```python
from scorer import evaluate_dataframe

df_scored, summary = evaluate_dataframe(
    df,
    generated_col="generated_program",
    gold_program_col="program",
    gold_answer_col="answer",
    table_col="table_raw",   # raw list-of-lists table, needed for table_* row lookup
)
# summary == {"program_accuracy": ..., "execution_accuracy": ...}
```

- **Program Accuracy (PA)** — `equal_program`: sympy-based symbolic
  equivalence between gold and predicted programs. A prediction may reorder
  or restructure the arithmetic, but may only use literals that appear in
  the gold program.
- **Execution Accuracy (EA)** — `eval_program`: executes the predicted
  program against the raw table and compares to `exe_ans` exactly (rounded
  to 5 decimals).

**Do not re-implement this parser inline in a new notebook.** Import or copy
`scorer.py` verbatim — every notebook that inlines its own copy does so
because of offline execution environments (Kaggle/Modal with no local
package install), not because the logic differs.

`evaluate.py` and `example_predictions.json` / `test.json` here are FinQA's
own original reference scripts/fixtures, kept for cross-checking `scorer.py`
against the unmodified upstream protocol — not used by any notebook directly.
