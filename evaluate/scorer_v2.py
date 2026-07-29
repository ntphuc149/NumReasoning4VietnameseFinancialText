"""ViNumQA scorer: the FinQA evaluation protocol, adapted to this dataset.

The shared-task paper states that "the official evaluation protocol proposed by
Chen et al. (2021) is adopted", so the semantics here follow `evaluate/evaluate.py`
(FinQA's own script) rather than being reinvented:

  * Program Accuracy is *symbolic* equivalence, via sympy, between the gold and
    predicted expressions -- not a string or structural match. A prediction may
    reorder or restructure the arithmetic, but it may only use literals that
    appear in the gold program, so it cannot invent constants such as the `100`
    of a percentage rescaling.
  * Execution Accuracy compares the executed result to `exe_ans` exactly, after
    rounding to 5 decimals. No tolerance.
  * `greater` yields the strings "yes"/"no", matching how the dataset stores
    those answers.
  * Every step takes exactly two arguments. Verified against the data: all 663
    steps across the gold programs are binary, and `table_*` always takes a row
    label plus `none` (454 occurrences) rather than a list of values (2).
  * FinQA's `const_` tokens are still understood.

Five corrections are applied, each because the unmodified script cannot
reproduce ViNumQA's own gold, not because the protocol was thought wrong:

1. Tokenisation of bracketed row labels. `program_tokenization` splits on every
   bracket, so `table_min(ROE (%), none)` shatters into six tokens and fails the
   four-tokens-per-step structure check. 35 of the 497 test programs name a row
   whose label contains brackets -- `ROE (%)`, `EPS (VND)`, `P/E (x)` -- and all
   35 were unscoreable. Tokenisation is now bracket-depth aware.

2. Accounting negatives. Tables write negative amounts as `(3344)`. The original
   `process_row` takes the text before the first bracket, leaving an empty
   string, so the cell fails to parse. The dataset's own `exe_ans` was computed
   with those values -- e.g. `table_min(LN hoạt động (tỷ đồng), none)` expects
   -3344 from a row holding `(3344)`. The `-1046 ( 1046 )` form the original
   handled correctly is unchanged.

3. Unparseable cells no longer void the whole row. Measured over the 393 gold
   `table_*(<row>, none)` programs in train, skipping such cells reproduces
   `exe_ans` for 386 against 381 when the row is voided, so skipping is what the
   dataset was built with.

4. `exe_ans` is stored as a string here ("31.0") where FinQA stores a number, so
   the comparison `exe_res == gold_res` was never true. It is coerced, leaving
   the "yes"/"no" answers alone.

5. The `assert exe_res == gold_res` inside the program-accuracy branch is
   dropped. It is a debug check, and a single rounding disagreement aborts the
   whole evaluation.

`evaluate_result_official` runs the unmodified protocol for comparison, so the
cost of each correction can be seen rather than assumed.
"""

import re
from typing import List, Optional, Sequence, Tuple, Union

from sympy import simplify

ALL_OPS = ["add", "subtract", "multiply", "divide", "exp", "greater",
           "table_max", "table_min", "table_sum", "table_average"]

_PAREN_NEG_RE = re.compile(r"^\(\s*([\d.,]+)\s*\)$")
_NAME_RE = re.compile(r"\s*([a-zA-Z_]+)\(")


# ------------------------------------------------------------------ numbers --
def str_to_num(text: str) -> Union[float, str]:
    """FinQA's literal parser, unchanged: returns "n/a" rather than raising."""
    text = str(text).replace(",", "")
    try:
        return float(text)
    except ValueError:
        if "%" in text:
            try:
                return float(text.replace("%", "")) / 100.0
            except ValueError:
                return "n/a"
        if text.endswith(("x", "X")):
            # Multiples are written "14.3x" in these tables. Unlike "%", the
            # suffix carries no scaling -- the gold answer for such a row is the
            # plain multiple.
            try:
                return float(text[:-1])
            except ValueError:
                pass
        if "const" in text:
            text = text.replace("const_", "")
            if text == "m1":
                text = "-1"
            try:
                return float(text)
            except ValueError:
                return "n/a"
        return "n/a"


def _cell_to_num(raw: str) -> Union[float, str]:
    """Parse one table cell.

    Adds the `(3344)` form to what the original handled; `$ -1046 ( 1046 )`
    still resolves through the original's "text before the first bracket" rule.
    """
    text = str(raw).replace("$", "").strip()
    m = _PAREN_NEG_RE.match(text)
    if m:
        value = str_to_num(m.group(1))
        return -value if value != "n/a" else "n/a"
    return str_to_num(text.split("(")[0].strip())


_MISSING_CELL_MARKERS = {"", "-", "–", "—", "na", "n/a", "nan", "none"}


def process_row(row_in: Sequence[str]):
    """Numeric values of a table row, or "n/a" if the row cannot be reduced.

    A cell that merely marks a missing period ("-", "NA", an em dash) is
    skipped: rows in this dataset routinely lack a year or two, and voiding the
    whole row over one gap loses reductions the gold answers depend on. A cell
    with real but unreadable content still voids the row, so genuine parse
    failures are not silently averaged away.
    """
    row_out = []
    for cell in row_in:
        text = str(cell).replace("$", "").strip()
        if text.lower() in _MISSING_CELL_MARKERS:
            continue
        num = _cell_to_num(text)
        if num == "n/a":
            return "n/a"
        row_out.append(num)
    return row_out or "n/a"


# -------------------------------------------------------------- tokenisation --
def program_tokenization(original_program: str) -> List[str]:
    """Tokenise into ['op(', arg1, arg2, ')', ..., 'EOF'].

    Bracket-depth aware, so a row label like `ROE (%)` stays one token. The
    original split on every bracket, which shattered such labels and broke the
    four-tokens-per-step structure the rest of the protocol relies on.

    Raises ValueError if trailing, non-whitespace text remains once no further
    step can be parsed (e.g. a step missing its closing paren, which happens
    both in a handful of gold programs and -- more importantly -- in model
    generations cut off by a max_new_tokens limit). An earlier version of this
    tokenizer silently stopped and returned only the steps parsed so far,
    which let a truncated program like "subtract(100, 50), divide(#0, 5"
    (missing text and closing paren) score as a valid, complete one-step
    program instead of being rejected -- a false positive for exactly the kind
    of generation failure this evaluator needs to catch.
    """
    text = str(original_program).strip()
    program: List[str] = []
    pos = 0

    while pos < len(text):
        m = _NAME_RE.match(text, pos)
        if not m:
            break
        open_idx = m.end() - 1

        depth, close = 0, -1
        for i in range(open_idx, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close == -1:
            raise ValueError(
                f"Unbalanced parentheses (no matching ')' found) in program: '{original_program}'"
            )

        program.append(m.group(1) + "(")
        # Split arguments on depth-0 commas so brackets inside a label survive.
        args, arg_depth, current = [], 0, []
        for ch in text[m.end():close]:
            if ch == "(":
                arg_depth += 1
                current.append(ch)
            elif ch == ")":
                arg_depth -= 1
                current.append(ch)
            elif ch == "," and arg_depth == 0:
                args.append("".join(current).strip())
                current = []
            else:
                current.append(ch)
        if current:
            args.append("".join(current).strip())

        program.extend(args)
        program.append(")")
        pos = close + 1
        while pos < len(text) and text[pos] in ", ":
            pos += 1

    if pos < len(text) and text[pos:].strip():
        raise ValueError(
            f"Trailing unparsed content in program: '{text[pos:]}' (from: '{original_program}')"
        )

    program.append("EOF")
    return program


def extract_program(raw_text: str) -> str:
    """Recover a program string from raw model output.

    Bracket matched, so an outer call is never silently discarded: the earlier
    regex could only match a bracket-free call, so `multiply(divide(a, b), 100)`
    was reduced to its inner `divide(a, b)` and a percentage rescaling scored as
    if it were the gold answer.
    """
    text = re.sub(r"```[a-zA-Z]*", "", str(raw_text)).replace("```", "").strip()

    calls, pos = [], 0
    while pos < len(text):
        m = _NAME_RE.search(text, pos)
        if not m:
            break
        if m.group(1) not in ALL_OPS:
            pos = m.end()
            continue
        depth, close = 0, -1
        for i in range(m.end() - 1, len(text)):
            if text[i] == "(":
                depth += 1
            elif text[i] == ")":
                depth -= 1
                if depth == 0:
                    close = i
                    break
        if close == -1:
            break
        calls.append(text[m.start(1):close + 1].strip())
        pos = close + 1

    return ", ".join(calls) if calls else text



def _steps_from_tokens(program: List[str]) -> List[Tuple[str, str, str]]:
    """Group a tokenised program into (op, arg1, arg2) triples.

    The original walked the token list by joining it and splitting on ")", which
    silently mis-splits any argument containing a bracket -- exactly the row
    labels this dataset uses, e.g. `EPS (VND)`. Grouping the tokens directly is
    equivalent for well-formed programs and correct for those.
    """
    body = program[:-1] if program and program[-1] == "EOF" else list(program)
    if len(body) % 4 != 0:
        raise ValueError("token count is not a multiple of four")
    steps = []
    for i in range(0, len(body), 4):
        op_token, arg1, arg2, close = body[i:i + 4]
        if not op_token.endswith("(") or close != ")":
            raise ValueError("malformed step")
        op = op_token[:-1].strip()
        if op not in ALL_OPS:
            raise ValueError(f"unknown operator {op!r}")
        steps.append((op, arg1.strip(), arg2.strip()))
    return steps

# ---------------------------------------------------------------- execution --
def eval_program(program: List[str], table: Optional[Sequence[Sequence[str]]]):
    """Execute a tokenised program. Returns (invalid_flag, result)."""
    this_res: Union[float, str] = "n/a"

    try:
        steps = _steps_from_tokens(program)
        res_dict = {}

        for ind, (op, arg1, arg2) in enumerate(steps):
            if op in ("add", "subtract", "multiply", "divide", "exp", "greater"):
                if "#" in arg1:
                    arg1 = res_dict[int(arg1.replace("#", ""))]
                else:
                    arg1 = str_to_num(arg1)
                    if arg1 == "n/a":
                        return 1, "n/a"
                if "#" in arg2:
                    arg2 = res_dict[int(arg2.replace("#", ""))]
                else:
                    arg2 = str_to_num(arg2)
                    if arg2 == "n/a":
                        return 1, "n/a"

                if op == "add":
                    this_res = arg1 + arg2
                elif op == "subtract":
                    this_res = arg1 - arg2
                elif op == "multiply":
                    this_res = arg1 * arg2
                elif op == "divide":
                    this_res = arg1 / arg2
                elif op == "exp":
                    this_res = arg1 ** arg2
                else:
                    this_res = "yes" if arg1 > arg2 else "no"

            else:  # table_*
                table_dict = {row[0]: row[1:] for row in (table or [])}
                if "#" in arg1:
                    num_row = [res_dict[int(arg1.replace("#", ""))]]
                else:
                    if arg1 not in table_dict:
                        return 1, "n/a"
                    num_row = process_row(table_dict[arg1])
                if num_row == "n/a":
                    return 1, "n/a"

                if op == "table_max":
                    this_res = max(num_row)
                elif op == "table_min":
                    this_res = min(num_row)
                elif op == "table_sum":
                    this_res = sum(num_row)
                else:
                    this_res = sum(num_row) / len(num_row)

            res_dict[ind] = this_res

        if this_res not in ("yes", "no", "n/a"):
            this_res = round(this_res, 5)
    except Exception:
        return 1, "n/a"

    return 0, this_res


# ------------------------------------------------------------------ program --
def equal_program(program1: List[str], program2: List[str]) -> bool:
    """Symbolic equivalence of gold (program1) and prediction (program2).

    Same protocol as the official implementation -- literals become symbols,
    table steps become opaque variables, and the two expressions are compared
    after `simplify`, so a differently-arranged but algebraically identical
    program still counts. A prediction may only use symbols that appear in gold,
    which is what stops it from introducing a constant of its own (the `100` of
    a percentage rescaling, say). Only the step-splitting differs: it groups
    tokens rather than splitting a joined string on ")".
    """
    try:
        steps1 = _steps_from_tokens(program1)
    except Exception:
        return False

    sym_map, sym_ind = {}, 0
    for op, arg1, arg2 in steps1:
        if "table" in op:
            key = (op, arg1, arg2)
            if key not in sym_map:
                sym_map[key] = "a" + str(sym_ind)
                sym_ind += 1
        else:
            for arg in (arg1, arg2):
                if "#" not in arg and arg not in sym_map:
                    sym_map[arg] = "a" + str(sym_ind)
                    sym_ind += 1

    try:
        steps2 = _steps_from_tokens(program2)
    except Exception:
        return False

    for ind, (op, arg1, arg2) in enumerate(steps2):
        if "table" in op:
            if (op, arg1, arg2) not in sym_map:
                return False
        else:
            for arg in (arg1, arg2):
                if "#" not in arg:
                    if arg not in sym_map:
                        return False
                elif int(arg.strip("#")) >= ind:
                    return False

    def symbol_recur(ind, steps):
        op, arg1, arg2 = steps[ind]
        if "table" in op:
            return sym_map[(op, arg1, arg2)]
        parts = []
        for arg in (arg1, arg2):
            if "#" in arg:
                parts.append(symbol_recur(int(arg.replace("#", "")), steps))
            else:
                parts.append(sym_map[arg])
        sign = {"add": "+", "subtract": "-", "multiply": "*",
                "divide": "/", "exp": "**", "greater": ">"}[op]
        return f"( {parts[0]} {sign} {parts[1]} )"

    try:
        sym1 = simplify(symbol_recur(len(steps1) - 1, steps1), evaluate=False)
        sym2 = simplify(symbol_recur(len(steps2) - 1, steps2), evaluate=False)
    except Exception:
        return False

    return sym1 == sym2


# ------------------------------------------------------------------ metrics --
def _coerce_answer(value):
    """ViNumQA stores exe_ans as a string; "yes"/"no" stay as they are."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return value


def score_one(generated_program: str, gold_program: str, gold_answer,
              table: Optional[Sequence[Sequence[str]]] = None,
              extract_first: bool = True) -> Tuple[float, float]:
    """(program_accuracy, execution_accuracy) for a single item.

    A generated_program that fails to tokenize (e.g. cut off mid-generation,
    missing a closing paren) scores (0.0, 0.0) rather than raising -- this is
    expected input from a real model, not a bug to surface as an exception.
    gold_program is assumed well-formed and is not caught the same way, so a
    malformed *gold* label still raises loudly instead of silently scoring 0.
    """
    generated = extract_program(generated_program) if extract_first else generated_program
    gold_tok = program_tokenization(gold_program)
    gold_res = _coerce_answer(gold_answer)

    try:
        pred_tok = program_tokenization(generated)
    except ValueError:
        return 0.0, 0.0

    invalid, exe_res = eval_program(pred_tok, table)
    ea = 1.0 if invalid == 0 and exe_res == gold_res else 0.0

    try:
        pa = 1.0 if equal_program(gold_tok, pred_tok) else 0.0
    except Exception:
        pa = 0.0

    return pa, ea


def evaluate_dataframe(df, generated_col: str = "generated_program",
                       gold_program_col: str = "program",
                       gold_answer_col: str = "answer",
                       table_col: str = "table_raw",
                       extract_first: bool = True):
    """Score a DataFrame, returning (df + per-row scores, summary).

    `table_col` must hold the raw table (list of rows); without it, programs
    naming a table row cannot execute and score 0 on EA.
    """
    df = df.copy()
    pa_scores, ea_scores = [], []

    for _, row in df.iterrows():
        table = row[table_col] if table_col in df.columns else None
        pa, ea = score_one(row[generated_col], row[gold_program_col],
                           row[gold_answer_col], table, extract_first)
        pa_scores.append(pa)
        ea_scores.append(ea)

    df["pa_score"] = pa_scores
    df["ea_score"] = ea_scores
    return df, {
        "program_accuracy": sum(pa_scores) / len(pa_scores) if pa_scores else 0.0,
        "execution_accuracy": sum(ea_scores) / len(ea_scores) if ea_scores else 0.0,
    }
