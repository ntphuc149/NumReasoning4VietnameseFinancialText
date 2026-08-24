"""Single point of access to the repo's one evaluator.

`notebooks/evaluate/scorer.py` is not importable as a package (it lives under a
notebooks directory with no `__init__.py`), so it is loaded here by path via
importlib. It is deliberately *not* copied: AGENTS.md is explicit that there is
one scorer and that a second parser must never be written. Every program this
package produces is tokenised, executed, and graded by that file.

Re-exported here:
  program_tokenization  bracket-depth-aware tokeniser (handles `ROE (%)` labels)
  extract_program       recover a program string from raw model output
  steps_from_tokens     group a tokenised program into (op, arg1, arg2) triples
  eval_program          the executor -- this is `Execute(p*)` from paper eq. (5)
  equal_program         sympy symbolic equivalence, used for "symbolic" voting
  score_one             (PA, EA) for one item
  evaluate_dataframe    (df + per-row scores, summary) for a whole run
"""

import importlib.util
import sys
from pathlib import Path
from types import ModuleType

_SCORER_RELPATH = Path("notebooks") / "evaluate" / "scorer.py"


def find_project_root(start: Path | None = None) -> Path:
    """Nearest ancestor containing a `.git` directory.

    Same resolution strategy the notebooks use, so this package works whether it
    is imported from the repo root, from the notebook's own folder, or from a
    test runner's working directory.
    """
    here = (start or Path(__file__)).resolve()
    for candidate in (here, *here.parents):
        if (candidate / ".git").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not locate the project root (no .git found above {here})."
    )


def _load_scorer() -> ModuleType:
    if "vinumqa_scorer" in sys.modules:
        return sys.modules["vinumqa_scorer"]

    path = find_project_root() / _SCORER_RELPATH
    if not path.exists():
        raise FileNotFoundError(f"Shared scorer not found at {path}")

    spec = importlib.util.spec_from_file_location("vinumqa_scorer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create a module spec for {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["vinumqa_scorer"] = module
    spec.loader.exec_module(module)
    return module


scorer = _load_scorer()

ALL_OPS = scorer.ALL_OPS
program_tokenization = scorer.program_tokenization
extract_program = scorer.extract_program
eval_program = scorer.eval_program
equal_program = scorer.equal_program
score_one = scorer.score_one
evaluate_dataframe = scorer.evaluate_dataframe
str_to_num = scorer.str_to_num
# Underscore-private in scorer.py, but it is the canonical grouping of a
# tokenised program into (op, arg1, arg2) triples. Reusing it is the whole
# point -- a second implementation here is exactly what AGENTS.md forbids.
steps_from_tokens = scorer._steps_from_tokens

# The paper's toolset (Appendix B.7/B.8) is these eight plus `join()`. ViNumQA
# additionally defines `exp` and `greater`; measured over the real splits, `exp`
# never appears and `greater` appears once in test.json and never in train, so
# the paper's eight cover 496/497 test samples. `exp`/`greater` are added to the
# prompt only when AgentConfig.use_prompt_ext is on.
PAPER_OPS = [
    "add", "subtract", "multiply", "divide",
    "table_max", "table_min", "table_sum", "table_average",
]
COMMUTATIVE_OPS = {"add", "multiply"}
TABLE_OPS = {"table_max", "table_min", "table_sum", "table_average"}
ARITH_OPS = {"add", "subtract", "multiply", "divide", "exp", "greater"}


def execute_program(program: str, table_raw) -> tuple[bool, object]:
    """`Execute(p*)` from paper eq. (5), delegated to the shared scorer.

    Returns (ok, result). `ok` is False when the program cannot be tokenised or
    the scorer flags it invalid, in which case `result` is "n/a".
    """
    try:
        tokens = program_tokenization(program)
    except ValueError:
        return False, "n/a"
    invalid, result = eval_program(tokens, table_raw)
    return invalid == 0, result
