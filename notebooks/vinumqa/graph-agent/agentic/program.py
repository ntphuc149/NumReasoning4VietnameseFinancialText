"""From planner text to a graded program: parse, transpile, vote.

Three stages of one job, kept in one file because they only ever run in
sequence and each is meaningless without the next:

    raw planner output
        -> PART 1  parse_plan()      the paper's plan DSL -> PlanGraph
        -> PART 2  transpile()       PlanGraph -> ViNumQA program string
        -> PART 3  vote()            n programs -> p*        (paper eq. 4)

PART 2 is the highest-risk code in this package. The paper's plan DSL is not the
format `scorer.py` grades, and the paper never documents its own conversion, so
this is a reconstruction. Its seven rules are each justified against the paper
text or measured against the real data -- see the PART 2 header.
"""

from __future__ import annotations

import heapq
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterator, Sequence

from agentic.scoring import (
    ARITH_OPS,
    COMMUTATIVE_OPS,
    TABLE_OPS,
    equal_program,
    program_tokenization,
    steps_from_tokens,
    str_to_num,
)

JOIN = "join"
KNOWN_TOOLS = ARITH_OPS | TABLE_OPS | {JOIN}

_REF_RE = re.compile(r"^\$\s*(\d+)$")
_QUOTE_PAIRS = {"'": "'", '"': '"', "‘": "’", "“": "”"}
_ALL_QUOTES = set(_QUOTE_PAIRS) | set(_QUOTE_PAIRS.values())
_QUOTE_CHARS = "'\"“”‘’"


class PlanParseError(ValueError):
    """The model's output is not a plan this DSL can represent."""


class TranspileError(ValueError):
    """A syntactically valid plan with no ViNumQA program equivalent."""


# =============================================================================
# PART 1 -- THE PLAN DSL  (paper Appendix B.11/B.12, section 4.3)
# =============================================================================
#
# The planner is asked to answer in exactly this shape:
#
#     1. subtract(a='600', b='500')
#     2. divide(a='$1', b='500')
#     3. join()
#     <END_OF_PLAN>
#
# `$k` refers to the output of action `k`. Because the prompt also asks for
# "maximum parallelization", a plan is not a list but a DAG: actions with no
# dependency between them sit on the same topological level and could run
# concurrently. `PlanGraph.levels()` recovers that structure -- it is the second
# of the paper's two graphs (the first being the four-node pipeline itself).
#
# Parsing is deliberately forgiving about surface form (code fences, `1)`
# instead of `1.`, a `Plan:`/`Kế hoạch:` prefix, positional instead of keyword
# args, curly quotes) and strict about semantics (unknown tool, bad arity,
# dangling reference, cycle). A candidate that survives parsing is a real
# hypothesis; one that does not is dropped before the vote rather than being
# silently repaired into something the model did not propose.

# Keyword-argument aliases. The paper names arithmetic args `a`/`b` (B.7) and
# the table arg `row_identifier` (B.7) -- but its own Figure 1 writes
# `table_max(column)`, and models drift to `row`/`row_name`, so accept those too.
_ARITH_ARG_SLOTS = {
    "a": 0, "x": 0, "first": 0, "value1": 0, "arg1": 0, "left": 0,
    "b": 1, "y": 1, "second": 1, "value2": 1, "arg2": 1, "right": 1,
}
_TABLE_ARG_SLOTS = {
    "row_identifier": 0, "row": 0, "row_name": 0, "rowname": 0,
    "name": 0, "column": 0, "col": 0, "identifier": 0,
}
# `add(a, b, context: Optional[list[str]])` -- the signature in B.7 has a third
# parameter the target program format has no place for.
_IGNORED_ARGS = {"context"}

_END_MARKERS = ("<END_OF_PLAN>", "<end_of_plan>")
_ACTION_RE = re.compile(r"(\d+)\s*[.)\]]\s*([A-Za-z_]\w*)\s*\(")
_BARE_CALL_RE = re.compile(r"\b([A-Za-z_]\w*)\s*\(")


@dataclass(frozen=True)
class Action:
    id: int
    tool: str
    args: tuple[str, ...]

    @property
    def is_join(self) -> bool:
        return self.tool == JOIN

    def refs(self) -> tuple[int, ...]:
        """Action ids this action reads, in argument order."""
        out = []
        for arg in self.args:
            match = _REF_RE.match(arg.strip())
            if match:
                out.append(int(match.group(1)))
        return tuple(out)

    def __str__(self) -> str:
        if self.is_join:
            return f"{self.id}. join()"
        if self.tool in TABLE_OPS:
            return f"{self.id}. {self.tool}(row_identifier='{self.args[0]}')"
        return f"{self.id}. {self.tool}(a='{self.args[0]}', b='{self.args[1]}')"


@dataclass
class PlanGraph:
    actions: list[Action]
    raw: str = ""
    warnings: list[str] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.by_id = {a.id: a for a in self.actions}

    @property
    def computation(self) -> list[Action]:
        """Everything except `join()` -- the actions that produce a value."""
        return [a for a in self.actions if not a.is_join]

    @property
    def joins(self) -> list[Action]:
        return [a for a in self.actions if a.is_join]

    def answer_action(self) -> Action:
        """The action whose output is the final answer.

        `join()` is the declared sink (B.11: "must be used as the last step"),
        so if it names an input, that input is the answer. The paper's own
        examples call `join()` with no arguments, in which case the answer is
        the last computation action.
        """
        for join in reversed(self.joins):
            refs = join.refs()
            if refs and refs[-1] in self.by_id:
                return self.by_id[refs[-1]]
        computation = self.computation
        if not computation:
            raise PlanParseError("plan has no computation actions")
        return max(computation, key=lambda a: a.id)

    def ancestors_of(self, action: Action) -> set[int]:
        """`action` plus everything it transitively depends on."""
        seen: set[int] = set()
        stack = [action.id]
        while stack:
            current = stack.pop()
            if current in seen or current not in self.by_id:
                continue
            seen.add(current)
            stack.extend(self.by_id[current].refs())
        return seen

    def levels(self) -> list[list[Action]]:
        """Topological levels: actions on the same level are independent.

        This is the "maximum parallelization" the planner prompt asks for, made
        explicit. Raises PlanParseError on a cycle.
        """
        computation = self.computation
        depth: dict[int, int] = {}

        def resolve(action_id: int, seen: frozenset[int]) -> int:
            if action_id in depth:
                return depth[action_id]
            if action_id in seen:
                raise PlanParseError(f"cyclic dependency at action {action_id}")
            action = self.by_id.get(action_id)
            if action is None:
                raise PlanParseError(f"dangling reference to action {action_id}")
            refs = list(action.refs())
            value = 0 if not refs else 1 + max(
                resolve(r, seen | {action_id}) for r in refs
            )
            depth[action_id] = value
            return value

        for action in computation:
            resolve(action.id, frozenset())

        out: list[list[Action]] = []
        for action in sorted(computation, key=lambda a: (depth[a.id], a.id)):
            level = depth[action.id]
            while len(out) <= level:
                out.append([])
            out[level].append(action)
        return out

    def validate(self) -> None:
        known = {a.id for a in self.actions}
        for action in self.actions:
            for ref in action.refs():
                if ref not in known:
                    raise PlanParseError(
                        f"action {action.id} references undefined ${ref}"
                    )
        self.levels()  # raises on cycles


def _strip_quotes(text: str) -> str:
    text = text.strip()
    while len(text) >= 2 and text[0] in _ALL_QUOTES and text[-1] in _ALL_QUOTES:
        text = text[1:-1].strip()
    return text


def _split_args(body: str) -> list[str]:
    """Split on commas at paren-depth 0 and outside quotes.

    Depth- and quote-awareness both matter: a row label can contain brackets
    (`ROE (%)`) and, in principle, a comma inside quotes.
    """
    args: list[str] = []
    current: list[str] = []
    depth = 0
    quote: str | None = None
    for char in body:
        if quote is not None:
            current.append(char)
            if char == quote or (quote in _QUOTE_PAIRS and char == _QUOTE_PAIRS[quote]):
                quote = None
            continue
        if char in _QUOTE_PAIRS:
            quote = char
            current.append(char)
        elif char in "([":
            depth += 1
            current.append(char)
        elif char in ")]":
            depth -= 1
            current.append(char)
        elif char == "," and depth == 0:
            args.append("".join(current).strip())
            current = []
        else:
            current.append(char)
    tail = "".join(current).strip()
    if tail:
        args.append(tail)
    return args


def _match_paren(text: str, open_idx: int) -> int:
    """Index of the `)` matching the `(` at `open_idx`, or -1."""
    depth = 0
    quote: str | None = None
    for i in range(open_idx, len(text)):
        char = text[i]
        if quote is not None:
            if char == quote or (quote in _QUOTE_PAIRS and char == _QUOTE_PAIRS[quote]):
                quote = None
            continue
        if char in _QUOTE_PAIRS:
            quote = char
        elif char == "(":
            depth += 1
        elif char == ")":
            depth -= 1
            if depth == 0:
                return i
    return -1


def _normalise_args(tool: str, raw_args: Sequence[str]) -> tuple[str, ...]:
    """Map keyword or positional args onto the tool's positional slots."""
    if tool == JOIN:
        # join()'s arguments are only read to find the answer action; keep any
        # $refs and drop the rest.
        return tuple(
            _strip_quotes(a.split("=", 1)[-1]) for a in raw_args
            if _REF_RE.match(_strip_quotes(a.split("=", 1)[-1]))
        )

    arity = 1 if tool in TABLE_OPS else 2
    slots = _TABLE_ARG_SLOTS if tool in TABLE_OPS else _ARITH_ARG_SLOTS
    filled: dict[int, str] = {}
    positional: list[str] = []

    for raw in raw_args:
        name, _, value = raw.partition("=")
        key = name.strip().lower()
        # Only treat `x=y` as a keyword arg if the left side is a bare
        # identifier; `a='>=3'` and the like must stay positional.
        if value and re.fullmatch(r"[A-Za-z_]\w*", key):
            if key in _IGNORED_ARGS:
                continue
            if key in slots:
                filled[slots[key]] = _strip_quotes(value)
                continue
            positional.append(_strip_quotes(value))
            continue
        positional.append(_strip_quotes(raw))

    ordered: list[str] = []
    spare = iter(positional)
    for slot in range(arity):
        if slot in filled:
            ordered.append(filled[slot])
        else:
            ordered.append(next(spare, ""))

    if any(part == "" for part in ordered):
        raise PlanParseError(
            f"{tool} expects {arity} argument(s), got {list(raw_args)!r}"
        )
    return tuple(ordered)


def _iter_calls(text: str) -> Iterator[tuple[int | None, str, str]]:
    """Yield (explicit_id, tool, body) for each call, in source order."""
    pos = 0
    while pos < len(text):
        numbered = _ACTION_RE.search(text, pos)
        bare = _BARE_CALL_RE.search(text, pos)
        # Whichever call comes first wins, with the numbered form preferred when
        # they overlap -- the bare regex also matches the tool inside `1. tool(`,
        # and that match starts later, so a plain `<=` on start positions picks
        # the numbered reading without discarding an earlier unnumbered call.
        use_numbered = numbered is not None and (
            bare is None or numbered.start() <= bare.start()
        )
        match = numbered if use_numbered else bare
        if match is None:
            return
        tool = match.group(2 if use_numbered else 1)
        open_idx = match.end() - 1
        close_idx = _match_paren(text, open_idx)
        if close_idx == -1:
            return  # truncated generation: stop at the last complete call
        if tool.lower() in KNOWN_TOOLS:
            yield (
                int(match.group(1)) if use_numbered else None,
                tool.lower(),
                text[open_idx + 1:close_idx],
            )
        pos = close_idx + 1


def parse_plan(raw: str) -> PlanGraph:
    """Parse the planner's raw output into a validated PlanGraph."""
    if not raw or not raw.strip():
        raise PlanParseError("empty plan")

    text = re.sub(r"```[A-Za-z]*", "", raw).replace("```", "")
    for marker in _END_MARKERS:
        index = text.find(marker)
        if index != -1:
            text = text[:index]
            break

    actions: list[Action] = []
    warnings: list[str] = []
    next_auto_id = 1
    for explicit_id, tool, body in _iter_calls(text):
        raw_args = _split_args(body)
        try:
            args = _normalise_args(tool, raw_args)
        except PlanParseError:
            if tool == JOIN:
                args = ()
            else:
                raise
        action_id = explicit_id if explicit_id is not None else next_auto_id
        if any(a.id == action_id for a in actions):
            warnings.append(f"duplicate action id {action_id}; renumbered")
            action_id = max(a.id for a in actions) + 1
        actions.append(Action(id=action_id, tool=tool, args=args))
        next_auto_id = max(next_auto_id, action_id) + 1

    if not actions:
        raise PlanParseError(f"no recognised tool calls in: {raw[:200]!r}")

    plan = PlanGraph(actions=actions, raw=raw, warnings=warnings)
    if not plan.computation:
        raise PlanParseError("plan contains only join()")
    plan.validate()
    return plan


# =============================================================================
# PART 2 -- TRANSPILE  (plan DSL -> ViNumQA program string)
# =============================================================================
#
# The paper's plan DSL is not the format this task is graded in, and the gap is
# where silent failures live. Every rule below is one difference, and each is
# justified against either the paper text or the real data:
#
#   1. `join()` and `<END_OF_PLAN>` are dropped -- they have no counterpart.
#   2. `$k` (action id, 1-based) becomes `#j` (step position, 0-based). The
#      mapping is computed from the emitted order, never assumed to be `k-1`:
#      ids can be non-contiguous, and pruning (rule 4) changes positions.
#   3. `table_max(row_identifier='X')` takes one argument in the DSL and two in
#      the target: `table_max(X, none)`. Verified against the data -- all 63
#      `table_*` calls in the gold test programs pass exactly `none` as the
#      second argument. The row label is emitted unquoted: real labels contain
#      brackets (`ROE (%)`, `P/E (x)`, `EPS (VND)`), which scorer.py's
#      bracket-aware tokeniser handles but which quotes would break.
#   4. Actions the answer does not depend on are pruned. `eval_program` returns
#      the last step's value, so a dead branch cannot change EA -- but
#      `equal_program` walks *every* step of the prediction and rejects one
#      whose literal is absent from gold, so a dead branch can silently cost PA.
#   5. Literals pass through verbatim apart from quotes, `$`, and thousands
#      separators. No rescaling: scorer.py's PA admits only literals that occur
#      in the gold program, so any well-meaning normalisation is a PA loss.
#      (Gold does legitimately contain `100` -- e.g. `subtract(96.67, 100),
#      divide(#0, 100)` for a base-100 index -- so `100` cannot be
#      special-cased.)
#   6. `#k` may reference a `table_*` result. B.9/B.10 forbids it ("Variables
#      ($x) may ONLY refer to the output of a previous add, subtract, multiply,
#      or divide action") but ViNumQA gold does it, and scorer.py executes it,
#      so the transpiler accepts it.
#   7. An argument that is neither `#k` nor a parseable number is rejected. The
#      scorer would return "n/a" for it anyway; failing here makes the reason
#      visible in the trace instead of hiding it in a zero score.

_THOUSANDS_RE = re.compile(r"^-?\d{1,3}(,\d{3})+(\.\d+)?%?$")


@dataclass
class TranspileResult:
    program: str
    emitted: list[Action]
    warnings: list[str] = field(default_factory=list)

    @property
    def n_steps(self) -> int:
        return len(self.emitted)


def clean_literal(raw: str) -> str:
    """Normalise a literal without changing its value.

    Only three things are removed: surrounding quotes (already handled at parse
    time, repeated here so the function is safe standalone), currency symbols,
    and thousands separators. The last is not cosmetic -- a comma inside an
    argument would be read as an argument separator by the scorer's tokeniser,
    turning `subtract(1,041, 500)` into a three-argument step.
    """
    text = raw.strip().strip(_QUOTE_CHARS).strip()
    text = text.replace("$", "").replace("\xa0", " ").strip()
    if _THOUSANDS_RE.match(text):
        text = text.replace(",", "")
    # Vietnamese financial text also groups thousands with a space (often the
    # non-breaking one normalised just above): "1 234" must become "1234" or
    # str_to_num rejects it. Only collapse when doing so yields a number, so a
    # genuinely non-numeric argument is left alone and rejected downstream.
    if " " in text and str_to_num(text) == "n/a":
        squeezed = text.replace(" ", "")
        if str_to_num(squeezed) != "n/a":
            text = squeezed
    return text


def _is_ref(arg: str) -> bool:
    return _REF_RE.match(arg.strip()) is not None


def _ref_id(arg: str) -> int:
    match = _REF_RE.match(arg.strip())
    if match is None:
        raise TranspileError(f"not a reference: {arg!r}")
    return int(match.group(1))


def _emit_arg(arg: str, index_of: dict[int, int]) -> str:
    if _is_ref(arg):
        target = _ref_id(arg)
        if target not in index_of:
            raise TranspileError(f"reference ${target} is not an emitted step")
        return f"#{index_of[target]}"
    literal = clean_literal(arg)
    if not literal:
        raise TranspileError("empty argument")
    if "," in literal or "(" in literal or ")" in literal:
        raise TranspileError(f"literal would break tokenisation: {literal!r}")
    if str_to_num(literal) == "n/a":
        raise TranspileError(f"argument is not numeric: {literal!r}")
    return literal


def _emit_row_label(arg: str) -> str:
    label = arg.strip().strip(_QUOTE_CHARS).strip()
    if not label:
        raise TranspileError("empty row label")
    if "," in label:
        # A depth-0 comma inside a row label splits the step into three
        # arguments when the scorer tokenises it.
        raise TranspileError(f"row label contains a comma: {label!r}")
    return label


def _topological_order(plan: PlanGraph, keep: set[int]) -> list[Action]:
    """Kahn's algorithm over the kept subgraph, ties broken by ascending id.

    Every kept action is an ancestor of the answer action, so the answer has no
    outgoing edge within the set and is necessarily emitted last -- which is
    what `eval_program` (last step is the result) and `equal_program` (builds
    the expression from the last step) both require.
    """
    pending = {
        aid: [r for r in plan.by_id[aid].refs() if r in keep] for aid in keep
    }
    dependents: dict[int, list[int]] = {aid: [] for aid in keep}
    for aid, refs in pending.items():
        for ref in refs:
            dependents[ref].append(aid)

    remaining = {aid: len(refs) for aid, refs in pending.items()}
    ready = [aid for aid, count in remaining.items() if count == 0]
    heapq.heapify(ready)

    order: list[Action] = []
    while ready:
        aid = heapq.heappop(ready)
        order.append(plan.by_id[aid])
        for dependent in dependents[aid]:
            remaining[dependent] -= 1
            if remaining[dependent] == 0:
                heapq.heappush(ready, dependent)

    if len(order) != len(keep):
        raise TranspileError("cyclic plan")
    return order


def transpile(plan: PlanGraph) -> TranspileResult:
    """Turn a parsed plan into a ViNumQA program string."""
    if not plan.computation:
        raise TranspileError("plan has no computation actions")

    answer = plan.answer_action()
    keep = plan.ancestors_of(answer) & {a.id for a in plan.computation}
    if answer.id not in keep:
        raise TranspileError("answer action is not a computation action")

    order = _topological_order(plan, keep)
    if order[-1].id != answer.id:
        raise TranspileError("answer action is not last in topological order")

    index_of = {action.id: position for position, action in enumerate(order)}

    warnings = list(plan.warnings)
    dropped = len(plan.computation) - len(order)
    if dropped:
        warnings.append(f"pruned {dropped} action(s) the answer does not depend on")

    steps: list[str] = []
    for action in order:
        if action.tool in TABLE_OPS:
            label = _emit_row_label(action.args[0])
            steps.append(f"{action.tool}({label}, none)")
        elif action.tool in ARITH_OPS:
            left = _emit_arg(action.args[0], index_of)
            right = _emit_arg(action.args[1], index_of)
            steps.append(f"{action.tool}({left}, {right})")
        else:
            raise TranspileError(f"cannot emit tool {action.tool!r}")

    program = ", ".join(steps)

    # Round-trip through the grader's own tokeniser: if it cannot read what we
    # just wrote, the candidate is broken regardless of how it looks.
    try:
        program_tokenization(program)
    except ValueError as exc:
        raise TranspileError(f"emitted program does not tokenise: {exc}") from exc

    return TranspileResult(program=program, emitted=order, warnings=warnings)


def plan_text_to_program(raw: str) -> TranspileResult:
    """Convenience: raw planner output -> ViNumQA program, in one call."""
    return transpile(parse_plan(raw))


def program_to_plan_text(program: str) -> str:
    """ViNumQA program -> plan DSL. The inverse of `transpile`, for testing.

    Used by the round-trip property test: for every gold program `g` in the real
    splits, `transpile(parse_plan(program_to_plan_text(g)))` must be equivalent
    to `g`. That exercises step renumbering, the `table_*(row, none)` arity fix,
    and bracketed row labels against thousands of real cases without spending a
    single API call.
    """
    steps = steps_from_tokens(program_tokenization(program))
    lines: list[str] = []
    for index, (op, arg1, arg2) in enumerate(steps):
        action_id = index + 1
        if op in TABLE_OPS:
            lines.append(f"{action_id}. {op}(row_identifier={_quote(arg1)})")
        else:
            lines.append(
                f"{action_id}. {op}(a={_quote(_to_ref(arg1))}, "
                f"b={_quote(_to_ref(arg2))})"
            )
    lines.append(f"{len(steps) + 1}. join()")
    return "\n".join(lines) + "\n<END_OF_PLAN>"


def _to_ref(arg: str) -> str:
    """`#0` (0-based step) -> `$1` (1-based action id)."""
    text = arg.strip()
    if text.startswith("#"):
        return f"${int(text[1:]) + 1}"
    return text


def _quote(value: str) -> str:
    return f'"{value}"' if "'" in value else f"'{value}'"


def summarise_plan(plan: PlanGraph) -> str:
    """Human-readable plan DAG, level by level. For traces and debugging."""
    lines = []
    for depth, level in enumerate(plan.levels()):
        rendered = "  |  ".join(str(action) for action in level)
        lines.append(f"  level {depth}: {rendered}")
    return "\n".join(lines)


def parallelism(plan: PlanGraph) -> float:
    """Mean actions per topological level -- how parallel the plan actually is.

    The planner prompt asks for "maximum parallelization"; this is the number
    that says whether it complied.
    """
    levels = list(plan.levels())
    if not levels:
        return 0.0
    return sum(len(level) for level in levels) / len(levels)


# =============================================================================
# PART 3 -- VOTING  (paper section 4.4, equations 4 and 5)
# =============================================================================
#
#     "First, the n candidate programs are canonicalized (e.g., arguments of
#      commutative operators are sorted) and grouped to find the set of unique
#      programs. The unique program that was generated most frequently is
#      selected as the optimal one, p*. [...] In case of a tie in frequency, the
#      program with fewer steps (lower complexity) is chosen."
#
# `canonical` (default, the paper's prose) keys a candidate on a canonical form
# of the expression tree rooted at its final step: commutative arguments sorted,
# numeric literals normalised so `600`, `600.0` and `600.00` agree, and steps the
# final answer does not depend on ignored.
#
# `symbolic` clusters with scorer.py's sympy `equal_program` instead. Worth being
# precise about what that buys, because it is less than it sounds:
# `equal_program` admits only literals present in the program it is compared
# against, so it merges *algebraic rearrangements over the same literals*
# (`multiply(add(a,b), c)` with `add(multiply(a,c), multiply(b,c))`) and nothing
# more. It does **not** merge the paper's Example 5.1 pair --
# `divide(1041, 0.292)` against `divide(29.2, 100), divide(1041, #0)` -- because
# the second introduces `29.2` and `100`. That case is a prompt problem, not a
# voting problem, and is addressed by `use_prompt_ext`.
#
# KNOWN GAP: the paper's Figure 1 labels this stage "Top result voting" and
# shows executed values beside each candidate. Its worked example only reaches
# the depicted 2-vs-1 majority if candidates are grouped by executed *value*
# rather than by program structure. Neither mode here does that yet.


def _canonical_number(arg: str) -> str:
    """Normalise a numeric literal so equal values share one key.

    Uses the grader's own `str_to_num`, so `48.8%` and `0.488` -- the same
    quantity written two ways -- land in the same cluster.
    """
    value = str_to_num(arg)
    if value == "n/a":
        return arg.strip().lower()
    return format(round(float(value), 10), ".10g")


def canonicalize(program: str) -> str:
    """Canonical key for one program. Raises ValueError if it cannot be read."""
    steps = steps_from_tokens(program_tokenization(program))
    if not steps:
        raise ValueError("empty program")

    def render(index: int, seen: frozenset[int]) -> str:
        if index in seen:
            raise ValueError(f"cyclic reference at step {index}")
        op, arg1, arg2 = steps[index]
        if op in TABLE_OPS:
            return f"{op}({arg1.strip().lower()})"
        parts = []
        for arg in (arg1, arg2):
            text = arg.strip()
            if text.startswith("#"):
                parts.append(render(int(text[1:]), seen | {index}))
            else:
                parts.append(_canonical_number(text))
        if op in COMMUTATIVE_OPS:
            parts.sort()
        return f"{op}({parts[0]},{parts[1]})"

    # Rooted at the final step, so dead branches do not split a cluster.
    return render(len(steps) - 1, frozenset())


def n_steps(program: str) -> int:
    try:
        return len(steps_from_tokens(program_tokenization(program)))
    except ValueError:
        return 10**6  # unreadable: never wins a "fewer steps" tie-break


@dataclass
class Cluster:
    key: str
    members: list[int] = field(default_factory=list)   # indices into candidates
    representative: str = ""

    @property
    def count(self) -> int:
        return len(self.members)


@dataclass
class VoteResult:
    winner: str | None
    winner_index: int | None
    clusters: list[Cluster]
    counts: Counter = field(default_factory=Counter)

    @property
    def n_clusters(self) -> int:
        return len(self.clusters)

    @property
    def consensus(self) -> float:
        """Winning cluster's share of the votes -- a confidence proxy.

        Near 1.0 means every sampled path agreed; the paper's "systematic
        reasoning error" failure mode looks exactly like high consensus on a
        wrong answer, so this is worth recording even though it is not used to
        select.
        """
        total = sum(c.count for c in self.clusters)
        if not total or self.winner_index is None:
            return 0.0
        best = max(self.clusters, key=lambda c: c.count)
        return best.count / total


def _cluster_canonical(programs: Sequence[str]) -> list[Cluster]:
    by_key: dict[str, Cluster] = {}
    for index, program in enumerate(programs):
        try:
            key = canonicalize(program)
        except (ValueError, KeyError, IndexError):
            key = f"__unparsed__::{program.strip()}"
        cluster = by_key.get(key)
        if cluster is None:
            cluster = Cluster(key=key, representative=program)
            by_key[key] = cluster
        cluster.members.append(index)
    return list(by_key.values())


def _cluster_symbolic(programs: Sequence[str]) -> list[Cluster]:
    """Pairwise `equal_program` against each cluster's representative.

    O(n * clusters), which is fine at n=15 -- and sympy `simplify` is the
    expensive part, so keeping the comparison count low matters more than the
    asymptotics.
    """
    clusters: list[Cluster] = []
    tokens: list[list[str] | None] = []
    for program in programs:
        try:
            tokens.append(program_tokenization(program))
        except ValueError:
            tokens.append(None)

    for index, program in enumerate(programs):
        placed = False
        if tokens[index] is not None:
            for cluster in clusters:
                rep_tokens = tokens[cluster.members[0]]
                if rep_tokens is None:
                    continue
                try:
                    same = equal_program(rep_tokens, tokens[index])
                except Exception:  # noqa: BLE001 - sympy raises freely
                    same = False
                if same:
                    cluster.members.append(index)
                    placed = True
                    break
        if not placed:
            clusters.append(
                Cluster(key=f"sym{len(clusters)}", members=[index],
                        representative=program)
            )
    return clusters


def vote(programs: Sequence[str], mode: str = "canonical") -> VoteResult:
    """`p* = argmax Score(p_i)` -- equation (4).

    Score is cluster frequency. Ties are broken on fewer steps, then on the
    earliest generation index: with `n` independent samples in no meaningful
    order, first-generated is the only stable tie-break available, and it is the
    same convention this repo's STaNR majority-of-k voting uses.
    """
    if not programs:
        return VoteResult(winner=None, winner_index=None, clusters=[])

    clusters = (
        _cluster_symbolic(programs) if mode == "symbolic"
        else _cluster_canonical(programs)
    )

    for cluster in clusters:
        # Within a cluster the members are equivalent by construction, so the
        # shortest surface form is the one to emit.
        best = min(cluster.members, key=lambda i: (n_steps(programs[i]), i))
        cluster.representative = programs[best]

    def rank(cluster: Cluster):
        best = min(cluster.members, key=lambda i: (n_steps(programs[i]), i))
        return (-cluster.count, n_steps(programs[best]), best)

    winner_cluster = min(clusters, key=rank)
    winner_index = min(
        winner_cluster.members, key=lambda i: (n_steps(programs[i]), i)
    )

    counts = Counter({cluster.key: cluster.count for cluster in clusters})
    return VoteResult(
        winner=programs[winner_index],
        winner_index=winner_index,
        clusters=sorted(clusters, key=lambda c: -c.count),
        counts=counts,
    )
