"""The four agent nodes and the pipeline graph they form.

    q, C ──▶ [1] SubqueryGenerator   G_sq(q, C)          §4.1 eq(1)  B.1/B.2
                    │ fan-out, one independent call per subquery
             [2] SubqueryAnswerer    A_sq(sq_j, C)       §4.2 eq(2)  B.3/B.4
                    │ fan-in
             [3] Planner             P_n-sample(...)     §4.3 eq(3)  B.5-B.12
                    │
             [4] EquationExtractor   vote -> p* -> a*    §4.4 eq(4)(5)

Every node is an `AgentState -> AgentState` step that *appends* to the state
rather than replacing it, so a finished state is a complete, inspectable record
of how the answer was reached: which subqueries were asked, what each returned,
all n sampled plans, how they clustered, and which one won. That record is what
makes the paper's failure-mode analysis (§5.5.2) reproducible instead of
anecdotal -- and it is the only way to compute oracle@n after the fact, or to
re-vote under a different mode without re-running anything.
"""

from __future__ import annotations

import json
import re
import time
from abc import ABC, abstractmethod
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence

from agentic import prompts
from agentic.config import AgentConfig
from agentic.llm import LLMClient, LLMError, table_row_labels
from agentic.program import (
    PlanGraph,
    PlanParseError,
    TranspileError,
    VoteResult,
    parallelism,
    parse_plan,
    transpile,
    vote,
)
from agentic.scoring import (
    TABLE_OPS,
    execute_program,
    extract_program,
    program_tokenization,
    steps_from_tokens,
)

# =============================================================================
# STATE
# =============================================================================


@dataclass
class Candidate:
    """One of the n sampled reasoning paths, at whatever stage it reached."""

    index: int
    raw_plan: str
    plan: Optional[PlanGraph] = None
    program: Optional[str] = None
    error: Optional[str] = None
    warnings: list[str] = field(default_factory=list)
    executable: bool = False
    exe_result: Any = None
    parallelism: float = 0.0

    @property
    def usable(self) -> bool:
        return self.program is not None and self.executable

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "program": self.program,
            "error": self.error,
            "warnings": self.warnings,
            "executable": self.executable,
            "exe_result": self.exe_result,
            "parallelism": round(self.parallelism, 3),
            "raw_plan": self.raw_plan,
        }


@dataclass
class NodeTrace:
    node: str
    seconds: float
    ok: bool
    detail: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "node": self.node,
            "seconds": round(self.seconds, 3),
            "ok": self.ok,
            **self.detail,
        }


@dataclass
class AgentState:
    sample_id: str
    question: str
    context: str
    pre_text: str = ""
    table: str = ""
    post_text: str = ""
    table_raw: Optional[Sequence[Sequence[str]]] = None

    subqueries: list[str] = field(default_factory=list)
    subquery_answers: list[tuple[str, str]] = field(default_factory=list)
    raw_plans: list[str] = field(default_factory=list)
    candidates: list[Candidate] = field(default_factory=list)
    vote: Optional[VoteResult] = None

    program: str = ""
    answer: Any = None
    fallback: Optional[str] = None

    traces: list[NodeTrace] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)

    @property
    def usable_candidates(self) -> list[Candidate]:
        return [c for c in self.candidates if c.usable]

    def to_dict(self, keep_candidates: bool = True) -> dict:
        out: dict[str, Any] = {
            "id": self.sample_id,
            "question": self.question,
            "subqueries": self.subqueries,
            "subquery_answers": [
                {"subquery": q, "answer": a} for q, a in self.subquery_answers
            ],
            "program": self.program,
            "answer": self.answer,
            "fallback": self.fallback,
            "n_candidates": len(self.candidates),
            "n_usable": len(self.usable_candidates),
            "errors": self.errors,
            "traces": [t.to_dict() for t in self.traces],
        }
        if self.vote is not None:
            out["vote"] = {
                "n_clusters": self.vote.n_clusters,
                "consensus": round(self.vote.consensus, 4),
                "counts": dict(self.vote.counts),
            }
        if keep_candidates:
            out["candidates"] = [c.to_dict() for c in self.candidates]
        return out


class Node(ABC):
    """A single vertex of the pipeline graph."""

    name: str = "node"

    @abstractmethod
    def execute(self, state: AgentState) -> AgentState:
        """Do the node's work. Exceptions are caught and traced by `run`."""

    def run(self, state: AgentState) -> AgentState:
        started = time.monotonic()
        ok = True
        try:
            state = self.execute(state)
        except Exception as exc:  # noqa: BLE001 - one node must not kill a run
            ok = False
            state.errors.append(f"{self.name}: {type(exc).__name__}: {exc}")
        state.traces.append(
            NodeTrace(
                node=self.name,
                seconds=time.monotonic() - started,
                ok=ok,
                detail=self.trace_detail(state),
            )
        )
        return state

    def trace_detail(self, state: AgentState) -> dict:
        return {}


# =============================================================================
# NODE 1 -- Subquery Generator   (§4.1, eq. 1, prompt B.1/B.2)
# =============================================================================
#
#     SQ = G_sq(q, C)
#
# Decomposes the question into k atomic, fact-seeking subqueries. The prompt's
# three "DO NOT" rules (no comparisons, no calculations, no final answers) are
# what enforce the division of labour the whole pipeline rests on: this node
# finds *where the numbers are*, and only the planner may combine them.
#
# Ablation note: turning this node off is the paper's "Multi-Path Only" row
# (Table 4). It costs surprisingly little on its own (-0.09 EA at 8B) -- the
# planner still receives the full context `C` either way.

_JSON_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_subqueries(raw: str) -> list[str]:
    """Recover the subquery list from the model's reply.

    The prompt asks for `{"subqueries": [...]}`, but models wrap it in prose or
    a code fence, or return a bare array. Each of those is recovered rather than
    thrown away -- a failed parse costs the whole decomposition stage for that
    sample.
    """
    if not raw or not raw.strip():
        return []

    fenced = _JSON_FENCE_RE.search(raw)
    fragments = [fenced.group(1)] if fenced else []
    fragments.append(raw)

    for fragment in fragments:
        for opener, closer in (("{", "}"), ("[", "]")):
            start = fragment.find(opener)
            end = fragment.rfind(closer)
            if start == -1 or end <= start:
                continue
            try:
                parsed = json.loads(fragment[start:end + 1])
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                for key in ("subqueries", "sub_queries", "questions", "queries"):
                    value = parsed.get(key)
                    if isinstance(value, list):
                        return [str(v).strip() for v in value if str(v).strip()]
            elif isinstance(parsed, list):
                return [str(v).strip() for v in parsed if str(v).strip()]
    return []


class SubqueryGenerator(Node):
    name = "subquery_generator"

    def __init__(self, client: LLMClient, config: AgentConfig):
        self.client = client
        self.config = config
        self.prompts = prompts.get(config.prompt_lang)

    def execute(self, state: AgentState) -> AgentState:
        if not self.config.use_decomposition:
            return state

        prompt = self.prompts.subquery_generator.format(
            context=state.context, question=state.question
        )
        raw = self.client.complete(
            system=None,
            user=prompt,
            model=self.config.model_subquery_gen,
            max_tokens=self.config.max_tokens_subquery_gen,
        )
        subqueries = extract_subqueries(raw)
        if not subqueries:
            # Degrade rather than fail: the planner still sees the full context,
            # so the sample continues as if decomposition were disabled.
            state.errors.append(f"{self.name}: no subqueries parsed")
        state.subqueries = subqueries[: self.config.max_subqueries]
        return state

    def trace_detail(self, state: AgentState) -> dict:
        return {"n_subqueries": len(state.subqueries)}


# =============================================================================
# NODE 2 -- Subquery Answerer   (§4.2, eq. 2, prompt B.3/B.4)
# =============================================================================
#
#     V = { v_j | v_j = A_sq(sq_j, C) }
#
# Each subquery is answered independently against the *full* context. That
# independence is the point: the paper calls this "grounded data extraction",
# and running the calls in isolation is what stops one retrieval mistake from
# contaminating the others. It also makes the stage embarrassingly parallel,
# which is the pipeline graph's one fan-out.


class SubqueryAnswerer(Node):
    name = "subquery_answerer"

    def __init__(self, client: LLMClient, config: AgentConfig):
        self.client = client
        self.config = config
        self.prompts = prompts.get(config.prompt_lang)

    def _answer_one(self, subquery: str, context: str) -> str:
        prompt = self.prompts.subquery_answerer.format(
            context=context, subquery=subquery
        )
        try:
            return self.client.complete(
                system=None,
                user=prompt,
                model=self.config.model_subquery_ans,
                max_tokens=self.config.max_tokens_subquery_ans,
            )
        except LLMError:
            # One unanswered subquery is not fatal -- it simply contributes no
            # fact to the planner's additional context.
            return ""

    def execute(self, state: AgentState) -> AgentState:
        if not state.subqueries:
            return state

        workers = min(self.config.max_workers_subquery, len(state.subqueries))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            answers = list(
                pool.map(
                    lambda sq: self._answer_one(sq, state.context),
                    state.subqueries,
                )
            )

        state.subquery_answers = [
            (sq, ans) for sq, ans in zip(state.subqueries, answers) if ans.strip()
        ]
        missing = len(state.subqueries) - len(state.subquery_answers)
        if missing:
            state.errors.append(f"{self.name}: {missing} subquery answer(s) empty")
        return state

    def trace_detail(self, state: AgentState) -> dict:
        return {"n_answers": len(state.subquery_answers)}


# =============================================================================
# NODE 3 -- Plan and Scheduler   (§4.3, eq. 3, prompts B.5-B.12)
# =============================================================================
#
#     P_cand = P_n-sample(V, C, q, T)      with n = 15
#
# The paper's stated core contribution. Instead of decoding one plan, it samples
# n independent ones at temperature 0.6 and lets the next node decide between
# them. The ablation (Table 4) shows this is the component doing most of the
# work: dropping it costs 5.6 EA / 9.3 PA at 8B, against 0.1 EA for dropping
# decomposition.
#
# This node produces *raw plan text* only. Parsing, transpiling, executing, and
# voting all belong to the Equation Extractor -- the fourth node in the paper's
# Figure 1 -- so that a plan which the model wrote but which cannot be expressed
# as a ViNumQA program is visible as a distinct, countable failure rather than
# being lost inside the planner.


class Planner(Node):
    name = "planner"

    def __init__(self, client: LLMClient, config: AgentConfig):
        self.client = client
        self.config = config
        self.prompts = prompts.get(config.prompt_lang)

    def build_user_prompt(self, state: AgentState) -> str:
        """B.7 + B.9 + B.11, then the reconstructed carrier block.

        The three instruction parts are printed as separate figures in the paper
        but are described there as one "comprehensive, multi-part user prompt",
        so they are concatenated in order.
        """
        parts = [self.prompts.planner_part1, self.prompts.planner_part2]
        if self.config.use_prompt_ext:
            parts.append(prompts.planner_extension(self.config.prompt_lang))
        parts.append(self.prompts.planner_part3)

        subquery_block = ""
        if state.subquery_answers:
            rendered = "\n".join(
                f"- {question}\n  {answer}"
                for question, answer in state.subquery_answers
            )
            subquery_block = self.prompts.subquery_block_header.format(
                subquery_answers=rendered
            )

        parts.append(
            self.prompts.planner_query_block.format(
                question=state.question,
                context=state.context,
                subquery_block=subquery_block,
            )
        )
        return "\n\n".join(parts)

    def execute(self, state: AgentState) -> AgentState:
        state.raw_plans = self.client.sample_n(
            system=self.prompts.planner_system,
            user=self.build_user_prompt(state),
            model=self.config.model_planner,
            n=self.config.n_samples,
            max_tokens=self.config.max_tokens_planner,
            temperature=self.config.temperature_planner,
            max_workers=self.config.max_workers_sample,
        )
        return state

    def trace_detail(self, state: AgentState) -> dict:
        # Distinct plan count is the honest measure of whether n-sampling is
        # doing anything: if it is 1, the vote has nothing to decide.
        return {
            "n_plans": len(state.raw_plans),
            "n_distinct_plans": len(set(state.raw_plans)),
        }


# =============================================================================
# NODE 4 -- Equation Extractor   (§4.4, eqs. 4 and 5, Figure 1)
# =============================================================================
#
# Turns n raw plans into one answer:
#
#     parse -> transpile -> validate -> execute -> vote -> p*   then a* = Execute(p*)
#
# Two things here are this implementation's, not the paper's, and both are
# flagged in the trace so their effect can be measured rather than assumed:
#
# * Candidates that cannot be parsed, transpiled, or executed are dropped before
#   the vote (`drop_invalid_candidates`). Voting over programs that cannot run
#   would let a malformed cluster outvote a working one.
# * If nothing survives, the sample degrades to the repo's shared direct prompt
#   (`use_direct_prompt_fallback`) rather than emitting an empty string, which
#   would be a guaranteed zero on both metrics.


class EquationExtractor(Node):
    name = "equation_extractor"

    def __init__(self, client: LLMClient, config: AgentConfig):
        self.client = client
        self.config = config

    # ------------------------------------------------------ candidate build --
    def _build_candidate(self, index: int, raw_plan: str,
                         state: AgentState) -> Candidate:
        candidate = Candidate(index=index, raw_plan=raw_plan)

        try:
            plan = parse_plan(raw_plan)
        except PlanParseError as exc:
            candidate.error = f"parse: {exc}"
            return candidate
        candidate.plan = plan
        try:
            candidate.parallelism = parallelism(plan)
        except PlanParseError:
            candidate.parallelism = 0.0

        try:
            result = transpile(plan)
        except (TranspileError, PlanParseError) as exc:
            candidate.error = f"transpile: {exc}"
            return candidate
        candidate.program = result.program
        candidate.warnings = result.warnings

        if self.config.validate_row_labels:
            missing = self._unknown_row_labels(result.program, state)
            if missing:
                candidate.error = f"unknown table row(s): {sorted(missing)}"
                return candidate

        ok, value = execute_program(result.program, state.table_raw)
        candidate.executable = ok
        candidate.exe_result = value
        if not ok:
            candidate.error = "execution returned n/a"
        return candidate

    @staticmethod
    def _unknown_row_labels(program: str, state: AgentState) -> set[str]:
        """Row names the program looks up that this sample's table lacks.

        `eval_program` resolves a row by exact first-cell match, so a label that
        is not there makes the whole program execute to "n/a". Catching it here
        turns a silent zero into a named, countable failure.
        """
        try:
            steps = steps_from_tokens(program_tokenization(program))
        except ValueError:
            return set()
        known = table_row_labels(state.table_raw)
        return {
            arg1 for op, arg1, _ in steps
            if op in TABLE_OPS and not arg1.startswith("#") and arg1 not in known
        }

    # ----------------------------------------------------------- fallbacks --
    def _direct_prompt(self, state: AgentState) -> str:
        user = prompts.USER_MESSAGE_FRAME.format(
            pre_text=state.pre_text,
            table=state.table,
            post_text=state.post_text,
            question=state.question,
        )
        raw = self.client.complete(
            system=prompts.SYSTEM_MESSAGE,
            user=user,
            model=self.config.model_fallback,
            max_tokens=self.config.max_tokens_fallback,
            temperature=0.0,
        )
        program = extract_program(raw)
        # `extract_program` returns its input unchanged when it finds no
        # operator call, so a model that answers in prose would otherwise land a
        # paragraph of Vietnamese in the `generated_program` column. Anything
        # the grader's own tokeniser cannot read is not a program.
        try:
            program_tokenization(program)
        except ValueError:
            return ""
        return program

    # -------------------------------------------------------------- execute --
    def execute(self, state: AgentState) -> AgentState:
        state.candidates = [
            self._build_candidate(index, raw, state)
            for index, raw in enumerate(state.raw_plans)
        ]

        pool = (
            state.usable_candidates
            if self.config.drop_invalid_candidates
            else [c for c in state.candidates if c.program is not None]
        )

        if pool:
            result = vote([c.program for c in pool], mode=self.config.vote_mode)
            state.vote = result
            if result.winner is not None:
                state.program = result.winner
                ok, value = execute_program(result.winner, state.table_raw)
                state.answer = value if ok else None
                return state

        # Nothing usable came out of the agent pipeline.
        if self.config.use_direct_prompt_fallback:
            try:
                program = self._direct_prompt(state)
            except LLMError as exc:
                state.errors.append(f"{self.name}: fallback failed: {exc}")
                return state
            state.fallback = "direct_prompt"
            state.program = program
            ok, value = execute_program(program, state.table_raw)
            state.answer = value if ok else None
        else:
            state.errors.append(f"{self.name}: no usable candidate and no fallback")
        return state

    def trace_detail(self, state: AgentState) -> dict:
        detail = {
            "n_usable": len(state.usable_candidates),
            "fallback": state.fallback,
        }
        if state.vote is not None:
            detail["n_clusters"] = state.vote.n_clusters
            detail["consensus"] = round(state.vote.consensus, 4)
        return detail


# =============================================================================
# THE PIPELINE GRAPH -- the first of the paper's two DAGs
# =============================================================================
#
#     "Our pipeline models the reasoning process as a directed acyclic graph,
#      ensuring a structured and transparent execution flow." (section 4)
#
# Nodes declare their dependencies and the graph resolves an execution order,
# rather than the four stages being hard-wired as a function call chain. That is
# not ceremony: the ablations in Table 4 are edge changes, and an explicit graph
# makes them a configuration rather than a second code path.
#
# The second DAG -- the plan itself, whose independent branches the planner
# prompt asks to parallelise -- lives in `program.PlanGraph.levels()`.


class GraphError(ValueError):
    pass


@dataclass
class AgentGraph:
    nodes: dict[str, Node] = field(default_factory=dict)
    edges: dict[str, tuple[str, ...]] = field(default_factory=dict)

    def add(self, node: Node, depends_on: tuple[str, ...] = ()) -> "AgentGraph":
        if node.name in self.nodes:
            raise GraphError(f"duplicate node name {node.name!r}")
        for dependency in depends_on:
            if dependency not in self.nodes:
                raise GraphError(
                    f"{node.name!r} depends on unknown node {dependency!r}"
                )
        self.nodes[node.name] = node
        self.edges[node.name] = depends_on
        return self

    def order(self) -> list[Node]:
        """Topological order, ties broken by insertion order."""
        resolved: list[str] = []
        visiting: set[str] = set()

        def visit(name: str) -> None:
            if name in resolved:
                return
            if name in visiting:
                raise GraphError(f"cycle in pipeline graph at {name!r}")
            visiting.add(name)
            for dependency in self.edges[name]:
                visit(dependency)
            visiting.discard(name)
            resolved.append(name)

        for name in self.nodes:
            visit(name)
        return [self.nodes[name] for name in resolved]

    def run(self, state: AgentState) -> AgentState:
        """Run every node in order. A node that raises is traced, not fatal."""
        for node in self.order():
            state = node.run(state)
        return state

    def describe(self) -> str:
        lines = ["pipeline graph:"]
        for node in self.order():
            deps = self.edges[node.name]
            arrow = f" <- {', '.join(deps)}" if deps else ""
            lines.append(f"  {node.name}{arrow}")
        return "\n".join(lines)


def build_default_graph(client: LLMClient, config: AgentConfig) -> AgentGraph:
    """The four-node pipeline of Figure 1.

    When `use_decomposition` is False the two extraction nodes stay in the graph
    but no-op (the generator returns early, the answerer has nothing to answer),
    so the "Multi-Path Only" ablation keeps the same trace shape as a full run
    and the two are directly comparable.
    """
    graph = AgentGraph()
    graph.add(SubqueryGenerator(client, config))
    graph.add(SubqueryAnswerer(client, config), depends_on=("subquery_generator",))
    graph.add(Planner(client, config), depends_on=("subquery_answerer",))
    graph.add(EquationExtractor(client, config), depends_on=("planner",))
    return graph
