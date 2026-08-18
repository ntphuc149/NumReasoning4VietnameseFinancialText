"""MPR-Agent: a graph-based multi-agent pipeline for ViNumQA.

An implementation of Nguyen et al., "A Graph-Based Agent Approach to Numerical
Reasoning Question Answering" (VLSP 2025, aclanthology.org/2025.vlsp-1.29),
targeting this repository's ViNumQA program format and shared PA/EA scorer.

    q, C ──▶ [1] SubqueryGenerator   G_sq(q, C)        -> SQ = {sq_1..sq_k}
                 [2] SubqueryAnswerer  A_sq(sq_j, C)   -> V  = {v_1..v_k}
                     [3] Planner        P_n-sample(...) -> n candidate plans
                         [4] EquationExtractor  vote -> p* -> a*

Nine modules, one job each:

    config    AgentConfig / RunConfig -- every knob, marked paper's or ours
    scoring   the ONE bridge to notebooks/evaluate/scorer.py (never a copy)
    prompts   Appendix B verbatim (VI + EN), opt-in patches, fallback prompt
    llm       context formatting + the OpenAI-compatible transport
    backends  routes each model_* name to the transport that actually serves
              it -- local transformers.generate() for the 3 models this repo
              trains, the API client for the other 8 baseline models
    program   plan DSL -> PlanGraph -> ViNumQA program -> vote
    agents    AgentState, the four nodes, the pipeline graph
    runner    batch run, checkpoint/resume, scoring, oracle@n, offline re-vote

See the README beside this package for the design, the seven transpilation
rules, and which behaviours are the paper's versus this implementation's.
"""

from agentic.agents import (
    AgentGraph,
    AgentState,
    Candidate,
    EquationExtractor,
    Node,
    NodeTrace,
    Planner,
    SubqueryAnswerer,
    SubqueryGenerator,
    build_default_graph,
)
from agentic.backends import (
    API_MODELS,
    MODEL_REGISTRY,
    LocalBackend,
    MultiModelClient,
    describe_backend,
)
from agentic.config import AgentConfig, RunConfig
from agentic.llm import LLMClient, LLMError, RateLimiter
from agentic.program import (
    PlanGraph,
    PlanParseError,
    TranspileError,
    canonicalize,
    parse_plan,
    plan_text_to_program,
    transpile,
    vote,
)
from agentic.runner import Runner, candidate_diagnostics, load_dataset, revote

__all__ = [
    "API_MODELS",
    "AgentConfig",
    "AgentGraph",
    "AgentState",
    "Candidate",
    "EquationExtractor",
    "LLMClient",
    "LLMError",
    "LocalBackend",
    "MODEL_REGISTRY",
    "MultiModelClient",
    "Node",
    "NodeTrace",
    "PlanGraph",
    "PlanParseError",
    "Planner",
    "RateLimiter",
    "RunConfig",
    "Runner",
    "SubqueryAnswerer",
    "SubqueryGenerator",
    "TranspileError",
    "build_default_graph",
    "canonicalize",
    "candidate_diagnostics",
    "describe_backend",
    "load_dataset",
    "parse_plan",
    "plan_text_to_program",
    "revote",
    "transpile",
    "vote",
]
