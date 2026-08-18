"""Batch runner: dataset in, scored predictions and full traces out.

Checkpointing is per sample and on by default. A full n=15 run over test.json is
thousands of requests; losing it to an interruption at sample 400 is not an
acceptable failure mode, and this repo has already been bitten by exactly that
(the GLM-5.2 few-shot run died at 73/497 on a 429).

The output DataFrame uses the same column names every other notebook in this
repo feeds to `scorer.evaluate_dataframe`, so scoring an agent run is the same
call as scoring a 0-shot run.
"""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import pandas as pd

from agentic.agents import AgentGraph, AgentState, build_default_graph
from agentic.backends import MultiModelClient
from agentic.config import AgentConfig, RunConfig
from agentic.llm import (
    format_context,
    format_post_text,
    format_pre_text,
    format_table,
)
from agentic.program import vote
from agentic.scoring import evaluate_dataframe, find_project_root, score_one


def load_dataset(path: str | Path, limit: Optional[int] = None) -> list[dict]:
    """Load a ViNumQA split, resolving a relative path against the repo root."""
    candidate = Path(path)
    if not candidate.is_absolute() and not candidate.exists():
        candidate = find_project_root() / path
    with open(candidate, encoding="utf-8") as handle:
        data = json.load(handle)
    return data[:limit] if limit else data


def build_state(sample: dict, lang: str = "vi") -> AgentState:
    return AgentState(
        sample_id=str(sample.get("id", "")),
        question=sample["qa"]["question"],
        context=format_context(sample, lang=lang),
        pre_text=format_pre_text(sample),
        table=format_table(sample),
        post_text=format_post_text(sample),
        table_raw=sample.get("table"),
    )


class Runner:
    def __init__(self, run_config: RunConfig, client: Optional[MultiModelClient] = None,
                 graph: Optional[AgentGraph] = None):
        self.run_config = run_config
        self.config: AgentConfig = run_config.agent
        # MultiModelClient, not LLMClient directly: it routes each of the four
        # model_* fields to whichever backend actually serves that name (see
        # backends.py) -- local transformers.generate() for the three models
        # this repo trains, the API client for everything else -- so a Runner
        # never has to know or care which of the eleven baseline models it
        # was configured with.
        self.client = client or MultiModelClient(self.config)
        self.graph = graph or build_default_graph(self.client, self.config)
        self._lock = threading.Lock()
        self._results: dict[str, dict] = {}

        root = find_project_root()
        out_dir = Path(run_config.output_dir)
        if not out_dir.is_absolute():
            out_dir = root / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = out_dir
        self.checkpoint_path = out_dir / f"{run_config.run_name}_traces.json"

    # ------------------------------------------------------------ checkpoint --
    def _load_checkpoint(self) -> None:
        if not (self.run_config.resume and self.checkpoint_path.exists()):
            return
        try:
            with open(self.checkpoint_path, encoding="utf-8") as handle:
                stored = json.load(handle)
        except (json.JSONDecodeError, OSError):
            return
        self._results = {str(item["id"]): item for item in stored}
        print(f"Resumed {len(self._results)} sample(s) from {self.checkpoint_path}")

    def _save_checkpoint(self) -> None:
        payload = list(self._results.values())
        tmp = self.checkpoint_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=1)
        tmp.replace(self.checkpoint_path)

    # ------------------------------------------------------------------ run --
    def run(self, samples: Optional[Iterable[dict]] = None,
            on_result: Optional[Callable[[dict], None]] = None,
            show_progress: bool = False) -> pd.DataFrame:
        """Run every sample not already in the checkpoint.

        `show_progress=True` prints how far the run has actually gotten as it
        goes -- a `tqdm` bar if it is importable, a periodic count otherwise
        -- instead of the previous silent block-until-done. Measured need:
        this pipeline's own per-sample time varies by more than an order of
        magnitude on a reasoning model (6s to 220s for one node alone, seen
        on a real DeepSeek-V4-Flash smoke run), so "no output yet" and "done
        in a minute" are not distinguishable without this.
        """
        data = list(samples) if samples is not None else load_dataset(
            self.run_config.dataset_path, self.run_config.limit
        )
        self._load_checkpoint()

        todo = [s for s in data if str(s.get("id", "")) not in self._results]
        print(f"{len(data)} sample(s) total, {len(todo)} to run.")

        progress = None
        if show_progress and todo:
            try:
                from tqdm import tqdm
                progress = tqdm(total=len(todo), desc=self.run_config.run_name, unit="sample")
            except ImportError:
                print("show_progress=True but tqdm is not installed -- "
                      "install it for a live bar; falling back to periodic counts.")

        done_since_save = 0
        n_errors = 0
        if todo:
            workers = min(self.config.max_workers_dataset, len(todo))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {
                    pool.submit(self._run_one, sample): sample for sample in todo
                }
                for future in as_completed(futures):
                    sample = futures[future]
                    try:
                        record = future.result()
                    except Exception as exc:  # noqa: BLE001
                        record = {
                            "id": str(sample.get("id", "")),
                            "program": "",
                            "answer": None,
                            "errors": [f"runner: {type(exc).__name__}: {exc}"],
                        }
                    if record.get("errors"):
                        n_errors += 1
                    with self._lock:
                        self._results[record["id"]] = record
                        done_since_save += 1
                        if done_since_save >= self.run_config.checkpoint_every:
                            self._save_checkpoint()
                            done_since_save = 0
                    done_total = len(self._results)
                    if progress is not None:
                        progress.set_postfix(errors=n_errors)
                        progress.update(1)
                    elif show_progress and (
                        done_total % 5 == 0 or done_total == len(data)
                    ):
                        # Periodic, not per-sample: 497 print lines would be its
                        # own kind of noise. Every 5th completion plus the last.
                        print(f"  {done_total}/{len(data)} done ({n_errors} with errors so far)")
                    if on_result is not None:
                        on_result(record)
            self._save_checkpoint()
            if progress is not None:
                progress.close()

        return self.to_dataframe(data)

    def _run_one(self, sample: dict) -> dict:
        state = build_state(sample, lang=self.config.prompt_lang)
        state = self.graph.run(state)
        return state.to_dict(keep_candidates=self.config.keep_all_candidates)

    # -------------------------------------------------------------- outputs --
    def to_dataframe(self, samples: list[dict]) -> pd.DataFrame:
        """Columns match what `scorer.evaluate_dataframe` expects by default."""
        rows = []
        for sample in samples:
            sample_id = str(sample.get("id", ""))
            record = self._results.get(sample_id, {})
            qa = sample.get("qa", {})
            rows.append(
                {
                    "id": sample_id,
                    "question": qa.get("question", ""),
                    "pre_text": format_pre_text(sample),
                    "table": format_table(sample),
                    "table_raw": sample.get("table"),
                    "post_text": format_post_text(sample),
                    "program": qa.get("program", ""),
                    "answer": qa.get("exe_ans", ""),
                    "generated_program": record.get("program", ""),
                    "generated_answer": record.get("answer"),
                    "fallback": record.get("fallback"),
                    "n_usable": record.get("n_usable", 0),
                    "consensus": (record.get("vote") or {}).get("consensus"),
                    "n_clusters": (record.get("vote") or {}).get("n_clusters"),
                }
            )
        return pd.DataFrame(rows)

    def score(self, df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
        """PA/EA via the shared scorer, plus this pipeline's own diagnostics."""
        scored, summary = evaluate_dataframe(
            df,
            generated_col="generated_program",
            gold_program_col="program",
            gold_answer_col="answer",
            table_col="table_raw",
        )
        summary = dict(summary)
        summary["n"] = len(scored)
        summary["fallback_rate"] = float(scored["fallback"].notna().mean())
        summary["empty_rate"] = float((scored["generated_program"] == "").mean())
        summary["mean_usable_candidates"] = float(scored["n_usable"].mean())
        if scored["consensus"].notna().any():
            summary["mean_consensus"] = float(scored["consensus"].mean())
        summary.update(self.oracle(df))
        summary["usage"] = self.client.usage.as_dict()
        return scored, summary

    def oracle(self, df: pd.DataFrame) -> dict:
        """Upper bound if selection were perfect: oracle@n for PA and EA.

        This is the diagnostic the paper's section 5.5.2 needs but does not
        report. `oracle_pa - program_accuracy` is what majority voting threw
        away (its "heuristic selection error"); `1 - oracle_pa` is what the base
        model never generated at all (its "systematic reasoning error").
        """
        hits_pa = hits_ea = considered = 0
        for _, row in df.iterrows():
            record = self._results.get(str(row["id"]))
            if not record:
                continue
            candidates = record.get("candidates") or []
            programs = [c["program"] for c in candidates if c.get("program")]
            if not programs:
                continue
            considered += 1
            best_pa = best_ea = 0.0
            for program in programs:
                try:
                    pa, ea = score_one(
                        program, row["program"], row["answer"],
                        row["table_raw"], extract_first=False,
                    )
                except Exception:  # noqa: BLE001 - a bad gold row must not abort
                    continue
                best_pa = max(best_pa, pa)
                best_ea = max(best_ea, ea)
            hits_pa += best_pa
            hits_ea += best_ea
        total = len(df) or 1
        return {
            "oracle_pa": hits_pa / total,
            "oracle_ea": hits_ea / total,
            "oracle_coverage": considered / total,
        }

    def save(self, scored: pd.DataFrame, summary: dict) -> tuple[Path, Path]:
        name = self.run_config.run_name
        results_path = self.output_dir / f"{name}_results.csv"
        summary_path = self.output_dir / f"{name}_summary.json"
        scored.drop(columns=["table_raw"]).to_csv(results_path, index=False)
        with open(summary_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "run_name": name,
                    "dataset": self.run_config.dataset_path,
                    "config": asdict(self.config),
                    **summary,
                },
                handle,
                ensure_ascii=False,
                indent=2,
            )
        return results_path, summary_path


# =============================================================================
# OFFLINE ANALYSIS -- works from a saved trace file, no API calls
# =============================================================================


def load_traces(traces_path: str | Path) -> list[dict]:
    with open(traces_path, encoding="utf-8") as handle:
        return json.load(handle)


def candidate_diagnostics(traces_path: str | Path) -> pd.DataFrame:
    """Per-candidate failure breakdown from a saved trace file.

    Answers the question the smoke run exists to answer: of n*samples generated
    plans, how many failed at parse, at transpile, at row lookup, at execution.
    """
    rows = []
    for record in load_traces(traces_path):
        for candidate in record.get("candidates", []):
            error = candidate.get("error") or ""
            if not error:
                stage = "ok"
            elif error.startswith("parse:"):
                stage = "parse"
            elif error.startswith("transpile:"):
                stage = "transpile"
            elif error.startswith("unknown table row"):
                stage = "row_lookup"
            else:
                stage = "execute"
            rows.append(
                {
                    "id": record["id"],
                    "index": candidate["index"],
                    "stage": stage,
                    "error": error,
                    "program": candidate.get("program"),
                    "parallelism": candidate.get("parallelism"),
                }
            )
    return pd.DataFrame(rows)


def revote(traces_path: str | Path, samples: list[dict],
           mode: str = "canonical") -> pd.DataFrame:
    """Re-run the selection stage from saved traces under a different vote mode.

    Because every candidate is kept (`keep_all_candidates`), changing how the
    winner is chosen costs nothing: no model is called, the n plans are already
    on disk. Use this to compare vote modes on an existing run rather than
    paying for the pipeline again.
    """
    by_id = {str(r["id"]): r for r in load_traces(traces_path)}
    rows = []
    for sample in samples:
        sample_id = str(sample.get("id", ""))
        record = by_id.get(sample_id, {})
        qa = sample.get("qa", {})
        usable = [
            c for c in record.get("candidates", [])
            if c.get("program") and c.get("executable")
        ]
        result = vote([c["program"] for c in usable], mode=mode) if usable else None
        rows.append(
            {
                "id": sample_id,
                "question": qa.get("question", ""),
                "table_raw": sample.get("table"),
                "program": qa.get("program", ""),
                "answer": qa.get("exe_ans", ""),
                # Fall back to whatever the original run emitted (including its
                # direct-prompt fallback) when there is nothing to re-vote on.
                "generated_program": (
                    result.winner if result and result.winner
                    else record.get("program", "")
                ),
                "fallback": record.get("fallback"),
                "n_usable": len(usable),
                "consensus": result.consensus if result else None,
                "n_clusters": result.n_clusters if result else None,
            }
        )
    return pd.DataFrame(rows)


def score_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """PA/EA for any frame with the standard columns -- no Runner needed."""
    scored, summary = evaluate_dataframe(
        df,
        generated_col="generated_program",
        gold_program_col="program",
        gold_answer_col="answer",
        table_col="table_raw",
    )
    summary = dict(summary)
    summary["n"] = len(scored)
    return scored, summary


__all__ = [
    "Runner",
    "build_state",
    "candidate_diagnostics",
    "load_dataset",
    "load_traces",
    "revote",
    "score_frame",
]
