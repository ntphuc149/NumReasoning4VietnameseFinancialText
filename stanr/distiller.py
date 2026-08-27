"""ReasoningDistiller: calls a teacher model to build a reasoning-trace
training set, following the CoNR protocol (Section 3.3 of the paper).

Pattern copied from
`notebooks/vinumqa/distill-reasoning-trace/gemma-4-31b-conr-trace-gen-independent-solve-en.ipynb`:
an OpenAI-compatible API call per sample (parallel, with retries and a
resumable checkpoint), asking the teacher to derive `<think>` + `<program>`
independently -- no gold program in the prompt. Only samples whose derived
program is verified against gold are kept; `qa.program`/`qa.exe_ans` in the
output are always the untouched gold values, never the teacher's own.
"""

from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from tqdm import tqdm

from stanr import data, scoring
from stanr.prompts import CONR_SYSTEM_PROMPT, CONR_USER_FRAME

FilterMode = Literal["pa", "strict", "union"]
_VALID_FILTERS = ("pa", "strict", "union")


def _normalize_token(token: str) -> str:
    """Numeric tokens are normalised (drop $, commas, trailing %, round to
    6dp) so "100" == "100.00" and "15.1%" == "0.151"; everything else
    (operator names, `#N` references, `none`, row labels) is left as-is."""
    token = token.strip()
    if token in ("(", ")") or token.endswith("(") or token.startswith("#") or token.lower() == "none":
        return token
    cleaned = token.replace(",", "").replace("$", "").strip()
    is_percent = cleaned.endswith("%")
    if is_percent:
        cleaned = cleaned[:-1].strip()
    try:
        value = float(cleaned)
        if is_percent:
            value /= 100.0
        return f"{round(value, 6)}"
    except ValueError:
        return token


def _is_strict_match(generated_program: str, gold_program: str) -> bool:
    """Exact structural match: same operators, same arguments, same order.

    Stricter than `scoring.equal_program` (which sympy-simplifies and so
    tolerates commuted arguments / reordered steps) -- if the teacher
    reformatted or reordered gold, `is_pa_match` still accepts it but this
    does not.
    """
    try:
        gen_tokens = scoring.program_tokenization(scoring.extract_program(generated_program))
        gold_tokens = scoring.program_tokenization(gold_program)
    except ValueError:
        return False
    return [_normalize_token(t) for t in gen_tokens] == [_normalize_token(t) for t in gold_tokens]


def _is_pa_match(generated_program: str, gold_program: str) -> bool:
    try:
        gen_tokens = scoring.program_tokenization(scoring.extract_program(generated_program))
        gold_tokens = scoring.program_tokenization(gold_program)
        return scoring.equal_program(gold_tokens, gen_tokens)
    except Exception:
        return False


def _is_verified(generated_program: str, gold_program: str, filter_with: FilterMode) -> bool:
    if filter_with == "pa":
        return _is_pa_match(generated_program, gold_program)
    if filter_with == "strict":
        return _is_strict_match(generated_program, gold_program)
    return _is_pa_match(generated_program, gold_program) or _is_strict_match(generated_program, gold_program)


class ReasoningDistiller:
    """Distill reasoning traces for every sample in `input_dataset`, keeping
    only the ones whose derived program is verified against gold.

    Parameters
    ----------
    model_name, api_key, base_url:
        Teacher model served behind an OpenAI-compatible `/chat/completions`
        endpoint (this repo's own teacher, gemma-4-31B-it, is served this
        way -- see `.env`'s `API_KEY`/`BASE_URL`).
    input_dataset:
        Path to a ViNumQA/FinQA-style JSON file (list of
        pre_text/table/post_text/id/qa samples).
    output_distill_path:
        Where the filtered, trace-augmented samples are written. Only
        verified samples are included -- this is not a copy of the input
        with gaps, it is the smaller, ready-to-train-on subset.
    filter_with:
        "pa" (default) -- symbolic program-accuracy match only (tolerates
        reordered/commuted steps).
        "strict" -- exact structural match only (same step order).
        "union" -- either test accepts.
    n_samples:
        Independent attempts per question before giving up on it (stops
        early at the first verified attempt). The paper's own distillation
        run used 1.
    """

    def __init__(
        self,
        model_name: str,
        api_key: str,
        input_dataset: str,
        output_distill_path: str,
        base_url: str = "https://mkp-api.fptcloud.com/v1",
        filter_with: FilterMode = "pa",
        conr_system_prompt: str = CONR_SYSTEM_PROMPT,
        conr_user_frame: str = CONR_USER_FRAME,
        n_samples: int = 1,
        n_workers: int = 8,
        max_tokens: int = 8192,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_retries: int = 3,
        checkpoint_path: str | None = None,
    ):
        if filter_with not in _VALID_FILTERS:
            raise ValueError(f"filter_with must be one of {_VALID_FILTERS}, got {filter_with!r}")

        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        self.input_dataset = Path(input_dataset)
        self.output_distill_path = Path(output_distill_path)
        self.filter_with = filter_with
        self.conr_system_prompt = conr_system_prompt
        self.conr_user_frame = conr_user_frame
        self.n_samples = n_samples
        self.n_workers = n_workers
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.max_retries = max_retries
        self.checkpoint_path = Path(checkpoint_path) if checkpoint_path else (
            self.output_distill_path.with_suffix(".checkpoint.json")
        )

    def _client(self):
        from openai import OpenAI
        return OpenAI(api_key=self.api_key, base_url=self.base_url)

    def _call_teacher_once(self, client, sample: dict) -> str:
        messages = [
            {"role": "system", "content": self.conr_system_prompt},
            {"role": "user", "content": data.build_user_message(self.conr_user_frame, sample)},
        ]
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=self.temperature,
                    top_p=self.top_p,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content
                if text and text.strip():
                    return text.strip()
            except Exception as exc:  # network/API errors -- retry, don't crash the whole run
                if attempt == self.max_retries:
                    print(f"[teacher call failed after {self.max_retries} tries] {exc}")
                    return ""
            if attempt < self.max_retries:
                time.sleep(2 * attempt)
        return ""

    def run(self) -> dict:
        """Run distillation end-to-end: call the teacher, verify, filter,
        write `output_distill_path`. Returns a summary dict."""
        samples = data.load_json(self.input_dataset)

        texts_by_id: dict[str, list[str]] = {}
        if self.checkpoint_path.exists():
            texts_by_id = {k: v for k, v in data.load_json(self.checkpoint_path).items()}
            print(f"Resumed {len(texts_by_id)} samples' attempts from {self.checkpoint_path}")

        def already_verified(sample_id: str) -> bool:
            texts = texts_by_id.get(sample_id, [])
            gold = id_to_sample[sample_id]["qa"]["program"]
            return any(_is_verified(data.extract_think_and_program(t)[1], gold, self.filter_with) for t in texts)

        id_to_sample = {s["id"]: s for s in samples}

        work = []
        for s in samples:
            have = len(texts_by_id.get(s["id"], []))
            if already_verified(s["id"]):
                continue
            work.extend([s] * max(0, self.n_samples - have))

        print(f"{len(work)} new teacher calls needed across "
              f"{len({w['id'] for w in work})} samples ({len(samples) - len({w['id'] for w in work})} "
              f"already covered).")

        if work:
            client = self._client()
            with ThreadPoolExecutor(max_workers=self.n_workers) as pool:
                futures = {pool.submit(self._call_teacher_once, client, s): s["id"] for s in work}
                completed = 0
                for future in tqdm(as_completed(futures), total=len(futures), desc="Distilling traces..."):
                    sid = futures[future]
                    texts_by_id.setdefault(sid, []).append(future.result())
                    completed += 1
                    if completed % 50 == 0:
                        data.save_json(texts_by_id, self.checkpoint_path)
            data.save_json(texts_by_id, self.checkpoint_path)

        results = []
        n_verified = 0
        for s in samples:
            gold_program = s["qa"]["program"]
            texts = texts_by_id.get(s["id"], [])
            chosen_trace, chosen_program, verified = "", "", False
            for text in texts:
                trace, program = data.extract_think_and_program(text)
                if _is_verified(program, gold_program, self.filter_with):
                    chosen_trace, chosen_program, verified = trace, program, True
                    break
            if verified:
                n_verified += 1
                results.append(data.with_reasoning_trace(s, chosen_trace, "independent"))

        data.save_json(results, self.output_distill_path)

        summary = {
            "total": len(samples),
            "verified": n_verified,
            "yield": n_verified / len(samples) if samples else 0.0,
            "filter_with": self.filter_with,
            "output_path": str(self.output_distill_path),
        }
        print(f"Verified {n_verified}/{len(samples)} ({100 * summary['yield']:.1f}%) "
              f"-> {self.output_distill_path}")
        return summary
