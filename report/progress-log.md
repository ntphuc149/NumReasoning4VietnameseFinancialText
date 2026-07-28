# Progress Log — VSF-Fintech VLSP 2025 Numerical Reasoning

Daily work log for the Numerical Reasoning QA project on Vietnamese financial
text (VLSP 2025 NumQA shared task).

---

## 16/7/2026

Explored problem formulations and datasets related to fintech.

## 17/7/2026

Read the following papers:

1. Zhiwei Liu, Keyi Wang, Zhuo Bao, Xin Zhang, Jiping Dong, Kailai Yang, Mohsinul Kabir, Polydoros Giannouris, Rui Xing, Seongchan Park, Jaehong Kim, Dong Li, Qianqian Xie, and Sophia Ananiadou. 2025. FinNLP-FNP-LLMFinLegal-2025 Shared Task: Financial Misinformation Detection Challenge Task. In Proceedings of the Joint Workshop of the 9th Financial Technology and Natural Language Processing (FinNLP), the 6th Financial Narrative Processing (FNP), and the 1st Workshop on Large Language Models for Finance and Legal (LLMFinLegal), pages 271–276, Abu Dhabi, UAE. Association for Computational Linguistics.
   *(Workshop co-located with **COLING 2025** — CORE rank **B**, per the CORE 2023 ranking.)*
2. Zhiwei Liu, Xin Zhang, Kailai Yang, Qianqian Xie, Jimin Huang, and Sophia Ananiadou. 2025. FMDLlama: Financial Misinformation Detection Based on Large Language Models. In Companion Proceedings of the ACM on Web Conference 2025 (WWW '25). Association for Computing Machinery, New York, NY, USA, 1153–1157. https://doi.org/10.1145/3701716.3715599
   *(**WWW (The Web Conference)** — CORE rank **A\*** (flagship conference).)*
3. Yuechen Jiang, Zhiwei Liu, Yupeng Cao, Yueru He, Ziyang Xu, Chen Xu, Zhiyang Deng, Prayag Tiwari, Xi Chen, Alejandro Lopez-Lira, Jimin Huang, Junichi Tsujii, and Sophia Ananiadou. 2026. All That Glisters Is Not Gold: A Benchmark for Reference-Free Counterfactual Financial Misinformation Detection. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 10737–10776, San Diego, California, United States. Association for Computational Linguistics.
   *(**ACL (Annual Meeting)**, main conference — CORE rank **A\*** (flagship conference).)*
4. Le Ngoc Toan, Ha My Linh, Pham Thi Duc, Ngo The Quyen, and Nguyen Thi Minh Huyen. 2025. VLSP 2025 challenge: Numerical Reasoning Question and Answer. In Proceedings of the 11th International Workshop on Vietnamese Language and Speech Processing, pages 187–196, Hanoi, Vietnam. Association for Computational Linguistics.
   *(VLSP is a Vietnamese-specific workshop with no standalone CORE rank — selected because it is the paper that directly describes the NumQA/ViNumQA shared task this project is built on, not for venue prestige.)*

**Why these four papers**: the first three (FinNLP-FNP-LLMFinLegal, WWW, ACL) span a range of venue tiers (B → A\* → A\*) on the related problem of financial misinformation detection, giving coverage of both a specialized workshop and top-tier general venues working on NLP-for-finance; the fourth is the direct source of the task/dataset (ViNumQA) this project builds on.

## 20/7/2026

Read the following paper and collected the ViNumQA dataset; ran the 0-shot experiment:

- Duc Dinh Chu, Thanh-Bac Nguyen Ba, Duy Dinh Le, and Khanh Van Tran. 2025. Enhancing Numerical Reasoning in Vietnamese Financial Question Answering through Program-Centric Policy Optimization. In Proceedings of the 11th International Workshop on Vietnamese Language and Speech Processing, pages 197–204, Hanoi, Vietnam. Association for Computational Linguistics.

## 21/7/2026

Ran the 1-shot experiment.

## 22/7/2026

Ran the few-shot experiment.

## 23/7/2026

Wrote and refined the prompt for reasoning-trace knowledge distillation from the teacher model.

## 24/7/2026

Generated reasoning traces from the teacher model using the refined instruction prompt.
Ran SFT without/with reasoning trace from the teacher model.

## 27/7/2026

Got results for SFT with reasoning-trace distillation (PA 0.6258, EA 0.6338), still below
SFT without reasoning trace (PA 0.6419, EA 0.6439). Investigated possible causes: repetitive
generation loops observed at inference, and a hypothesis that the distilled reasoning traces
being in Vietnamese hurts performance since Qwen3's instruction-tuning is predominantly
English. To test the language hypothesis cheaply before committing to re-distilling from the
teacher model, built a notebook to machine-translate the existing Vietnamese reasoning traces
to English (VietAI/envit5-translation), with sentence-level translation and a number-recovery
step (splicing original numeric tokens back into the translation, with a regex-based spacing
cleanup fallback) to avoid corrupting the numbers embedded in each trace. Also tuned inference
decoding parameters (temperature/top-k/top-p) for the reasoning-trace SFT model to reduce
repetition. Next: fine-tune Qwen3-4B on the translated (English) reasoning traces and compare
PA/EA against the Vietnamese-trace run — if it improves, re-distill directly from
Qwen3-Next-80B-A3B-Thinking with an English-output prompt instead of relying on MT.

## 28/7/2026

Found and fixed a scoring bug affecting every notebook: ViNumQA gold programs use
`table_sum/table_average/table_max/table_min(row_name, none)`, where the first
argument is a table row *name* to look up — not a list of numeric values, as every
notebook's SYSTEM_MESSAGE incorrectly described and every hand-rolled evaluator
assumed. This silently scored EA=0 on any sample using a table_* op (~12% of
test.json) and, more importantly, meant every 0/1/few-shot prompted model was being
taught the wrong output format directly in the prompt. Built a shared evaluator
(`notebooks/evaluate/scorer.py`) porting FinQA's official evaluation protocol (Chen
et al. 2021) — sympy-based symbolic Program Accuracy, table-row-lookup-aware
Execution Accuracy — with five corrections justified by measurement against this
dataset (bracket-depth-aware tokenization for row labels like "ROE (%)", the pure
"(3344)" accounting-negative form, skipping missing/unparseable table cells instead
of voiding the whole row, coercing exe_ans to float, dropping a debug assert that
aborted scoring on rounding disagreements) and a fix for silent truncation (a
generated program cut off mid-step must score invalid, not be evaluated on whatever
steps happened to parse). Applied the fix (corrected SYSTEM_MESSAGE, added a
table_raw column so the evaluator can see the real table, swapped in scorer.py)
across all 0-shot/1-shot/few-shot notebooks and both SFT notebooks. Added the two
missing 1-shot notebooks (FPT API and Qwen3-4B/Kaggle) to match the existing
0-shot/few-shot coverage.

Re-ran two notebooks end-to-end against the fixed evaluator: 0-shot DeepSeek-V4-Flash
scored PA 14.69% / EA 18.71%; 1-shot gpt-5-nano (medium effort, Batch API) scored
PA 28.37% / EA 31.39%. Also tuned SFT-without-reasoning-trace eval-time decoding
(temperature=0.7, top_p=0.8, top_k=20, min_p=0, matching the training generation
config) and wired the SFT-with-reasoning-trace notebook up to the same shared
scorer so all methods are now compared on identical scoring logic.

Repo housekeeping: deleted the stale `scorer-fix` branch (superseded, its useful
content — Chi's independent-solve trace distillation work — was not on this
branch's history) and reset the `chi`/`ldh` branches to match `main`, since prior
work on those branches was not meant to be kept as separate history.
