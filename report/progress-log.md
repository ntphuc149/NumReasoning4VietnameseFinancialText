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
