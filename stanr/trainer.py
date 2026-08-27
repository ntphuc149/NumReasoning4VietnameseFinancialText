"""STaNRTrainer: SFT a student on distilled reasoning traces (round 0), then
run `loop` further self-teaching rounds.

Each round after round 0:
  1. find questions in the raw training set not yet in the current distilled
     training set (`data.samples_missing_from`);
  2. the CURRENT round's checkpoint attempts every one of them;
  3. keep only attempts whose program exactly matches gold (PA) -- the same
     filter every self-distillation notebook in this repo uses, independent
     of whatever `filter_with` `ReasoningDistiller` used for round 0;
  4. merge newly-passed samples into the training set;
  5. retrain FROM THE BASE MODEL on the merged set -- not a continued
     fine-tune of the previous round's adapter, matching the paper's own
     "Setting and Configuration" section ("each round retrains the base
     model from scratch ... rather than continuing the previous adapter").

GPU-heavy imports (unsloth/torch/trl/datasets) are deferred to inside the
methods that need them, so `from stanr import STaNRTrainer` and everything
except `.train()` works without those installed. Install them with
`pip install -e ".[train]"` when you actually want to run training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

from stanr import data, scoring
from stanr.prompts import SFT_SYSTEM_MESSAGE, SFT_USER_FRAME

Precision = Literal["4-bit", "16-bit", "full"]
_VALID_PRECISIONS = ("4-bit", "16-bit", "full")

# Matches the LoRA target modules used across every SFT notebook in this
# repo (Qwen3-4B, Gemma3-4B, qwen3-4b-thinking all share this list).
_LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

# Rendered literally by decode(skip_special_tokens=False) regardless of
# whether the tokenizer treats them as "special" -- stripped from the tail
# of a generated continuation before it is scored or stored.
_CHAT_END_MARKERS = ("<|im_end|>", "<end_of_turn>", "<|endoftext|>")


def _build_conversation(sample: dict) -> list[dict]:
    user_msg = data.build_user_message(SFT_USER_FRAME, sample)
    reasoning_trace = str(sample["qa"]["reasoning_trace"]).strip()
    program = str(sample["qa"]["program"]).strip()
    assistant_content = f"<think>{reasoning_trace}\n</think>\n\n{program}"
    return [
        {"role": "system", "content": SFT_SYSTEM_MESSAGE},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_content},
    ]


def _build_prompt_only(sample: dict) -> list[dict]:
    return [
        {"role": "system", "content": SFT_SYSTEM_MESSAGE},
        {"role": "user", "content": data.build_user_message(SFT_USER_FRAME, sample)},
    ]


def _strip_chat_end_markers(text: str) -> str:
    for marker in _CHAT_END_MARKERS:
        idx = text.find(marker)
        if idx != -1:
            text = text[:idx]
    return text.strip()


class STaNRTrainer:
    """SFT + iterative self-teaching (STaNR) for a single student model.

    `pretrained_model_name` is any Unsloth/HF model name -- the same class
    trains Qwen3-4B, Gemma3-4B, or qwen3-4b-thinking, differing only by this
    argument and (optionally) `precision`.

    `loop=0` degenerates to a single plain SFT run (round 0 only, no
    self-teaching) -- useful for reproducing the paper's SFT baseline with
    the same class used for STaNR.
    """

    def __init__(
        self,
        input_train_raw_dataset: str,
        input_distilled_train_set: str,
        input_val_raw_dataset: str,
        input_distilled_val_set: str,
        input_test_raw_dataset: str,
        pretrained_model_name: str,
        model_checkpoint_path: str,
        learning_rate: float = 2e-4,
        train_batch_size: int = 2,
        val_batch_size: int = 2,
        gradient_accumulation_steps: int = 8,
        precision: Precision = "4-bit",
        num_epochs_per_round: int = 3,
        loop: int = 1,
        max_seq_length: int | None = None,
        lora_r: int = 32,
        lora_alpha: int = 32,
        lora_dropout: float = 0.0,
        seed: int = 3407,
        eval_max_new_tokens: int = 1024,
        warmup_ratio: float = 0.03,
        weight_decay: float = 0.001,
    ):
        if precision not in _VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {_VALID_PRECISIONS}, got {precision!r}")
        if loop < 0:
            raise ValueError("loop must be >= 0")

        self.train_raw = data.load_json(input_train_raw_dataset)
        self.distilled_train = data.load_json(input_distilled_train_set)
        self.valid_raw = data.load_json(input_val_raw_dataset)
        self.distilled_valid = data.load_json(input_distilled_val_set)
        self.test_raw = data.load_json(input_test_raw_dataset)

        self.pretrained_model_name = pretrained_model_name
        self.model_checkpoint_path = Path(model_checkpoint_path)
        self.learning_rate = learning_rate
        self.train_batch_size = train_batch_size
        self.val_batch_size = val_batch_size
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.precision = precision
        self.num_epochs_per_round = num_epochs_per_round
        self.loop = loop
        self.max_seq_length = max_seq_length
        self.lora_r = lora_r
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.seed = seed
        self.eval_max_new_tokens = eval_max_new_tokens
        self.warmup_ratio = warmup_ratio
        self.weight_decay = weight_decay

        self.history: list[dict] = []
        self.final_train_set: list[dict] = []

    # ------------------------------------------------------------- public --
    def train(self) -> list[dict]:
        """Round 0 (plain SFT on the distilled set) + `loop` self-teaching
        rounds. Returns the per-round history (also kept at `self.history`);
        each entry has round/adapter_dir/train_size/newly_passed/pa/ea."""
        self.history = []
        train_set = self.distilled_train

        adapter_dir = self._sft_round(train_set, round_idx=0)
        pa, ea = self._evaluate(adapter_dir)
        self.history.append({
            "round": 0, "adapter_dir": str(adapter_dir),
            "train_size": len(train_set), "newly_passed": 0,
            "pa": pa, "ea": ea,
        })
        print(f"[round 0] train_size={len(train_set)} PA={pa:.4f} EA={ea:.4f}")

        for round_idx in range(1, self.loop + 1):
            unresolved = data.samples_missing_from(self.train_raw, train_set)
            if not unresolved:
                print(f"[round {round_idx}] no unresolved questions left -- stopping.")
                break

            raw_outputs = self._generate(adapter_dir, unresolved, do_sample=True,
                                         max_new_tokens=self.eval_max_new_tokens)
            newly_passed = self._filter_and_build(unresolved, raw_outputs)
            if not newly_passed:
                print(f"[round {round_idx}] no new sample passed the exact-match filter -- "
                      f"stopping (further rounds would repeat this one).")
                break

            train_set = train_set + newly_passed
            adapter_dir = self._sft_round(train_set, round_idx=round_idx)
            pa, ea = self._evaluate(adapter_dir)
            self.history.append({
                "round": round_idx, "adapter_dir": str(adapter_dir),
                "train_size": len(train_set), "newly_passed": len(newly_passed),
                "pa": pa, "ea": ea,
            })
            print(f"[round {round_idx}] +{len(newly_passed)} samples "
                  f"(train_size={len(train_set)}) PA={pa:.4f} EA={ea:.4f}")

        self.final_train_set = train_set
        return self.history

    # -------------------------------------------------------- GPU internals --
    def _load_model(self, model_name: str):
        """`model_name` is either `self.pretrained_model_name` (fresh base,
        used at the start of every round -- STaNR never continues a previous
        round's adapter) or a saved adapter directory (used only for
        generation/eval, never as a round's training starting point)."""
        from unsloth import FastLanguageModel

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=self.max_seq_length or 4096,
            load_in_4bit=self.precision == "4-bit",
            load_in_8bit=False,
            full_finetuning=self.precision == "full",
        )
        return model, tokenizer

    def _apply_lora(self, model):
        from unsloth import FastLanguageModel

        return FastLanguageModel.get_peft_model(
            model,
            r=self.lora_r,
            target_modules=_LORA_TARGET_MODULES,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            bias="none",
            use_gradient_checkpointing="unsloth",
            random_state=self.seed,
        )

    def _measure_max_seq_length(self, tokenizer, samples: list[dict]) -> int:
        lengths = []
        for s in samples:
            text = tokenizer.apply_chat_template(_build_conversation(s), tokenize=False, enable_thinking=True)
            lengths.append(len(tokenizer(text, add_special_tokens=False)["input_ids"]))
        return max(lengths) if lengths else 2048

    def _sft_round(self, train_samples: list[dict], round_idx: int) -> Path:
        from datasets import Dataset
        from trl import SFTTrainer, SFTConfig
        from unsloth.chat_templates import train_on_responses_only

        model, tokenizer = self._load_model(self.pretrained_model_name)
        if self.precision != "full":
            model = self._apply_lora(model)

        max_seq_length = self.max_seq_length or self._measure_max_seq_length(
            tokenizer, train_samples + self.distilled_valid,
        )

        train_texts = [
            tokenizer.apply_chat_template(_build_conversation(s), tokenize=False, enable_thinking=True)
            for s in train_samples
        ]
        valid_texts = [
            tokenizer.apply_chat_template(_build_conversation(s), tokenize=False, enable_thinking=True)
            for s in self.distilled_valid
        ]
        train_dataset = Dataset.from_dict({"text": train_texts}).shuffle(seed=self.seed)
        valid_dataset = Dataset.from_dict({"text": valid_texts})

        round_dir = self.model_checkpoint_path / f"round_{round_idx}"
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=valid_dataset,
            args=SFTConfig(
                output_dir=str(round_dir / "trainer_state"),
                dataset_text_field="text",
                max_seq_length=max_seq_length,
                per_device_train_batch_size=self.train_batch_size,
                per_device_eval_batch_size=self.val_batch_size,
                gradient_accumulation_steps=self.gradient_accumulation_steps,
                warmup_ratio=self.warmup_ratio,
                num_train_epochs=self.num_epochs_per_round,
                learning_rate=self.learning_rate,
                logging_steps=20,
                eval_strategy="epoch",
                save_strategy="epoch",
                save_total_limit=2,
                load_best_model_at_end=True,
                metric_for_best_model="eval_loss",
                optim="adamw_8bit",
                weight_decay=self.weight_decay,
                lr_scheduler_type="cosine",
                seed=self.seed,
                report_to="none",
            ),
        )
        # Loss-masks everything except the assistant turn -- standard
        # practice, and important here since the context block is routinely
        # much longer than the program string actually being learned.
        trainer = train_on_responses_only(
            trainer,
            instruction_part="<|im_start|>user\n",
            response_part="<|im_start|>assistant\n",
        )
        trainer.train()

        adapter_dir = round_dir / "adapter"
        model.save_pretrained(adapter_dir)
        tokenizer.save_pretrained(adapter_dir)
        return adapter_dir

    def _generate(self, model_dir: Path, samples: list[dict], do_sample: bool, max_new_tokens: int) -> list[str]:
        """Raw decoded continuations for `samples`, one `model.generate()`
        call each. `skip_special_tokens=False` on purpose: the model's own
        `</think>` needs to survive decoding so `data.extract_think_and_program`
        can find it as a literal substring, whether or not the tokenizer
        registers it as a dedicated special token (Qwen does; Gemma may not,
        in which case it is just ordinary text the model was trained to
        produce, which decodes as such either way)."""
        import torch
        from unsloth import FastLanguageModel

        model, tokenizer = self._load_model(str(model_dir))
        FastLanguageModel.for_inference(model)

        gen_kwargs = dict(max_new_tokens=max_new_tokens)
        if do_sample:
            gen_kwargs.update(do_sample=True, temperature=0.6, top_p=0.95, top_k=20, min_p=0)
        else:
            # Explicit greedy decoding. Several checkpoints' own
            # generation_config.json default to do_sample=True, so this must
            # be passed here rather than assumed, or "greedy" eval silently
            # samples instead.
            gen_kwargs.update(do_sample=False, temperature=None, top_p=None, top_k=None)

        outputs = []
        for s in samples:
            prompt = tokenizer.apply_chat_template(
                _build_prompt_only(s), tokenize=False, add_generation_prompt=True, enable_thinking=True,
            )
            model_inputs = tokenizer([prompt], return_tensors="pt").to(model.device)
            with torch.no_grad():
                generated_ids = model.generate(**model_inputs, **gen_kwargs)
            output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist()
            raw = tokenizer.decode(output_ids, skip_special_tokens=False)
            outputs.append(_strip_chat_end_markers(raw))
            torch.cuda.empty_cache()
        return outputs

    def _filter_and_build(self, unresolved: list[dict], raw_outputs: list[str]) -> list[dict]:
        """PA-only filter -- same as every self-distillation notebook in
        this repo, independent of whatever filter round 0's distillation
        used."""
        newly_passed = []
        for sample, raw in zip(unresolved, raw_outputs):
            trace, program = data.extract_think_and_program(raw)
            try:
                gen_tok = scoring.program_tokenization(scoring.extract_program(program))
                gold_tok = scoring.program_tokenization(sample["qa"]["program"])
                passed = scoring.equal_program(gold_tok, gen_tok)
            except Exception:
                passed = False
            if passed:
                newly_passed.append(data.with_reasoning_trace(sample, trace, "self_distilled"))
        return newly_passed

    def _evaluate(self, model_dir: Path) -> tuple[float, float]:
        """Greedy PA/EA on `input_test_raw_dataset`."""
        raw_outputs = self._generate(model_dir, self.test_raw, do_sample=False,
                                     max_new_tokens=self.eval_max_new_tokens)
        pa_scores, ea_scores = [], []
        for sample, raw in zip(self.test_raw, raw_outputs):
            _, program = data.extract_think_and_program(raw)
            pa, ea = scoring.score_one(
                program, sample["qa"]["program"], sample["qa"]["exe_ans"], table=sample["table"],
            )
            pa_scores.append(pa)
            ea_scores.append(ea)
        n = len(pa_scores) or 1
        return sum(pa_scores) / n, sum(ea_scores) / n
