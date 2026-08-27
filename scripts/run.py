"""Fixed runner -- every setting has a sensible built-in default, override
whatever you need directly on the command line. No file to edit.

    python scripts/run.py
        # runs with all defaults: Qwen3-4B, reuses the distilled data
        # already in this repo, one self-teaching round.

    python scripts/run.py --teacher_model_name "some-other-teacher" \\
        --dataset datasets/ViNumQA/origin --run_distill --loop 5
        # distills fresh from a different teacher, then trains 5 rounds.

Run `python scripts/run.py --help` to see every overridable flag.

Teacher API credentials (only needed with --run_distill) are read from
`.env` at the repo root, the same convention every API notebook in this
repo already uses:

    API_KEY=...
    BASE_URL=https://mkp-api.fptcloud.com/v1
"""

import argparse
import os

from dotenv import load_dotenv

from stanr import ReasoningDistiller, STaNRTrainer


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    dist = p.add_argument_group("distillation (only used with --run_distill)")
    dist.add_argument("--run_distill", action="store_true",
                      help="Regenerate --train_distilled/--valid_distilled by calling --teacher_model_name.")
    dist.add_argument("--teacher_model_name", default="gemma-4-31B-it")
    dist.add_argument("--teacher_base_url", default="https://mkp-api.fptcloud.com/v1")
    dist.add_argument("--filter_with", default="pa", choices=["pa", "strict", "union"])

    data = p.add_argument_group("data")
    data.add_argument("--dataset", default="datasets/ViNumQA/origin",
                      help="Directory containing train.json/valid.json/test.json. "
                           "Ignored for any of --train_raw/--valid_raw/--test_raw passed explicitly.")
    data.add_argument("--train_raw", default=None, help="Default: <dataset>/train.json")
    data.add_argument("--valid_raw", default=None, help="Default: <dataset>/valid.json")
    data.add_argument("--test_raw", default=None, help="Default: <dataset>/test.json")
    data.add_argument("--train_distilled",
                      default="datasets/ViNumQA/train_with_reasoning_trace_pa_en_distill.json")
    data.add_argument("--valid_distilled",
                      default="datasets/ViNumQA/valid_with_reasoning_trace_pa_en_distill.json")

    train = p.add_argument_group("student model & training")
    train.add_argument("--pretrained_model_name", default="unsloth/Qwen3-4B",
                       help="e.g. unsloth/Qwen3-4B, unsloth/gemma-3-4b-it, unsloth/Qwen3-4B-Thinking-2507, "
                            "or any other Unsloth/HF model name.")
    train.add_argument("--model_checkpoint_path", default="output/stanr-run")
    train.add_argument("--precision", default="4-bit", choices=["4-bit", "16-bit", "full"])
    train.add_argument("--loop", type=int, default=1, help="Number of self-teaching rounds after round 0.")
    train.add_argument("--learning_rate", type=float, default=2e-4)
    train.add_argument("--train_batch_size", type=int, default=2)
    train.add_argument("--val_batch_size", type=int, default=2)
    train.add_argument("--gradient_accumulation_steps", type=int, default=8)
    train.add_argument("--num_epochs_per_round", type=int, default=3)

    return p


def main():
    args = build_parser().parse_args()

    train_raw = args.train_raw or f"{args.dataset}/train.json"
    valid_raw = args.valid_raw or f"{args.dataset}/valid.json"
    test_raw = args.test_raw or f"{args.dataset}/test.json"

    if args.run_distill:
        load_dotenv()
        api_key = os.environ["API_KEY"]
        base_url = os.environ.get("BASE_URL", args.teacher_base_url)

        print(f"Distilling with teacher '{args.teacher_model_name}' -> {args.train_distilled}")
        ReasoningDistiller(
            model_name=args.teacher_model_name, api_key=api_key, base_url=base_url,
            input_dataset=train_raw, output_distill_path=args.train_distilled,
            filter_with=args.filter_with,
        ).run()

        print(f"Distilling with teacher '{args.teacher_model_name}' -> {args.valid_distilled}")
        ReasoningDistiller(
            model_name=args.teacher_model_name, api_key=api_key, base_url=base_url,
            input_dataset=valid_raw, output_distill_path=args.valid_distilled,
            filter_with=args.filter_with,
        ).run()
    else:
        print("--run_distill not set -- reusing --train_distilled/--valid_distilled as given.")

    trainer = STaNRTrainer(
        input_train_raw_dataset=train_raw,
        input_distilled_train_set=args.train_distilled,
        input_val_raw_dataset=valid_raw,
        input_distilled_val_set=args.valid_distilled,
        input_test_raw_dataset=test_raw,
        pretrained_model_name=args.pretrained_model_name,
        model_checkpoint_path=args.model_checkpoint_path,
        learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size,
        val_batch_size=args.val_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        precision=args.precision,
        num_epochs_per_round=args.num_epochs_per_round,
        loop=args.loop,
    )
    history = trainer.train()

    print("\nPer-round results:")
    for round_result in history:
        print(round_result)


if __name__ == "__main__":
    main()
