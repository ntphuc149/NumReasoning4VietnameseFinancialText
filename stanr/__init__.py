"""STaNR: Self-Taught Numerical Reasoning.

Two entry points:

    ReasoningDistiller -- calls a teacher model (via an OpenAI-compatible API)
    with the CoNR prompt to build a reasoning-trace training set, keeping only
    the samples whose derived program matches gold.

    STaNRTrainer -- SFTs a student on that distilled set (round 0), then runs
    `loop` further self-teaching rounds: the current checkpoint attempts every
    question still unresolved, exact-match passes are merged into the
    training set, and the student is retrained from the base model on the
    merged set.

Minimal usage::

    from stanr import ReasoningDistiller, STaNRTrainer

    rd = ReasoningDistiller(
        model_name="gemma-4-31B-it",
        api_key="...",
        input_dataset="datasets/train.json",
        output_distill_path="datasets/train_distilled.json",
        filter_with="pa",
    )
    rd.run()

    trainer = STaNRTrainer(
        input_train_raw_dataset="datasets/train.json",
        input_distilled_train_set="datasets/train_distilled.json",
        input_val_raw_dataset="datasets/valid.json",
        input_distilled_val_set="datasets/valid_distilled.json",
        input_test_raw_dataset="datasets/test.json",
        pretrained_model_name="unsloth/Qwen3-4B",
        model_checkpoint_path="output/",
        precision="4-bit",
        loop=1,
    )
    trainer.train()

`STaNRTrainer` needs the optional GPU dependencies -- install with
`pip install -e ".[train]"`. `ReasoningDistiller` and everything in
`stanr.data`/`stanr.scoring` only need the base install.
"""

from stanr.distiller import ReasoningDistiller
from stanr.trainer import STaNRTrainer

__all__ = ["ReasoningDistiller", "STaNRTrainer"]
