"""Deploy a tiny Modal app whose only purpose is to register a custom image that the
Modal *Notebook* UI can then attach to (Image -> Custom image -> Environment: main ->
Function: grpo_env).

Run once from a machine with the modal CLI configured:

    modal deploy modal/grpo_env_image.py

After it deploys, go back to the notebook's Image picker, choose Custom image, pick
Environment `main`, and the Function dropdown will now list `grpo_env`. Select it, Save.

See modal/README.md for the full setup walkthrough and why this exists.
"""

import modal

# CUDA -devel base = ships the full dev headers (curand.h, nvrtc, etc.) that Modal's default
# image lacks -- this is what makes unsloth/vllm build cleanly, the same way Kaggle's image does.
image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.8.1-devel-ubuntu22.04",
        add_python="3.11",
    )
    .pip_install("unsloth", "vllm")
    .pip_install("typing_extensions>=4.13")
    .pip_install("transformers==4.56.2")
    .pip_install("trl==0.22.2", extra_options="--no-deps")
    .pip_install("tabulate", "sympy", "huggingface_hub")
)

app = modal.App("grpo-vinumqa-env")


# The notebook only needs this function to *exist* so its image is selectable; the body is
# never actually called from the notebook. A GPU is declared so the image is provisioned on a
# GPU host consistent with how you'll run the notebook.
@app.function(image=image, gpu="A100-80GB")
def grpo_env():
    import torch, unsloth  # noqa: F401
    print("unsloth import OK, CUDA available:", torch.cuda.is_available())
