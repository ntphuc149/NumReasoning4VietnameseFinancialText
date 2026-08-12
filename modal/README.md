# Modal environment setup

This directory holds Modal infrastructure shared across this repo's notebooks that run on
Modal (currently the GRPO notebooks under `notebooks/vinumqa/sft-grpo/`).

## Why this exists

Modal Notebook's default and Colab images ship a CUDA **runtime** but not the full CUDA **dev**
toolchain (headers like `curand.h`, NVRTC 13, etc.). `unsloth`/`vllm` install fine on top of
them, but fail at import/first-use time in ways that look unrelated to the missing headers:

- `ImportError: cannot import name 'Sentinel' from 'typing_extensions'` — actually an unrelated
  dependency-resolution issue (see below), easy to confuse with a headers problem but isn't one.
- `ImportError: libnvrtc.so.13: cannot open shared object file` — `UNSLOTH_VLLM_STANDBY`'s
  VRAM-saving mode needs vLLM's `cumem_allocator`, which needs CUDA 13's NVRTC runtime.
- `RuntimeError: FlashInfer failed to JIT-compile` — Unsloth's error handler mislabels this as
  "ninja not found"; the real cause is `fatal error: curand.h: No such file or directory` when
  FlashInfer JIT-compiles a CUDA sampling kernel via `nvcc`.

The first three fixes for these were environment-variable workarounds (pin
`typing_extensions`, drop `UNSLOTH_VLLM_STANDBY`, set `UNSLOTH_VLLM_NO_FLASHINFER=1`) applied
directly in each notebook's install cell. That works, but it's patching around a missing
toolchain one symptom at a time. Kaggle's default image already ships full CUDA dev headers,
which is why the identical `pip install unsloth vllm` just works there with none of the above.

`grpo_env_image.py` closes the gap at the source: it defines a Modal image built from
`nvidia/cuda:12.8.1-devel-ubuntu22.04` (the `-devel` variant, not `-runtime`) with
`unsloth`/`vllm`/`trl`/etc. pre-installed, so none of the three workarounds above are needed.

## How Modal Notebook's "Custom image" picker works

Modal Notebook does not accept a pasted image definition. Instead, its **Image → Custom image**
panel lists **Modal Functions already deployed to a given Environment**, and reuses whichever
function's image you pick. So attaching a custom image means: deploy an app containing a
function that uses the image you want, then select that function from the dropdown.

`grpo_env_image.py` is exactly that — a one-function Modal app whose only purpose is to make
its image selectable in the notebook UI. The function body itself is never called from the
notebook.

## Setup

1. Make sure the `modal` CLI is authenticated (one-time):

   ```bash
   modal token new
   ```

   This opens a browser to log in and saves credentials to `~/.modal.toml`. (If you're doing
   this from inside a Modal Notebook cell instead of a local shell, the notebook already has
   credentials — just run the `modal deploy` command below with a `!` prefix.)

2. Deploy the image-registering app (from the repo root):

   ```bash
   modal deploy modal/grpo_env_image.py --env main
   ```

   First deploy takes a few minutes (pulling the CUDA base image + installing packages).
   Subsequent deploys of the same file are fast (cached layers).

3. In the Modal Notebook UI: **Image → Custom image**, set **Environment** to `main`, then open
   the **Function** dropdown — it should now list `grpo_env`. Select it, **Save changes**.

4. **Restart the kernel.** The notebook's own install cell (`pip install unsloth vllm`, the
   `typing_extensions`/`UNSLOTH_VLLM_*` workarounds) is no longer needed once this image is
   attached — everything is pre-installed. Skip or delete that cell.

## Updating the image

Edit the `pip_install(...)` chain in `grpo_env_image.py` and re-run `modal deploy
modal/grpo_env_image.py --env main`. The notebook's attached function (`grpo_env`) doesn't need
to be re-selected — Modal resolves it to the new image on the next container start.
