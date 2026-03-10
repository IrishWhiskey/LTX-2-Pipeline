# ltx-replicate

A [Cog](https://github.com/replicate/cog) wrapper that exposes the LTX-2.3 `DistilledPipeline` as a [Replicate](https://replicate.com) model. It downloads model weights from HuggingFace at container build time and serves a single `predict` endpoint that generates an MP4 video from a text prompt.

## Overview

This package wraps [`DistilledPipeline`](../ltx-pipelines/src/ltx_pipelines/distilled.py) — the fastest LTX-2.3 inference mode, using 8 predefined sigmas instead of a full DDPM schedule. The spatial upscaler is also loaded by default, producing 2× upscaled output at native resolution.

**Model weights loaded at setup:**

| File | Source |
|------|--------|
| `ltx-2.3-22b-distilled.safetensors` | `Lightricks/LTX-2.3` |
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | `Lightricks/LTX-2.3` |
| Gemma 3 12B text encoder | `google/gemma-3-12b-it-qat-q4_0-unquantized` |

## Requirements

- **GPU:** NVIDIA A100 (80 GB) or equivalent — the 22B model requires ~80 GB VRAM
- **CUDA:** 12.4
- **Python:** 3.11
- **System packages:** `ffmpeg`, `libgl1-mesa-glx`, `libglib2.0-0`

All Python dependencies are declared in [`cog.yaml`](cog.yaml) and are installed automatically by Cog.

## Deploying to Replicate

1. Install Docker:
   - **Linux:** `curl -fsSL https://get.docker.com | sh`
   - **macOS:** download [Docker Desktop](https://www.docker.com/products/docker-desktop/) or `brew install --cask docker`

   Start the Docker daemon before continuing (`sudo systemctl start docker` on Linux, or launch the Docker Desktop app on macOS).

2. Install the Cog CLI:
   ```bash
   sh <(curl -fsSL https://cog.run/install.sh)
   ```
   On macOS you can also use `brew install replicate/tap/cog`.

3. From the **repo root**, log in and push:
   ```bash
   cog login
   cog push r8.im/<your-username>/<your-model-name>
   ```

   `cog.yaml` lives at the repo root so that the full monorepo is copied into `/src` inside the container, making `packages/ltx-core` and `packages/ltx-pipelines` available at build time.

## Running Locally with Cog

Build the image from the **repo root**:
```bash
cog build
```

Run a prediction:
```bash
cog predict -i prompt="A golden retriever runs along a sunny beach, waves crashing in the background"
```

> **Note:** The first run downloads ~80 GB of weights from HuggingFace into `/weights`. Subsequent runs reuse the cached weights if the container is not removed.

## API Reference

### Inputs

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `prompt` | `str` | *(required)* | Text description of the video to generate |
| `width` | `int` | `768` | Output width in pixels; must be divisible by 64 (range: 256–1536) |
| `height` | `int` | `512` | Output height in pixels; must be divisible by 64 (range: 256–1536) |
| `num_frames` | `int` | `97` | Number of frames; must satisfy `(num_frames − 1) % 8 == 0` (e.g. 9, 17, 25 … 97, 121, …) |
| `frame_rate` | `float` | `24.0` | Frames per second (range: 1–60) |
| `seed` | `int` | `-1` | Random seed for reproducibility; `-1` picks a random seed |
| `enhance_prompt` | `bool` | `false` | Run prompt through the Gemma LLM rewriter before generation |

### Output

A single `.mp4` file (H.264 video, AAC audio).

### Input constraints

- `width` and `height` must both be divisible by **64**.
- `num_frames` must satisfy `(num_frames − 1) % 8 == 0` — valid values: 9, 17, 25, 33, …, 97, 121, …, 257.

## Structure

```
cog.yaml                          # Cog build configuration (repo root)
packages/ltx-replicate/
├── requirements.txt              # Python dependencies for the Cog image
└── predict.py                    # Cog Predictor class (setup + predict)
```

`predict.py` contains two methods:

- **`setup()`** — Downloads weights from HuggingFace and instantiates `DistilledPipeline`. Called once when the container starts.
- **`predict()`** — Runs inference and encodes the result to an MP4 file in a temporary path, which Cog streams back to the caller.

## Prompt Tips

The model responds best to detailed, cinematic descriptions. See the [prompting guide](../../README.md#️-prompting-for-ltx-2) in the top-level README for best practices, or enable `enhance_prompt=true` to let the built-in Gemma rewriter expand a short prompt automatically.
