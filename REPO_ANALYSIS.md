# LTX-2 Repository Analysis

## Overview

**LTX-2** is Lightricks' open-source DiT-based (Diffusion Transformer) audio-video foundation model. The current version is **LTX-2.3** at 22B parameters. It supports synchronized audio+video generation, text-to-video, image-to-video, inpainting, keyframe interpolation, camera control, and more.

- Paper: https://arxiv.org/abs/2601.03233
- HuggingFace: https://huggingface.co/Lightricks/LTX-2.3

---

## Repository Structure

Monorepo managed with `uv`, containing 3 packages:

```
LTX-2/
├── packages/
│   ├── ltx-core/        # Core model, inference stack, VAE, text encoders, quantization
│   ├── ltx-pipelines/   # High-level generation pipelines
│   └── ltx-trainer/     # LoRA / full fine-tuning tools
├── pyproject.toml
└── uv.lock
```

### ltx-core
```
src/ltx_core/
├── model/
│   ├── transformer/     # Main diffusion transformer (LTXModel)
│   ├── video_vae/       # Video encoder/decoder
│   ├── audio_vae/       # Audio encoder/decoder + vocoder
│   └── upsampler/       # Spatial upscaler
├── text_encoders/gemma/ # Gemma 3 text encoder pipeline
├── components/          # Scheduler, diffusion steps, guiders, patchifiers
├── conditioning/        # Attention masks, conditioning types
├── guidance/            # Perturbations
├── loader/              # Model loading utilities
└── quantization/        # FP8 quantization
```

### ltx-pipelines
```
src/ltx_pipelines/
├── ti2vid_two_stages.py      # Recommended: 2-stage text/image-to-video with 2x upsampling
├── ti2vid_two_stages_hq.py   # Same but with res_2s second-order sampler
├── ti2vid_one_stage.py       # Single-stage, faster/lower quality
├── distilled.py              # Fastest: 8 predefined sigmas (8+4 steps)
├── ic_lora.py                # Image/video-to-video via IC-LoRA
├── a2vid_two_stage.py        # Audio-to-video
├── retake.py                 # Regenerate a time region of an existing video
└── keyframe_interpolation.py # Interpolate between keyframe images
```

---

## Weights

**Not included in the repo.** Must be downloaded separately from HuggingFace (`Lightricks/LTX-2.3`):

| File | Purpose | Required |
|---|---|---|
| `ltx-2.3-22b-dev.safetensors` or `ltx-2.3-22b-distilled.safetensors` | Main model checkpoint | Yes (pick one) |
| `ltx-2.3-spatial-upscaler-x2-1.0.safetensors` | Spatial upscaler | Yes (for 2-stage pipelines) |
| `ltx-2.3-22b-distilled-lora-384.safetensors` | Distilled LoRA | Yes (for most 2-stage pipelines) |
| `ltx-2.3-temporal-upscaler-x2-1.0.safetensors` | Temporal upscaler | Future use |
| Gemma 3 12B (`google/gemma-3-12b-it-qat-q4_0-unquantized`) | Text encoder | Yes |

Additional optional LoRAs available for camera control, pose, inpainting, motion tracking, and detailing.

---

## How to Run

```bash
# 1. Install dependencies
uv sync --frozen
source .venv/bin/activate

# 2. Download weights from HuggingFace

# 3. Use a pipeline in Python
from ltx_pipelines import DistilledPipeline
```

### Optimization Tips
- Use `DistilledPipeline` for fastest inference (8 steps)
- Enable FP8 quantization to reduce VRAM: `quantization=QuantizationPolicy.fp8_cast()`
- Install xFormers: `uv sync --extra xformers`
- For Hopper GPUs: Flash Attention 3 + `fp8-scaled-mm`

---

## Prompt Enhancement System

Pipelines support automatic prompt enhancement via the `enhance_prompt` parameter. When enabled, the user's raw prompt is passed through **Gemma 3** acting as a rewriting assistant before being fed into the diffusion model.

Two system prompts govern this behavior, stored at:
```
packages/ltx-core/src/ltx_core/text_encoders/gemma/encoders/prompts/
├── gemma_t2v_system_prompt.txt   # Text-to-video
└── gemma_i2v_system_prompt.txt   # Image-to-video
```

Both prompts instruct Gemma to:
- Expand vague inputs into detailed, chronological scene descriptions
- Add a complete audio/soundscape layer
- Use present-progressive verbs and temporal connectors
- Output a single continuous paragraph (no markdown)
- Avoid inventing camera motion or dialogue unless requested

These prompts can be overridden via the `system_prompt` parameter in `base_encoder.py`.

---

## Content Safety

**There are no meaningful content safety controls in this codebase.**

### What exists
- A single line in both system prompts: `"If unsafe/invalid, return original user prompt."` — applies only to the Gemma prompt enhancer, not the diffusion model; has no definition of "unsafe" and no enforcement mechanism.

### What does NOT exist
- NSFW classifier
- Content policy enforcement
- Prompt blocklist/allowlist
- Post-generation safety checker
- Model-level refusal behavior (no evidence of safety fine-tuning or RLHF in the training code)
- Rate limiting or moderation layer

**This is a developer/research release. Safety enforcement is left entirely to the deployer.**

To verify whether the weights themselves embed any safety alignment, the HuggingFace model card (`Lightricks/LTX-2.3`) would be the authoritative source — the repo code contains no evidence of it.

---

## Replicate Deployment

No Cog/Replicate configuration exists in the repo. A custom deployment would require building from scratch:

1. **`cog.yaml`** — defines GPU environment (CUDA 12.1, Python 3.11, uv install)
2. **`predict.py`** — wraps a pipeline in a `cog.BasePredictor` class
3. **Weights strategy** — either bundled in image (large, fast cold start) or downloaded at runtime in `setup()` (smaller image, slower cold start)

### Key deployment challenges
| Challenge | Detail |
|---|---|
| VRAM | 22B model requires ~80GB GPU; Replicate A100 (80GB) is minimum viable |
| Gemma license | Requires accepting Google's terms + HuggingFace token |
| Cold start | Model loading is slow; careful `setup()` design needed |
| Audio+video output | Requires muxing audio and video into a container format |
