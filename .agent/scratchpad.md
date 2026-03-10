# Scratchpad — Custom Replicate Pipeline for LTX-2.3

## Objective
Create a Cog/Replicate pipeline (`packages/ltx-replicate/`) that wraps LTX-2.3's DistilledPipeline for text-to-video generation.

## Understanding

### What exists
- Monorepo with `ltx-core`, `ltx-pipelines`, `ltx-trainer` packages
- `DistilledPipeline` class in `packages/ltx-pipelines/src/ltx_pipelines/distilled.py`
  - Constructor: needs `distilled_checkpoint_path`, `gemma_root`, `spatial_upsampler_path`, `loras`, optional `device`/`quantization`
  - `__call__`: takes `prompt`, `seed`, `height`, `width`, `num_frames`, `frame_rate`, `images`, optional `tiling_config`/`enhance_prompt`
  - Returns `tuple[Iterator[torch.Tensor], Audio]`
- `encode_video()` in `ltx_pipelines.utils.media_io` handles MP4 output (H.264 + AAC)
- Model weights: `ltx-2.3-22b-distilled.safetensors`, `ltx-2.3-spatial-upscaler-x2-1.0.safetensors`, Gemma 3 text encoder
- Default params: 768x512 (stage_1) → 1536x1024 (stage_2), 121 frames, 24fps, seed=10

### Requirements summary
1. Pipeline in `packages/ltx-replicate/` with `cog.yaml` + `predict.py`
2. Start with `distilled` pipeline, design for easy switch to `ti2vid_two_stages`
3. All params configurable with sensible defaults (prompt required, rest optional)
4. MP4 output, single file URL
5. Download weights from HuggingFace in setup()
6. Target A100 (80GB)
7. Defaults: 768x512, 97 frames, guidance ~3.0, random seed
8. Design for optional image input later (text-to-video only for now)
9. Public model with obscure name

### Key design decisions needed
- How to structure predict.py to abstract pipeline selection (distilled vs ti2vid_two_stages)
- Weight download strategy in setup() — pget vs huggingface-cli
- How to reference sibling packages (ltx-core, ltx-pipelines) from within the Cog container
- Resolution validation (must be divisible by 64 for 2-stage)
- num_frames validation (must be 8k+1)

## Plan
Delegating to Designer to create comprehensive design document covering:
1. `cog.yaml` configuration (CUDA, Python deps, system packages)
2. `predict.py` structure (setup/predict methods, input/output types)
3. Weight download approach
4. Package dependency resolution within Docker
5. Pipeline abstraction for future switchability

## Design Iteration 1

### Design document created: `docs/local/design.md`

Evaluated 3 options:
- **Option A (Direct Integration)** — Recommended. predict.py directly wraps DistilledPipeline. 2 files, minimal abstraction, PoC-appropriate.
- **Option B (Factory Pattern)** — Over-engineered. Runtime pipeline switching, but wastes GPU memory and adds unnecessary abstraction.
- **Option C (Env-Var Driven)** — Middle ground. Pipeline mode via env var, but conditional logic grows and switching still requires redeployment.

### Key design decisions made:
1. **Package installation**: Use `run:` commands in cog.yaml to `pip install` ltx-core and ltx-pipelines from `/src/packages/`
2. **Weight download**: `huggingface_hub.hf_hub_download()` in setup(), HF_TOKEN via Replicate env var
3. **Input validation**: Enforce width/height divisible by 64, num_frames as 8k+1
4. **No abstraction layer**: Direct import of DistilledPipeline — switching to ti2vid_two_stages is a small code change (~20 lines)
5. **TilingConfig**: Use default for PoC, don't expose to users

## Design Review (Iteration 2)

Reviewed `docs/local/design.md` as Design Reviewer.

### Criteria evaluation
- **2+ options with pros/cons**: PASS — 3 options with detailed analysis
- **Affected files identified**: PASS — Key Files to Create/Read tables complete
- **Edge cases & error handling**: PASS — Input validation covered (resolution div by 64, frames 8k+1)
- **Data/schema changes**: PASS — Explicitly "None"
- **Recommendation justified**: PASS — 5 clear rationale points
- **No blocking gaps**: PASS — Open questions have recommendations

### Codebase verification
All referenced files, classes, and signatures confirmed against actual code:
- DistilledPipeline constructor/call signatures match
- encode_video() signature matches
- get_video_chunks_number() exists and is exported from ltx_core
- packages/ltx-replicate/ does not exist yet (correct — it's new)

### Verdict: APPROVED — publishing design.ready

## Implementation (Iteration 3)

### What was built
Implemented Option A (Direct Integration) as approved in the design:

**`packages/ltx-replicate/cog.yaml`**
- CUDA 12.4, Python 3.11, GPU enabled
- System packages: ffmpeg, libgl1-mesa-glx, libglib2.0-0
- Python deps pinned to match ltx-core/ltx-pipelines requirements
- Sibling packages installed via `run:` commands from `/src/packages/`

**`packages/ltx-replicate/predict.py`**
- `setup()`: Downloads weights from HuggingFace (distilled checkpoint, spatial upscaler, Gemma 3 text encoder), instantiates DistilledPipeline
- `predict()`: Validates inputs, generates video via pipeline, encodes to MP4 with encode_video()
- Input validation: width/height must be divisible by 64, num_frames must be 8k+1
- Seed handling: -1 generates random seed
- Uses TilingConfig.default() for VAE decoding
- All API signatures verified against actual codebase

### Verification
- Python syntax: PASS
- YAML syntax: PASS
- API signatures cross-referenced with actual code: PASS

### Committed as: `feat(ltx-replicate): add Cog/Replicate pipeline for LTX-2.3`

## Validation (Iteration 4)

### Test results for task-1773165327-d117
- **Python syntax**: PASS — `python3 -m py_compile predict.py` clean
- **YAML syntax**: PASS — `yaml.safe_load()` parses without errors
- **Ruff linting**: PASS — `ruff check predict.py` all checks passed
- **Import path validation**: PASS — all referenced module files confirmed to exist:
  - `ltx_core.model.video_vae` exports `TilingConfig`, `get_video_chunks_number` ✓
  - `ltx_pipelines.distilled.DistilledPipeline` class exists ✓
  - `ltx_pipelines.utils.media_io.encode_video` function exists ✓
- **torch not installed locally**: Expected — packages resolve correctly through torch import chain; failure is at `torch` not at our import paths

### Conclusion
All validations pass. Pipeline is ready for Cog build on GPU-capable machine.
