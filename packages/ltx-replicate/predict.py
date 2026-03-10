"""Cog predictor for LTX-2.3 distilled text-to-video pipeline."""

import random
import tempfile

import torch
from cog import BasePredictor, Input, Path
from huggingface_hub import hf_hub_download, snapshot_download

from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_pipelines.distilled import DistilledPipeline
from ltx_pipelines.utils.media_io import encode_video

WEIGHTS_DIR = "/weights"
HF_MODEL_REPO = "Lightricks/LTX-2.3"
HF_GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Download model weights and instantiate the pipeline."""
        hf_hub_download(
            HF_MODEL_REPO,
            "ltx-2.3-22b-distilled.safetensors",
            local_dir=WEIGHTS_DIR,
        )
        hf_hub_download(
            HF_MODEL_REPO,
            "ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            local_dir=WEIGHTS_DIR,
        )
        snapshot_download(
            HF_GEMMA_REPO,
            local_dir=f"{WEIGHTS_DIR}/gemma-3-12b",
        )

        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=f"{WEIGHTS_DIR}/ltx-2.3-22b-distilled.safetensors",
            gemma_root=f"{WEIGHTS_DIR}/gemma-3-12b",
            spatial_upsampler_path=f"{WEIGHTS_DIR}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            loras=[],
        )

    @torch.inference_mode()
    def predict(
        self,
        prompt: str = Input(description="Text prompt for video generation"),
        width: int = Input(
            description="Video width (must be divisible by 64)",
            default=768,
            ge=256,
            le=1536,
        ),
        height: int = Input(
            description="Video height (must be divisible by 64)",
            default=512,
            ge=256,
            le=1536,
        ),
        num_frames: int = Input(
            description="Number of frames (must be 8k+1, e.g. 17, 25, 33, ..., 97, 121)",
            default=97,
            ge=9,
            le=257,
        ),
        frame_rate: float = Input(
            description="Frames per second",
            default=24.0,
            ge=1.0,
            le=60.0,
        ),
        seed: int = Input(
            description="Random seed for reproducibility (-1 for random)",
            default=-1,
        ),
        enhance_prompt: bool = Input(
            description="Use LLM to enhance the prompt",
            default=False,
        ),
    ) -> Path:
        """Generate a video from a text prompt."""
        self._validate_inputs(width, height, num_frames)

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)

        video_iterator, audio = self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[],
            tiling_config=TilingConfig.default(),
            enhance_prompt=enhance_prompt,
        )

        video_chunks_number = get_video_chunks_number(
            num_frames, TilingConfig.default()
        )

        output_path = tempfile.mktemp(suffix=".mp4")
        encode_video(
            video=video_iterator,
            fps=int(frame_rate),
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )

        return Path(output_path)

    def _validate_inputs(self, width: int, height: int, num_frames: int) -> None:
        """Validate resolution and frame count constraints."""
        if width % 64 != 0 or height % 64 != 0:
            raise ValueError(
                f"width ({width}) and height ({height}) must be divisible by 64"
            )
        if (num_frames - 1) % 8 != 0:
            raise ValueError(
                f"num_frames ({num_frames}) must be of form 8k+1 "
                f"(e.g. 9, 17, 25, ..., 97, 121)"
            )
