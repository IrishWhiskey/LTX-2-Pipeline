"""Cog predictor for LTX-2.3 distilled text-to-video pipeline."""

# === Immediate startup logging (before ANY heavy imports) ===
import os
import sys
import time

_module_load_start = time.time()
print(f"[predict.py] Module loading started (pid={os.getpid()})", flush=True)

print("[predict.py] Importing stdlib modules...", flush=True)
import logging
import random
import subprocess
import tempfile

print(f"[predict.py] Stdlib imports done in {time.time() - _module_load_start:.1f}s", flush=True)

print("[predict.py] Importing torch...", flush=True)
_t0 = time.time()
import torch

print(f"[predict.py] torch imported in {time.time() - _t0:.1f}s", flush=True)

print("[predict.py] Importing cog...", flush=True)
_t0 = time.time()
from cog import BasePredictor, Input, Path

print(f"[predict.py] cog imported in {time.time() - _t0:.1f}s", flush=True)
print(
    f"[predict.py] All top-level imports done in {time.time() - _module_load_start:.1f}s",
    flush=True,
)

# Configure logging to stdout with timestamps so Replicate captures it
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    stream=sys.stdout,
    force=True,
)
logger = logging.getLogger("ltx-replicate")

WEIGHTS_DIR = "/weights"
HF_MODEL_REPO = "Lightricks/LTX-2.3"
HF_GEMMA_REPO = "google/gemma-3-12b-it-qat-q4_0-unquantized"


class Predictor(BasePredictor):
    def setup(self) -> None:
        """Install local packages and instantiate the pipeline from baked-in weights."""
        setup_start = time.time()
        logger.info("=" * 60)
        logger.info("SETUP STARTED")
        logger.info("=" * 60)

        # Log environment info
        logger.info(f"Python version: {sys.version}")
        logger.info(f"PyTorch version: {torch.__version__}")
        logger.info(f"CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.info(f"CUDA device: {torch.cuda.get_device_name(0)}")
            logger.info(
                f"CUDA memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB"
            )
        logger.info(f"Working directory: {os.getcwd()}")

        # Check weight files exist
        logger.info("--- Checking weight files ---")
        weight_files = [
            f"{WEIGHTS_DIR}/ltx-2.3-22b-distilled.safetensors",
            f"{WEIGHTS_DIR}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            f"{WEIGHTS_DIR}/gemma-3-12b",
        ]
        for wf in weight_files:
            exists = os.path.exists(wf)
            if exists and os.path.isfile(wf):
                size_gb = os.path.getsize(wf) / (1024**3)
                logger.info(f"  {wf}: EXISTS ({size_gb:.2f} GB)")
            elif exists and os.path.isdir(wf):
                logger.info(f"  {wf}: EXISTS (directory)")
                try:
                    for item in sorted(os.listdir(wf)):
                        item_path = os.path.join(wf, item)
                        if os.path.isfile(item_path):
                            size_mb = os.path.getsize(item_path) / (1024**2)
                            logger.info(f"    {item}: {size_mb:.1f} MB")
                        else:
                            logger.info(f"    {item}/ (dir)")
                except Exception as e:
                    logger.error(f"    Error listing directory: {e}")
            else:
                logger.error(f"  {wf}: MISSING!")

        # Install local packages
        logger.info("--- Installing local packages ---")
        t0 = time.time()
        subprocess.run(
            [
                "pip",
                "install",
                "/src/packages/ltx-core",
                "/src/packages/ltx-pipelines",
            ],
            check=True,
        )
        logger.info(f"--- Package install completed in {time.time() - t0:.1f}s ---")

        logger.info("--- Importing pipeline modules ---")
        t0 = time.time()
        from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.media_io import encode_video

        logger.info(f"--- Imports completed in {time.time() - t0:.1f}s ---")

        self._TilingConfig = TilingConfig
        self._get_video_chunks_number = get_video_chunks_number
        self._encode_video = encode_video

        logger.info("--- Creating DistilledPipeline ---")
        t0 = time.time()
        self.pipeline = DistilledPipeline(
            distilled_checkpoint_path=f"{WEIGHTS_DIR}/ltx-2.3-22b-distilled.safetensors",
            gemma_root=f"{WEIGHTS_DIR}/gemma-3-12b",
            spatial_upsampler_path=f"{WEIGHTS_DIR}/ltx-2.3-spatial-upscaler-x2-1.0.safetensors",
            loras=[],
        )
        logger.info(
            f"--- DistilledPipeline created in {time.time() - t0:.1f}s ---"
        )

        logger.info("=" * 60)
        logger.info(
            f"SETUP COMPLETED in {time.time() - setup_start:.1f}s"
        )
        logger.info("=" * 60)
        sys.stdout.flush()

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
        predict_start = time.time()
        logger.info("=" * 60)
        logger.info("PREDICT STARTED")
        logger.info(
            f"  prompt={prompt!r}, width={width}, height={height}, "
            f"num_frames={num_frames}, frame_rate={frame_rate}, "
            f"seed={seed}, enhance_prompt={enhance_prompt}"
        )
        logger.info("=" * 60)

        self._validate_inputs(width, height, num_frames)

        if seed == -1:
            seed = random.randint(0, 2**32 - 1)
            logger.info(f"Generated random seed: {seed}")

        logger.info("--- Running pipeline ---")
        t0 = time.time()
        video_iterator, audio = self.pipeline(
            prompt=prompt,
            seed=seed,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            images=[],
            tiling_config=self._TilingConfig.default(),
            enhance_prompt=enhance_prompt,
        )
        logger.info(f"--- Pipeline returned in {time.time() - t0:.1f}s ---")

        video_chunks_number = self._get_video_chunks_number(
            num_frames, self._TilingConfig.default()
        )
        logger.info(f"Video chunks number: {video_chunks_number}")

        output_path = tempfile.mktemp(suffix=".mp4")
        logger.info(f"--- Encoding video to {output_path} ---")
        t0 = time.time()
        self._encode_video(
            video=video_iterator,
            fps=int(frame_rate),
            audio=audio,
            output_path=output_path,
            video_chunks_number=video_chunks_number,
        )
        logger.info(f"--- Video encoded in {time.time() - t0:.1f}s ---")

        logger.info(
            f"PREDICT COMPLETED in {time.time() - predict_start:.1f}s"
        )
        sys.stdout.flush()
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
