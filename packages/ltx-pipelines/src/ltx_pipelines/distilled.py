import logging
import time
from collections.abc import Iterator

import torch

from ltx_core.components.diffusion_steps import EulerDiffusionStep
from ltx_core.components.noisers import GaussianNoiser
from ltx_core.components.protocols import DiffusionStepProtocol
from ltx_core.loader import LoraPathStrengthAndSDOps
from ltx_core.model.audio_vae import decode_audio as vae_decode_audio
from ltx_core.model.upsampler import upsample_video
from ltx_core.model.video_vae import TilingConfig, get_video_chunks_number
from ltx_core.model.video_vae import decode_video as vae_decode_video
from ltx_core.quantization import QuantizationPolicy
from ltx_core.types import Audio, LatentState, VideoPixelShape
from ltx_pipelines.utils import ModelLedger, euler_denoising_loop
from ltx_pipelines.utils.args import (
    ImageConditioningInput,
    default_2_stage_distilled_arg_parser,
    detect_checkpoint_path,
)
from ltx_pipelines.utils.constants import (
    DISTILLED_SIGMA_VALUES,
    STAGE_2_DISTILLED_SIGMA_VALUES,
    detect_params,
)
from ltx_pipelines.utils.helpers import (
    assert_resolution,
    cleanup_memory,
    combined_image_conditionings,
    denoise_audio_video,
    encode_prompts,
    get_device,
    simple_denoising_func,
)
from ltx_pipelines.utils.media_io import encode_video
from ltx_pipelines.utils.types import PipelineComponents

device = get_device()
logger = logging.getLogger("ltx-pipelines.distilled")


class DistilledPipeline:
    """
    Two-stage distilled video generation pipeline.
    Stage 1 generates video at half of the target resolution, then Stage 2 upsamples
    by 2x and refines with additional denoising steps for higher quality output.
    """

    def __init__(
        self,
        distilled_checkpoint_path: str,
        gemma_root: str,
        spatial_upsampler_path: str,
        loras: list[LoraPathStrengthAndSDOps],
        device: torch.device = device,
        quantization: QuantizationPolicy | None = None,
    ):
        logger.info("DistilledPipeline.__init__ starting")
        logger.info(f"  distilled_checkpoint_path={distilled_checkpoint_path}")
        logger.info(f"  gemma_root={gemma_root}")
        logger.info(f"  spatial_upsampler_path={spatial_upsampler_path}")
        logger.info(f"  loras={loras}")
        logger.info(f"  device={device}")
        logger.info(f"  quantization={quantization}")

        self.device = device
        self.dtype = torch.bfloat16

        logger.info("Creating ModelLedger...")
        t0 = time.time()
        self.model_ledger = ModelLedger(
            dtype=self.dtype,
            device=device,
            checkpoint_path=distilled_checkpoint_path,
            spatial_upsampler_path=spatial_upsampler_path,
            gemma_root_path=gemma_root,
            loras=loras,
            quantization=quantization,
        )
        logger.info(f"ModelLedger created in {time.time() - t0:.1f}s")

        logger.info("Creating PipelineComponents...")
        t0 = time.time()
        self.pipeline_components = PipelineComponents(
            dtype=self.dtype,
            device=device,
        )
        logger.info(f"PipelineComponents created in {time.time() - t0:.1f}s")
        logger.info("DistilledPipeline.__init__ complete")

    def __call__(
        self,
        prompt: str,
        seed: int,
        height: int,
        width: int,
        num_frames: int,
        frame_rate: float,
        images: list[ImageConditioningInput],
        tiling_config: TilingConfig | None = None,
        enhance_prompt: bool = False,
    ) -> tuple[Iterator[torch.Tensor], Audio]:
        call_start = time.time()
        logger.info("DistilledPipeline.__call__ starting")
        logger.info(f"  prompt={prompt!r}, seed={seed}, {height}x{width}, frames={num_frames}, fps={frame_rate}")
        assert_resolution(height=height, width=width, is_two_stage=True)

        generator = torch.Generator(device=self.device).manual_seed(seed)
        noiser = GaussianNoiser(generator=generator)
        stepper = EulerDiffusionStep()
        dtype = torch.bfloat16

        logger.info("Encoding prompts (text encoder + embeddings processor)...")
        t0 = time.time()
        (ctx_p,) = encode_prompts(
            [prompt],
            self.model_ledger,
            enhance_first_prompt=enhance_prompt,
            enhance_prompt_image=images[0][0] if len(images) > 0 else None,
        )
        logger.info(f"Prompt encoding done in {time.time() - t0:.1f}s")
        video_context, audio_context = ctx_p.video_encoding, ctx_p.audio_encoding

        # Stage 1: Initial low resolution video generation.
        logger.info("Loading video encoder...")
        t0 = time.time()
        video_encoder = self.model_ledger.video_encoder()
        logger.info(f"Video encoder loaded in {time.time() - t0:.1f}s")

        logger.info("Loading transformer...")
        t0 = time.time()
        transformer = self.model_ledger.transformer()
        logger.info(f"Transformer loaded in {time.time() - t0:.1f}s")

        stage_1_sigmas = torch.Tensor(DISTILLED_SIGMA_VALUES).to(self.device)
        logger.info(f"Stage 1 sigmas: {DISTILLED_SIGMA_VALUES}")

        def denoising_loop(
            sigmas: torch.Tensor, video_state: LatentState, audio_state: LatentState, stepper: DiffusionStepProtocol
        ) -> tuple[LatentState, LatentState]:
            return euler_denoising_loop(
                sigmas=sigmas,
                video_state=video_state,
                audio_state=audio_state,
                stepper=stepper,
                denoise_fn=simple_denoising_func(
                    video_context=video_context,
                    audio_context=audio_context,
                    transformer=transformer,  # noqa: F821
                ),
            )

        stage_1_output_shape = VideoPixelShape(
            batch=1,
            frames=num_frames,
            width=width // 2,
            height=height // 2,
            fps=frame_rate,
        )
        logger.info(f"Stage 1 output shape: {stage_1_output_shape}")

        stage_1_conditionings = combined_image_conditionings(
            images=images,
            height=stage_1_output_shape.height,
            width=stage_1_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        logger.info("Stage 1: Denoising audio+video...")
        t0 = time.time()
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_1_output_shape,
            conditionings=stage_1_conditionings,
            noiser=noiser,
            sigmas=stage_1_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
        )
        logger.info(f"Stage 1 denoising done in {time.time() - t0:.1f}s")

        # Stage 2: Upsample and refine the video at higher resolution with distilled LORA.
        logger.info("Loading spatial upsampler and upscaling video...")
        t0 = time.time()
        upscaled_video_latent = upsample_video(
            latent=video_state.latent[:1], video_encoder=video_encoder, upsampler=self.model_ledger.spatial_upsampler()
        )
        logger.info(f"Upsampling done in {time.time() - t0:.1f}s")

        torch.cuda.synchronize()
        cleanup_memory()

        stage_2_sigmas = torch.Tensor(STAGE_2_DISTILLED_SIGMA_VALUES).to(self.device)
        stage_2_output_shape = VideoPixelShape(batch=1, frames=num_frames, width=width, height=height, fps=frame_rate)
        logger.info(f"Stage 2 output shape: {stage_2_output_shape}")

        stage_2_conditionings = combined_image_conditionings(
            images=images,
            height=stage_2_output_shape.height,
            width=stage_2_output_shape.width,
            video_encoder=video_encoder,
            dtype=dtype,
            device=self.device,
        )

        logger.info("Stage 2: Denoising audio+video...")
        t0 = time.time()
        video_state, audio_state = denoise_audio_video(
            output_shape=stage_2_output_shape,
            conditionings=stage_2_conditionings,
            noiser=noiser,
            sigmas=stage_2_sigmas,
            stepper=stepper,
            denoising_loop_fn=denoising_loop,
            components=self.pipeline_components,
            dtype=dtype,
            device=self.device,
            noise_scale=stage_2_sigmas[0],
            initial_video_latent=upscaled_video_latent,
            initial_audio_latent=audio_state.latent,
        )
        logger.info(f"Stage 2 denoising done in {time.time() - t0:.1f}s")

        torch.cuda.synchronize()
        del transformer
        del video_encoder
        cleanup_memory()

        logger.info("Decoding video with VAE...")
        t0 = time.time()
        decoded_video = vae_decode_video(
            video_state.latent, self.model_ledger.video_decoder(), tiling_config, generator
        )
        logger.info(f"Video VAE decode initiated in {time.time() - t0:.1f}s")

        logger.info("Decoding audio with VAE + vocoder...")
        t0 = time.time()
        decoded_audio = vae_decode_audio(
            audio_state.latent, self.model_ledger.audio_decoder(), self.model_ledger.vocoder()
        )
        logger.info(f"Audio decode done in {time.time() - t0:.1f}s")

        logger.info(f"DistilledPipeline.__call__ total: {time.time() - call_start:.1f}s")
        return decoded_video, decoded_audio


@torch.inference_mode()
def main() -> None:
    logging.getLogger().setLevel(logging.INFO)
    checkpoint_path = detect_checkpoint_path(distilled=True)
    params = detect_params(checkpoint_path)
    parser = default_2_stage_distilled_arg_parser(params=params)
    args = parser.parse_args()
    pipeline = DistilledPipeline(
        distilled_checkpoint_path=args.distilled_checkpoint_path,
        spatial_upsampler_path=args.spatial_upsampler_path,
        gemma_root=args.gemma_root,
        loras=tuple(args.lora) if args.lora else (),
        quantization=args.quantization,
    )
    tiling_config = TilingConfig.default()
    video_chunks_number = get_video_chunks_number(args.num_frames, tiling_config)
    video, audio = pipeline(
        prompt=args.prompt,
        seed=args.seed,
        height=args.height,
        width=args.width,
        num_frames=args.num_frames,
        frame_rate=args.frame_rate,
        images=args.images,
        tiling_config=tiling_config,
        enhance_prompt=args.enhance_prompt,
    )

    encode_video(
        video=video,
        fps=args.frame_rate,
        audio=audio,
        output_path=args.output_path,
        video_chunks_number=video_chunks_number,
    )


if __name__ == "__main__":
    main()
