import logging
import time
from dataclasses import replace

import torch

logger = logging.getLogger("ltx-pipelines.model_ledger")

from ltx_core.loader import SDOps
from ltx_core.loader.primitives import LoraPathStrengthAndSDOps
from ltx_core.loader.registry import DummyRegistry, Registry
from ltx_core.loader.single_gpu_model_builder import SingleGPUModelBuilder as Builder
from ltx_core.model.audio_vae import (
    AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
    AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
    VOCODER_COMFY_KEYS_FILTER,
    AudioDecoder,
    AudioDecoderConfigurator,
    AudioEncoder,
    AudioEncoderConfigurator,
    Vocoder,
    VocoderConfigurator,
)
from ltx_core.model.transformer import (
    LTXV_MODEL_COMFY_RENAMING_MAP,
    LTXModelConfigurator,
    X0Model,
)
from ltx_core.model.upsampler import LatentUpsampler, LatentUpsamplerConfigurator
from ltx_core.model.video_vae import (
    VAE_DECODER_COMFY_KEYS_FILTER,
    VAE_ENCODER_COMFY_KEYS_FILTER,
    VideoDecoder,
    VideoDecoderConfigurator,
    VideoEncoder,
    VideoEncoderConfigurator,
)
from ltx_core.quantization import QuantizationPolicy
from ltx_core.text_encoders.gemma import (
    EMBEDDINGS_PROCESSOR_KEY_OPS,
    GEMMA_LLM_KEY_OPS,
    GEMMA_MODEL_OPS,
    EmbeddingsProcessor,
    EmbeddingsProcessorConfigurator,
    GemmaTextEncoder,
    GemmaTextEncoderConfigurator,
    module_ops_from_gemma_root,
)
from ltx_core.utils import find_matching_file


class ModelLedger:
    """
    Central coordinator for loading and building models used in an LTX pipeline.
    The ledger wires together multiple model builders (transformer, video VAE encoder/decoder,
    audio VAE decoder, vocoder, text encoder, and optional latent upsampler) and exposes
    factory methods for constructing model instances.
    ### Model Building
    Each model method (e.g. :meth:`transformer`, :meth:`video_decoder`, :meth:`text_encoder`)
    constructs a new model instance on each call. The builder uses the
    :class:`~ltx_core.loader.registry.Registry` to load weights from the checkpoint,
    instantiates the model with the configured ``dtype``, and moves it to ``self.device``.
    .. note::
        Models are **not cached**. Each call to a model method creates a new instance.
        Callers are responsible for storing references to models they wish to reuse
        and for freeing GPU memory (e.g. by deleting references and calling
        ``torch.cuda.empty_cache()``).
    ### Constructor parameters
    dtype:
        Torch dtype used when constructing all models (e.g. ``torch.bfloat16``).
    device:
        Target device to which models are moved after construction (e.g. ``torch.device("cuda")``).
    checkpoint_path:
        Path to a checkpoint directory or file containing the core model weights
        (transformer, video VAE, audio VAE, text encoder, vocoder). If ``None``, the
        corresponding builders are not created and calling those methods will raise
        a :class:`ValueError`.
    gemma_root_path:
        Base path to Gemma-compatible CLIP/text encoder weights. Required to
        initialize the text encoder builder; if omitted, :meth:`text_encoder` cannot be used.
    spatial_upsampler_path:
        Optional path to a latent upsampler checkpoint. If provided, the
        :meth:`spatial_upsampler` method becomes available; otherwise calling it raises
        a :class:`ValueError`.
    loras:
        Tuple of LoRA configurations (path, strength, sd_ops) applied on top of the base
        transformer weights. Use ``()`` for none.
    registry:
        Optional :class:`Registry` instance for weight caching across builders.
        Defaults to :class:`DummyRegistry` which performs no cross-builder caching.
    quantization:
        Optional :class:`QuantizationPolicy` controlling how transformer weights
        are stored and how matmul is executed. Defaults to None, which means no quantization.
    ### Creating Variants
    Use :meth:`with_additional_loras` to create a new ``ModelLedger`` instance that
    includes additional LoRA configurations or :meth:`with_loras` to replace existing
    lora configurations while sharing the same registry for weight caching.
    """

    def __init__(
        self,
        dtype: torch.dtype,
        device: torch.device,
        checkpoint_path: str | None = None,
        gemma_root_path: str | None = None,
        spatial_upsampler_path: str | None = None,
        loras: tuple[LoraPathStrengthAndSDOps, ...] = (),
        registry: Registry | None = None,
        quantization: QuantizationPolicy | None = None,
    ):
        logger.info("ModelLedger.__init__ starting")
        logger.info(f"  dtype={dtype}, device={device}")
        logger.info(f"  checkpoint_path={checkpoint_path}")
        logger.info(f"  gemma_root_path={gemma_root_path}")
        logger.info(f"  spatial_upsampler_path={spatial_upsampler_path}")
        logger.info(f"  loras={loras}")
        logger.info(f"  quantization={quantization}")
        self.dtype = dtype
        self.device = device
        self.checkpoint_path = checkpoint_path
        self.gemma_root_path = gemma_root_path
        self.spatial_upsampler_path = spatial_upsampler_path
        self.loras = loras
        self.registry = registry or DummyRegistry()
        self.quantization = quantization
        self.build_model_builders()
        logger.info("ModelLedger.__init__ complete")

    def build_model_builders(self) -> None:
        logger.info("build_model_builders starting")
        t_total = time.time()

        if self.checkpoint_path is not None:
            logger.info("Creating transformer builder...")
            t0 = time.time()
            self.transformer_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=LTXModelConfigurator,
                model_sd_ops=LTXV_MODEL_COMFY_RENAMING_MAP,
                loras=tuple(self.loras),
                registry=self.registry,
            )
            logger.info(f"  transformer builder created in {time.time() - t0:.1f}s")

            logger.info("Creating VAE decoder builder...")
            t0 = time.time()
            self.vae_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoDecoderConfigurator,
                model_sd_ops=VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )
            logger.info(f"  VAE decoder builder created in {time.time() - t0:.1f}s")

            logger.info("Creating VAE encoder builder...")
            t0 = time.time()
            self.vae_encoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VideoEncoderConfigurator,
                model_sd_ops=VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )
            logger.info(f"  VAE encoder builder created in {time.time() - t0:.1f}s")

            logger.info("Creating audio encoder builder...")
            t0 = time.time()
            self.audio_encoder_builder = Builder[AudioEncoder](
                model_path=self.checkpoint_path,
                model_class_configurator=AudioEncoderConfigurator,
                model_sd_ops=AUDIO_VAE_ENCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )
            logger.info(f"  audio encoder builder created in {time.time() - t0:.1f}s")

            logger.info("Creating audio decoder builder...")
            t0 = time.time()
            self.audio_decoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=AudioDecoderConfigurator,
                model_sd_ops=AUDIO_VAE_DECODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )
            logger.info(f"  audio decoder builder created in {time.time() - t0:.1f}s")

            logger.info("Creating vocoder builder...")
            t0 = time.time()
            self.vocoder_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=VocoderConfigurator,
                model_sd_ops=VOCODER_COMFY_KEYS_FILTER,
                registry=self.registry,
            )
            logger.info(f"  vocoder builder created in {time.time() - t0:.1f}s")

            # Embeddings processor only needs the LTX checkpoint (no Gemma weights)
            logger.info("Creating embeddings processor builder...")
            t0 = time.time()
            self.embeddings_processor_builder = Builder(
                model_path=self.checkpoint_path,
                model_class_configurator=EmbeddingsProcessorConfigurator,
                model_sd_ops=EMBEDDINGS_PROCESSOR_KEY_OPS,
                registry=self.registry,
            )
            logger.info(f"  embeddings processor builder created in {time.time() - t0:.1f}s")

            if self.gemma_root_path is not None:
                logger.info(f"Setting up Gemma text encoder from {self.gemma_root_path}...")
                t0 = time.time()
                logger.info("  Loading module_ops_from_gemma_root...")
                module_ops = module_ops_from_gemma_root(self.gemma_root_path)
                logger.info(f"  module_ops loaded in {time.time() - t0:.1f}s")

                t0 = time.time()
                logger.info("  Finding model*.safetensors...")
                model_folder = find_matching_file(self.gemma_root_path, "model*.safetensors").parent
                weight_paths = [str(p) for p in model_folder.rglob("*.safetensors")]
                logger.info(f"  Found {len(weight_paths)} safetensors files in {model_folder}")
                for wp in weight_paths:
                    logger.info(f"    {wp}")

                logger.info("Creating text encoder builder...")
                self.text_encoder_builder = Builder(
                    model_path=tuple(weight_paths),
                    model_class_configurator=GemmaTextEncoderConfigurator,
                    model_sd_ops=GEMMA_LLM_KEY_OPS,
                    registry=self.registry,
                    module_ops=(GEMMA_MODEL_OPS, *module_ops),
                )
                logger.info(f"  text encoder builder created in {time.time() - t0:.1f}s")

        if self.spatial_upsampler_path is not None:
            logger.info("Creating spatial upsampler builder...")
            t0 = time.time()
            self.upsampler_builder = Builder(
                model_path=self.spatial_upsampler_path,
                model_class_configurator=LatentUpsamplerConfigurator,
                registry=self.registry,
            )
            logger.info(f"  upsampler builder created in {time.time() - t0:.1f}s")

        logger.info(f"build_model_builders complete in {time.time() - t_total:.1f}s")

    def _target_device(self) -> torch.device:
        if isinstance(self.registry, DummyRegistry) or self.registry is None:
            return self.device
        else:
            return torch.device("cpu")

    def with_additional_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> "ModelLedger":
        """Add new lora configurations to the existing ones."""
        return self.with_loras((*self.loras, *loras))

    def with_loras(self, loras: tuple[LoraPathStrengthAndSDOps, ...]) -> "ModelLedger":
        """Replace existing lora configurations with new ones."""
        return ModelLedger(
            dtype=self.dtype,
            device=self.device,
            checkpoint_path=self.checkpoint_path,
            gemma_root_path=self.gemma_root_path,
            spatial_upsampler_path=self.spatial_upsampler_path,
            loras=loras,
            registry=self.registry,
            quantization=self.quantization,
        )

    def transformer(self) -> X0Model:
        if not hasattr(self, "transformer_builder"):
            raise ValueError(
                "Transformer not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        logger.info("Building transformer model...")
        t0 = time.time()
        if self.quantization is None:
            result = (
                X0Model(self.transformer_builder.build(device=self._target_device(), dtype=self.dtype))
                .to(self.device)
                .eval()
            )
        else:
            sd_ops = self.transformer_builder.model_sd_ops
            if self.quantization.sd_ops is not None:
                sd_ops = SDOps(
                    name=f"sd_ops_chain_{sd_ops.name}+{self.quantization.sd_ops.name}",
                    mapping=(*sd_ops.mapping, *self.quantization.sd_ops.mapping),
                )
            builder = replace(
                self.transformer_builder,
                module_ops=(*self.transformer_builder.module_ops, *self.quantization.module_ops),
                model_sd_ops=sd_ops,
            )
            result = X0Model(builder.build(device=self._target_device())).to(self.device).eval()
        logger.info(f"Transformer built in {time.time() - t0:.1f}s")
        return result

    def video_decoder(self) -> VideoDecoder:
        if not hasattr(self, "vae_decoder_builder"):
            raise ValueError(
                "Video decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        logger.info("Building video decoder...")
        t0 = time.time()
        result = self.vae_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        logger.info(f"Video decoder built in {time.time() - t0:.1f}s")
        return result

    def video_encoder(self) -> VideoEncoder:
        if not hasattr(self, "vae_encoder_builder"):
            raise ValueError(
                "Video encoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        logger.info("Building video encoder...")
        t0 = time.time()
        result = self.vae_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        logger.info(f"Video encoder built in {time.time() - t0:.1f}s")
        return result

    def text_encoder(self) -> GemmaTextEncoder:
        if not hasattr(self, "text_encoder_builder"):
            raise ValueError(
                "Text encoder not initialized. Please provide a checkpoint path and gemma root path to the "
                "ModelLedger constructor."
            )

        logger.info("Building text encoder (Gemma)...")
        t0 = time.time()
        result = self.text_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        logger.info(f"Text encoder built in {time.time() - t0:.1f}s")
        return result

    def gemma_embeddings_processor(self) -> EmbeddingsProcessor:
        if not hasattr(self, "embeddings_processor_builder"):
            raise ValueError(
                "Embeddings processor not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        logger.info("Building embeddings processor...")
        t0 = time.time()
        result = (
            self.embeddings_processor_builder.build(device=self._target_device(), dtype=self.dtype)
            .to(self.device)
            .eval()
        )
        logger.info(f"Embeddings processor built in {time.time() - t0:.1f}s")
        return result

    def audio_encoder(self) -> AudioEncoder:
        if not hasattr(self, "audio_encoder_builder"):
            raise ValueError(
                "Audio encoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        logger.info("Building audio encoder...")
        t0 = time.time()
        result = self.audio_encoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        logger.info(f"Audio encoder built in {time.time() - t0:.1f}s")
        return result

    def audio_decoder(self) -> AudioDecoder:
        if not hasattr(self, "audio_decoder_builder"):
            raise ValueError(
                "Audio decoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        logger.info("Building audio decoder...")
        t0 = time.time()
        result = self.audio_decoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        logger.info(f"Audio decoder built in {time.time() - t0:.1f}s")
        return result

    def vocoder(self) -> Vocoder:
        if not hasattr(self, "vocoder_builder"):
            raise ValueError(
                "Vocoder not initialized. Please provide a checkpoint path to the ModelLedger constructor."
            )

        logger.info("Building vocoder...")
        t0 = time.time()
        result = self.vocoder_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        logger.info(f"Vocoder built in {time.time() - t0:.1f}s")
        return result

    def spatial_upsampler(self) -> LatentUpsampler:
        if not hasattr(self, "upsampler_builder"):
            raise ValueError("Upsampler not initialized. Please provide upsampler path to the ModelLedger constructor.")

        logger.info("Building spatial upsampler...")
        t0 = time.time()
        result = self.upsampler_builder.build(device=self._target_device(), dtype=self.dtype).to(self.device).eval()
        logger.info(f"Spatial upsampler built in {time.time() - t0:.1f}s")
        return result
