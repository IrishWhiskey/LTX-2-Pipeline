import json
import logging
import time

import safetensors
import torch

from ltx_core.loader.primitives import StateDict, StateDictLoader
from ltx_core.loader.sd_ops import SDOps

logger = logging.getLogger("ltx-core.sft_loader")


class SafetensorsStateDictLoader(StateDictLoader):
    """
    Loads weights from safetensors files without metadata support.
    Use this for loading raw weight files. For model files that include
    configuration metadata, use SafetensorsModelStateDictLoader instead.
    """

    def metadata(self, path: str) -> dict:
        raise NotImplementedError("Not implemented")

    def load(self, path: str | list[str], sd_ops: SDOps, device: torch.device | None = None) -> StateDict:
        """
        Load state dict from path or paths (for sharded model storage) and apply sd_ops
        """
        sd = {}
        size = 0
        dtype = set()
        device = device or torch.device("cpu")
        model_paths = path if isinstance(path, list) else [path]
        logger.info(f"Loading safetensors from {len(model_paths)} shard(s) to {device}")
        t_total = time.time()
        for i, shard_path in enumerate(model_paths):
            t0 = time.time()
            logger.info(f"  Loading shard {i+1}/{len(model_paths)}: {shard_path}")
            with safetensors.safe_open(shard_path, framework="pt", device=str(device)) as f:
                safetensor_keys = f.keys()
                logger.info(f"    {len(list(safetensor_keys))} keys in shard")
                for name in f.keys():
                    expected_name = name if sd_ops is None else sd_ops.apply_to_key(name)
                    if expected_name is None:
                        continue
                    value = f.get_tensor(name).to(device=device, non_blocking=True, copy=False)
                    key_value_pairs = ((expected_name, value),)
                    if sd_ops is not None:
                        key_value_pairs = sd_ops.apply_to_key_value(expected_name, value)
                    for key, value in key_value_pairs:
                        size += value.nbytes
                        dtype.add(value.dtype)
                        sd[key] = value
            logger.info(f"    Shard loaded in {time.time() - t0:.1f}s ({size / 1024**3:.2f} GB so far)")

        logger.info(
            f"  Total: {len(sd)} keys, {size / 1024**3:.2f} GB, "
            f"dtypes={dtype}, loaded in {time.time() - t_total:.1f}s"
        )
        return StateDict(sd=sd, device=device, size=size, dtype=dtype)


class SafetensorsModelStateDictLoader(StateDictLoader):
    """
    Loads weights and configuration metadata from safetensors model files.
    Unlike SafetensorsStateDictLoader, this loader can read model configuration
    from the safetensors file metadata via the metadata() method.
    """

    def __init__(self, weight_loader: SafetensorsStateDictLoader | None = None):
        self.weight_loader = weight_loader if weight_loader is not None else SafetensorsStateDictLoader()

    def metadata(self, path: str) -> dict:
        with safetensors.safe_open(path, framework="pt") as f:
            meta = f.metadata()
            if meta is None or "config" not in meta:
                return {}
            return json.loads(meta["config"])

    def load(self, path: str | list[str], sd_ops: SDOps | None = None, device: torch.device | None = None) -> StateDict:
        return self.weight_loader.load(path, sd_ops, device)
