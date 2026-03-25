"""Custom IP-Adapter mixin using DINOv2 as the image encoder.

The standard diffusers ``IPAdapterMixin`` uses a CLIP image encoder.
CtrlRegen replaces it with ``facebook/dinov2-giant`` for richer
semantic features.  This mixin provides ``load_ctrlregen_ip_adapter``
which handles the custom weight format and encoder swap.

Attribution:
    Adapted from https://github.com/yepengliu/CtrlRegen .
"""

from __future__ import annotations

import logging

import torch
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from diffusers.utils import (
    _get_model_file,
    is_accelerate_available,
    is_torch_version,
)
from diffusers.utils import (
    logging as diffusers_logging,
)
from huggingface_hub.utils import validate_hf_hub_args
from safetensors import safe_open
from transformers import AutoImageProcessor, AutoModel

logger = logging.getLogger(__name__)
_diffusers_logger = diffusers_logging.get_logger(__name__)

DINOV2_MODEL_ID = "facebook/dinov2-giant"


class CustomIPAdapterMixin:
    """Mixin that adds ``load_ctrlregen_ip_adapter`` to a diffusers pipeline."""

    @validate_hf_hub_args
    def load_ctrlregen_ip_adapter(
        self,
        pretrained_model_name_or_path_or_dict: str | list[str] | dict[str, torch.Tensor],
        subfolder: str | list[str],
        weight_name: str | list[str],
        image_encoder_folder: str | None = "image_encoder",
        **kwargs,
    ) -> None:
        """Load CtrlRegen IP-Adapter weights and DINOv2 image encoder.

        Parameters mirror ``IPAdapterMixin.load_ip_adapter`` but the
        image encoder is always ``facebook/dinov2-giant`` regardless of
        the ``image_encoder_folder`` value in the checkpoint.
        """
        if not isinstance(weight_name, list):
            weight_name = [weight_name]
        if not isinstance(pretrained_model_name_or_path_or_dict, list):
            pretrained_model_name_or_path_or_dict = [pretrained_model_name_or_path_or_dict]
        if len(pretrained_model_name_or_path_or_dict) == 1:
            pretrained_model_name_or_path_or_dict = pretrained_model_name_or_path_or_dict * len(weight_name)
        if not isinstance(subfolder, list):
            subfolder = [subfolder]
        if len(subfolder) == 1:
            subfolder = subfolder * len(weight_name)

        if len(weight_name) != len(pretrained_model_name_or_path_or_dict):
            raise ValueError("`weight_name` and `pretrained_model_name_or_path_or_dict` must have the same length.")
        if len(weight_name) != len(subfolder):
            raise ValueError("`weight_name` and `subfolder` must have the same length.")

        cache_dir = kwargs.pop("cache_dir", None)
        force_download = kwargs.pop("force_download", False)
        kwargs.pop("resume_download", None)
        proxies = kwargs.pop("proxies", None)
        local_files_only = kwargs.pop("local_files_only", None)
        token = kwargs.pop("token", None)
        revision = kwargs.pop("revision", None)
        low_cpu_mem_usage = kwargs.pop("low_cpu_mem_usage", _LOW_CPU_MEM_USAGE_DEFAULT)

        if low_cpu_mem_usage and not is_accelerate_available():
            low_cpu_mem_usage = False
            _diffusers_logger.warning(
                "Cannot initialize model with low cpu memory usage because "
                "`accelerate` was not found. Defaulting to "
                "`low_cpu_mem_usage=False`."
            )

        if low_cpu_mem_usage is True and not is_torch_version(">=", "1.9.0"):
            raise NotImplementedError("Low memory initialization requires torch >= 1.9.0.")

        user_agent = {
            "file_type": "attn_procs_weights",
            "framework": "pytorch",
        }

        state_dicts: list[dict] = []
        for path_or_dict, wn, sf in zip(pretrained_model_name_or_path_or_dict, weight_name, subfolder):
            if not isinstance(path_or_dict, dict):
                model_file = _get_model_file(
                    path_or_dict,
                    weights_name=wn,
                    cache_dir=cache_dir,
                    force_download=force_download,
                    proxies=proxies,
                    local_files_only=local_files_only,
                    token=token,
                    revision=revision,
                    subfolder=sf,
                    user_agent=user_agent,
                )
                if wn.endswith(".safetensors"):
                    state_dict: dict = {"image_proj": {}, "ip_adapter": {}}
                    with safe_open(model_file, framework="pt", device="cpu") as f:
                        for key in f.keys():
                            if key.startswith("image_proj."):
                                state_dict["image_proj"][key.replace("image_proj.", "")] = f.get_tensor(key)
                            elif key.startswith("ip_adapter."):
                                state_dict["ip_adapter"][key.replace("ip_adapter.", "")] = f.get_tensor(key)
                else:
                    state_dict = torch.load(model_file, map_location="cpu")
            else:
                state_dict = path_or_dict

            keys = list(state_dict.keys())
            if keys != ["image_proj", "ip_adapter"]:
                raise ValueError("Required keys (`image_proj` and `ip_adapter`) missing from the state dict.")
            state_dicts.append(state_dict)

        # Always use DINOv2-giant as the image encoder.
        if hasattr(self, "image_encoder") and getattr(self, "image_encoder", None) is None:
            if image_encoder_folder is not None:
                logger.info("Loading DINOv2-giant image encoder for CtrlRegen")
                enc_dtype = getattr(self, "dtype", torch.float32)  # type: ignore[attr-defined]
                image_encoder = AutoModel.from_pretrained(DINOV2_MODEL_ID).to(
                    self.device,
                    dtype=enc_dtype,  # type: ignore[attr-defined]
                )
                self.register_modules(image_encoder=image_encoder)  # type: ignore[attr-defined]

        if hasattr(self, "feature_extractor") and getattr(self, "feature_extractor", None) is None:
            feature_extractor = AutoImageProcessor.from_pretrained(DINOV2_MODEL_ID)
            self.register_modules(feature_extractor=feature_extractor)  # type: ignore[attr-defined]

        unet = (
            getattr(self, self.unet_name)  # type: ignore[attr-defined]
            if not hasattr(self, "unet")
            else self.unet  # type: ignore[attr-defined]
        )
        unet._load_ip_adapter_weights(state_dicts, low_cpu_mem_usage=low_cpu_mem_usage)
