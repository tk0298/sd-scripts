# LoRA network module for Krea 2 (K2).
#
# Unlike HunyuanImage/FLUX, K2's recommended default LoRA target is *every* Linear in the
# DiT (matching the model authors' rank 32 / alpha 32 recipe): first, last.linear, the
# per-block attention (wq/wk/wv/wo/gate) and SwiGLU MLP (gate/up/down), the text-fusion
# transformer (its attn/mlp blocks + projector), and the time/text projection MLPs
# (tmlp/txtmlp/tproj). So TARGET_REPLACE_MODULES is None here (not a list of class names),
# which triggers lora_flux.LoRANetwork's "wrap every Linear in the whole model" path.
#
# The modulation (DoubleSharedModulation / SimpleModulation) and all RMSNorm hold raw
# nn.Parameter tensors, not Linear modules, so they are never wrapped -- no explicit
# exclude needed. To reproduce the authors' "long training run" config (attention-only,
# to better preserve prompt adherence), pass --network_args with e.g.:
#   network_reg_dims=".*\.mlp\..*=0,first=0,last\.linear=0,tmlp\..*=0,txtmlp\..*=0,tproj\.1=0,txtfusion\..*=0"
# (regex-dim overrides at dim 0 == skip that module).

import os
from typing import Dict, List, Optional, Type, Union
import torch
import torch.nn as nn
import re

from networks import lora_flux
from library.krea2_models import SingleStreamDiT

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def create_network(
    multiplier: float,
    network_dim: Optional[int],
    network_alpha: Optional[float],
    vae: nn.Module,
    text_encoders: List[nn.Module],
    unet: SingleStreamDiT,
    neuron_dropout: Optional[float] = None,
    **kwargs,
):
    if network_dim is None:
        network_dim = 32  # matches the model authors' recommended default for K2
    if network_alpha is None:
        network_alpha = 32.0

    rank_dropout = kwargs.get("rank_dropout", None)
    if rank_dropout is not None:
        rank_dropout = float(rank_dropout)
    module_dropout = kwargs.get("module_dropout", None)
    if module_dropout is not None:
        module_dropout = float(module_dropout)

    verbose = kwargs.get("verbose", False)
    if verbose is not None:
        verbose = True if verbose == "True" else False

    def parse_kv_pairs(kv_pair_str: str, is_int: bool) -> Dict[str, float]:
        pairs = {}
        for pair in kv_pair_str.split(","):
            pair = pair.strip()
            if not pair:
                continue
            if "=" not in pair:
                logger.warning(f"Invalid format: {pair}, expected 'key=value'")
                continue
            key, value = pair.split("=", 1)
            key, value = key.strip(), value.strip()
            try:
                pairs[key] = int(value) if is_int else float(value)
            except ValueError:
                logger.warning(f"Invalid value for {key}: {value}")
        return pairs

    network_reg_lrs = kwargs.get("network_reg_lrs", None)
    reg_lrs = parse_kv_pairs(network_reg_lrs, is_int=False) if network_reg_lrs is not None else None

    network_reg_dims = kwargs.get("network_reg_dims", None)
    reg_dims = parse_kv_pairs(network_reg_dims, is_int=True) if network_reg_dims is not None else None

    network = Krea2LoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        lora_dim=network_dim,
        alpha=network_alpha,
        dropout=neuron_dropout,
        rank_dropout=rank_dropout,
        module_dropout=module_dropout,
        reg_dims=reg_dims,
        reg_lrs=reg_lrs,
        verbose=verbose,
    )

    loraplus_lr_ratio = kwargs.get("loraplus_lr_ratio", None)
    loraplus_unet_lr_ratio = kwargs.get("loraplus_unet_lr_ratio", None)
    loraplus_text_encoder_lr_ratio = kwargs.get("loraplus_text_encoder_lr_ratio", None)
    loraplus_lr_ratio = float(loraplus_lr_ratio) if loraplus_lr_ratio is not None else None
    loraplus_unet_lr_ratio = float(loraplus_unet_lr_ratio) if loraplus_unet_lr_ratio is not None else None
    loraplus_text_encoder_lr_ratio = float(loraplus_text_encoder_lr_ratio) if loraplus_text_encoder_lr_ratio is not None else None
    if loraplus_lr_ratio is not None or loraplus_unet_lr_ratio is not None or loraplus_text_encoder_lr_ratio is not None:
        network.set_loraplus_lr_ratio(loraplus_lr_ratio, loraplus_unet_lr_ratio, loraplus_text_encoder_lr_ratio)

    return network


# Create network from weights for inference; weights are not loaded here (because they can be merged).
def create_network_from_weights(multiplier, file, ae, text_encoders, unet, weights_sd=None, for_inference=False, **kwargs):
    if weights_sd is None:
        if os.path.splitext(file)[1] == ".safetensors":
            from safetensors.torch import load_file

            weights_sd = load_file(file)
        else:
            weights_sd = torch.load(file, map_location="cpu")

    modules_dim = {}
    modules_alpha = {}
    for key, value in weights_sd.items():
        if "." not in key:
            continue
        lora_name = key.split(".")[0]
        if "alpha" in key:
            modules_alpha[lora_name] = value
        elif "lora_down" in key:
            modules_dim[lora_name] = value.size()[0]

    module_class = lora_flux.LoRAInfModule if for_inference else lora_flux.LoRAModule

    network = Krea2LoRANetwork(
        text_encoders,
        unet,
        multiplier=multiplier,
        modules_dim=modules_dim,
        modules_alpha=modules_alpha,
        module_class=module_class,
    )
    return network, weights_sd


class Krea2LoRANetwork(lora_flux.LoRANetwork):
    # None -> wrap every Linear in the whole DiT (see module docstring above).
    TARGET_REPLACE_MODULES = None
    LORA_PREFIX_KREA2_DIT = "lora_unet"  # ComfyUI-compatible prefix

    def __init__(
        self,
        text_encoders: list,
        unet: SingleStreamDiT,
        multiplier: float = 1.0,
        lora_dim: int = 32,
        alpha: float = 32,
        dropout: Optional[float] = None,
        rank_dropout: Optional[float] = None,
        module_dropout: Optional[float] = None,
        module_class: Type[object] = lora_flux.LoRAModule,
        modules_dim: Optional[Dict[str, int]] = None,
        modules_alpha: Optional[Dict[str, int]] = None,
        reg_dims: Optional[Dict[str, int]] = None,
        reg_lrs: Optional[Dict[str, float]] = None,
        verbose: Optional[bool] = False,
    ) -> None:
        nn.Module.__init__(self)
        self.multiplier = multiplier

        self.lora_dim = lora_dim
        self.alpha = alpha
        self.conv_lora_dim = None
        self.conv_alpha = None
        self.dropout = dropout
        self.rank_dropout = rank_dropout
        self.module_dropout = module_dropout
        self.split_qkv = False
        self.reg_dims = reg_dims
        self.reg_lrs = reg_lrs

        self.loraplus_lr_ratio = None
        self.loraplus_unet_lr_ratio = None
        self.loraplus_text_encoder_lr_ratio = None

        if modules_dim is not None:
            logger.info("create Krea 2 LoRA network from weights")
        else:
            logger.info(f"create Krea 2 LoRA network. base dim (rank): {lora_dim}, alpha: {alpha}")
            logger.info(f"neuron dropout: p={self.dropout}, rank dropout: p={self.rank_dropout}, module dropout: p={self.module_dropout}")

        def create_modules(root_module: nn.Module) -> List[lora_flux.LoRAModule]:
            prefix = self.LORA_PREFIX_KREA2_DIT
            loras = []
            skipped = []
            # TARGET_REPLACE_MODULES is None: wrap every Linear reachable from root_module.
            for child_name, child_module in root_module.named_modules():
                if child_module.__class__.__name__ != "Linear":
                    continue

                lora_name = (prefix + "." + child_name).replace(".", "_")

                dim, alpha = None, None
                if modules_dim is not None:
                    if lora_name in modules_dim:
                        dim, alpha = modules_dim[lora_name], modules_alpha[lora_name]
                elif self.reg_dims is not None:
                    for reg, d in self.reg_dims.items():
                        if re.search(reg, lora_name):
                            dim, alpha = d, self.alpha
                            break

                if dim is None and modules_dim is None:
                    dim, alpha = self.lora_dim, self.alpha

                if dim is None or dim == 0:
                    skipped.append(lora_name)
                    continue

                lora = module_class(
                    lora_name,
                    child_module,
                    self.multiplier,
                    dim,
                    alpha,
                    dropout=dropout,
                    rank_dropout=rank_dropout,
                    module_dropout=module_dropout,
                )
                loras.append(lora)
            return loras, skipped

        self.unet_loras: List[Union[lora_flux.LoRAModule, lora_flux.LoRAInfModule]]
        self.unet_loras, skipped_un = create_modules(unet)
        self.text_encoder_loras = []  # K2 does not train the Qwen3-VL text encoder

        logger.info(f"create LoRA for Krea 2 DiT: {len(self.unet_loras)} modules.")
        if verbose:
            for lora in self.unet_loras:
                logger.info(f"\t{lora.lora_name:50} {lora.lora_dim}, {lora.alpha}")

        if verbose and len(skipped_un) > 0:
            logger.warning(f"because dim (rank) is 0, {len(skipped_un)} LoRA modules are skipped:")
            for name in skipped_un:
                logger.info(f"\t{name}")

        names = set()
        for lora in self.text_encoder_loras + self.unet_loras:
            assert lora.lora_name not in names, f"duplicated lora name: {lora.lora_name}"
            names.add(lora.lora_name)
