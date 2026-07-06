import os
from typing import Any, List, Optional, Tuple, Union
import torch
import numpy as np

from library import krea2_utils
import library.accelerator_setup as accelerator_setup
import library.device_utils as device_utils
from library.qwen_image_autoencoder_kl import AutoencoderKLQwenImage
from library.krea2_text_encoder import Qwen3VLConditioner, TextEncoderConfig
from library import caching
from library.strategy_base import LatentsCachingStrategy, TextEncodingStrategy, TokenizeStrategy, TextEncoderOutputsCachingStrategy

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


class Krea2TokenizeStrategy(TokenizeStrategy):
    # K2's Qwen3VLConditioner (library.krea2_text_encoder) owns its tokenizer + processor
    # internally -- it builds the combined prefix/suffix input_ids itself from raw text, so
    # there is no separate stateless tokenize step to split out. tokenize() here just wraps
    # the captions so they flow through the same TokenizeStrategy/TextEncodingStrategy
    # interface the rest of sd-scripts uses.
    def __init__(self, tokenizer_cache_dir: Optional[str] = None) -> None:
        pass

    def tokenize(self, text: Union[str, List[str]]) -> List[Any]:
        text = [text] if isinstance(text, str) else text
        return [text]


class Krea2TextEncodingStrategy(TextEncodingStrategy):
    def __init__(self) -> None:
        pass

    def encode_tokens(self, tokenize_strategy: TokenizeStrategy, models: List[Any], tokens: List[Any]) -> List[torch.Tensor]:
        captions = tokens[0]
        conditioner: Qwen3VLConditioner = models[0]
        # get_krea2_prompt_embeds handles no_grad / autocast internally (Qwen3VLConditioner.forward).
        hiddens, mask = krea2_utils.get_krea2_prompt_embeds(conditioner, captions)
        return [hiddens, mask]


class Krea2TextEncoderOutputsCachingStrategy(TextEncoderOutputsCachingStrategy):
    KREA2_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX = "_k2_te.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool, is_partial: bool = False) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check, is_partial)

    def get_outputs_npz_path(self, image_abs_path: str) -> str:
        return os.path.splitext(image_abs_path)[0] + Krea2TextEncoderOutputsCachingStrategy.KREA2_TEXT_ENCODER_OUTPUTS_NPZ_SUFFIX

    def is_disk_cached_outputs_expected(self, npz_path: str) -> bool:
        if not self.cache_to_disk:
            return False
        if not os.path.exists(npz_path):
            return False
        if self.skip_disk_cache_validity_check:
            return True

        try:
            npz = np.load(npz_path)
            if "hiddens" not in npz:
                return False
            if "mask" not in npz:
                return False
        except Exception as e:
            logger.error(f"Error loading file: {npz_path}")
            raise e

        return True

    def load_outputs_npz(self, npz_path: str) -> List[np.ndarray]:
        data = np.load(npz_path)
        hiddens = data["hiddens"]  # (valid_len, num_layers*dim), flattened -- see cache_batch_outputs
        mask = data["mask"]  # (valid_len,) all ones
        return [hiddens, mask]

    def cache_batch_outputs(
        self, tokenize_strategy: TokenizeStrategy, models: List[Any], text_encoding_strategy: TextEncodingStrategy, infos: List
    ):
        # Cache *valid-only* (padding dropped) hidden states, flattened to 2D (valid_len, L*D):
        #  - valid-only matches musubi's on-disk format and is far smaller than caching the
        #    encoder's full fixed max_length (~500 tokens) for every caption.
        #  - flattened to 2D because sd-scripts' generic per-sample padding/collate
        #    (library.dataset.none_or_stack_elements) pads dim 0 of a 2D tensor; a 3D
        #    (seq, L, D) tensor would get padded on the wrong axis. krea2_train_network.py
        #    reshapes back to (B, max_len, L, D) after the batch is collated.
        # The companion "mask" (all ones, same length as hiddens) lets the same generic
        # collate reconstruct which tokens in the zero-padded batch are valid vs padding.
        krea2_text_encoding_strategy: Krea2TextEncodingStrategy = text_encoding_strategy
        captions = [info.caption for info in infos]

        tokens = tokenize_strategy.tokenize(captions)
        with torch.no_grad():
            hiddens, mask = krea2_text_encoding_strategy.encode_tokens(tokenize_strategy, models, tokens)

        if hiddens.dtype == torch.bfloat16:
            hiddens = hiddens.float()
        mask = mask.bool()

        for i, info in enumerate(infos):
            valid = mask[i]
            hiddens_i = hiddens[i][valid]  # (valid_len, L, D)
            hiddens_i = hiddens_i.reshape(hiddens_i.shape[0], -1)  # (valid_len, L*D)
            hiddens_i = hiddens_i.cpu().numpy()
            mask_i = np.ones((hiddens_i.shape[0],), dtype=np.float32)

            if self.cache_to_disk:
                np.savez(info.text_encoder_outputs_npz, hiddens=hiddens_i, mask=mask_i)
            else:
                info.text_encoder_outputs = (hiddens_i, mask_i)


class Krea2LatentsCachingStrategy(LatentsCachingStrategy):
    KREA2_LATENTS_NPZ_SUFFIX = "_k2.npz"

    def __init__(self, cache_to_disk: bool, batch_size: int, skip_disk_cache_validity_check: bool) -> None:
        super().__init__(cache_to_disk, batch_size, skip_disk_cache_validity_check)

    @property
    def cache_suffix(self) -> str:
        return Krea2LatentsCachingStrategy.KREA2_LATENTS_NPZ_SUFFIX

    def get_latents_npz_path(self, absolute_path: str, image_size: Tuple[int, int]) -> str:
        return os.path.splitext(absolute_path)[0] + f"_{image_size[0]:04d}x{image_size[1]:04d}" + Krea2LatentsCachingStrategy.KREA2_LATENTS_NPZ_SUFFIX

    def is_disk_cached_latents_expected(self, bucket_reso: Tuple[int, int], npz_path: str, flip_aug: bool, alpha_mask: bool):
        # Qwen-Image VAE spatial compression is 8x (like SDXL/FLUX), not 32x like HunyuanVideo/HunyuanImage.
        return self._default_is_disk_cached_latents_expected(8, bucket_reso, npz_path, flip_aug, alpha_mask, multi_resolution=True)

    def load_latents_from_disk(self, npz_path: str, bucket_reso: Tuple[int, int]):
        return self._default_load_latents_from_disk(8, npz_path, bucket_reso)

    # TODO remove circular dependency for ImageInfo
    def cache_batch_latents(self, vae: AutoencoderKLQwenImage, image_infos: List, flip_aug: bool, alpha_mask: bool, random_crop: bool):
        # Krea 2 uses the Qwen-Image VAE, same normalization/API as HunyuanImage's VAE wrapper:
        # pixels arrive as [B, 3, H, W] in [-1, 1] (from caching.load_images_and_masks_for_caching);
        # encode_pixels_to_latents handles the temporal-dim unsqueeze and mean/std normalization
        # internally, and squeezes it back for 4D input.
        def encode_by_vae(img_tensor):
            nonlocal vae
            with torch.autocast(device_type=vae.device.type, dtype=vae.dtype):
                return vae.encode_pixels_to_latents(img_tensor)

        vae_device = vae.device
        vae_dtype = vae.dtype

        self._default_cache_batch_latents(
            encode_by_vae, vae_device, vae_dtype, image_infos, flip_aug, alpha_mask, random_crop, multi_resolution=True
        )

        if not accelerator_setup.HIGH_VRAM:
            device_utils.clean_memory_on_device(vae.device)
