# LoRA training for Krea 2 (K2), ported from kohya-ss/musubi-tuner
# (src/musubi_tuner/krea2_train_network.py) to sd-scripts' train_network.NetworkTrainer.
#
# Scope of this port (first pass): full training loop (bf16 + gradient checkpointing, dynamic
# scaled fp8 for the DiT via --fp8_scaled, block swap via --blocks_to_swap, text encoder output
# caching) and sample-image-during-training on the RAW checkpoint (flow-matching Euler sampler,
# optional CFG).
#
# NOT YET PORTED from musubi: the RAW-train / Turbo-sample base-weight swap feature
# (--turbo_dit / --turbo_dit_cache, on_before/after_sample_images weight stashing). musubi's own
# docs call this "fully optional" (VRAM-permitting convenience) -- the recommended workflow is
# to train on RAW and simply run inference on Turbo afterwards with the saved LoRA, which is
# unaffected by this omission. Can be added later if useful.
#
# Latents are kept as plain (B, C, H, W) here (no video frame axis) -- sd-scripts' image
# datasets and Qwen-Image VAE caching (library.strategy_krea2.Krea2LatentsCachingStrategy) never
# carry musubi's frame dimension.

import argparse
import copy
import gc
import os
import time
from typing import Any, Optional, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from accelerate import Accelerator, PartialState

import library.args as args_util
import library.model_io as model_io
from library.dataset import DatasetGroup, MinimalDataset
from library.device_utils import clean_memory_on_device, init_ipex

init_ipex()

import train_network
from library import (
    flux_train_utils,
    krea2_models,
    krea2_sampling,
    krea2_text_encoder,
    krea2_utils,
    qwen_image_autoencoder_kl,
    sd3_train_utils,
    strategy_base,
    strategy_krea2,
    sampling,
)
from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


# region sampling


def sample_images(
    accelerator: Accelerator,
    args: argparse.Namespace,
    epoch,
    steps,
    dit: krea2_models.SingleStreamDiT,
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    text_encoders,
    sample_prompts_te_outputs,
    prompt_replacement=None,
):
    if steps == 0:
        if not args.sample_at_first:
            return
    else:
        if args.sample_every_n_steps is None and args.sample_every_n_epochs is None:
            return
        if args.sample_every_n_epochs is not None:
            if epoch is None or epoch % args.sample_every_n_epochs != 0:
                return
        else:
            if steps % args.sample_every_n_steps != 0 or epoch is not None:
                return

    logger.info("")
    logger.info(f"generating sample images at step / サンプル画像生成 ステップ: {steps}")
    if not os.path.isfile(args.sample_prompts) and sample_prompts_te_outputs is None:
        logger.error(f"No prompt file / プロンプトファイルがありません: {args.sample_prompts}")
        return

    distributed_state = PartialState()

    dit = accelerator.unwrap_model(dit)
    dit.switch_block_swap_for_inference()
    if text_encoders is not None:
        text_encoders = [(accelerator.unwrap_model(te) if te is not None else None) for te in text_encoders]

    prompts = sampling.load_prompts(args.sample_prompts)

    save_dir = args.output_dir + "/sample"
    os.makedirs(save_dir, exist_ok=True)

    rng_state = torch.get_rng_state()
    cuda_rng_state = None
    try:
        cuda_rng_state = torch.cuda.get_rng_state() if torch.cuda.is_available() else None
    except Exception:
        pass

    if distributed_state.num_processes <= 1:
        with torch.no_grad():
            for prompt_dict in prompts:
                sample_image_inference(
                    accelerator, args, dit, text_encoders, vae, save_dir, prompt_dict, epoch, steps, sample_prompts_te_outputs, prompt_replacement
                )
    else:
        per_process_prompts = []
        for i in range(distributed_state.num_processes):
            per_process_prompts.append(prompts[i :: distributed_state.num_processes])

        with torch.no_grad():
            with distributed_state.split_between_processes(per_process_prompts) as prompt_dict_lists:
                for prompt_dict in prompt_dict_lists[0]:
                    sample_image_inference(
                        accelerator, args, dit, text_encoders, vae, save_dir, prompt_dict, epoch, steps, sample_prompts_te_outputs, prompt_replacement
                    )

    torch.set_rng_state(rng_state)
    if cuda_rng_state is not None:
        torch.cuda.set_rng_state(cuda_rng_state)

    dit.switch_block_swap_for_training()
    clean_memory_on_device(accelerator.device)


def sample_image_inference(
    accelerator: Accelerator,
    args: argparse.Namespace,
    dit: krea2_models.SingleStreamDiT,
    text_encoders: Optional[list],
    vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage,
    save_dir,
    prompt_dict,
    epoch,
    steps,
    sample_prompts_te_outputs,
    prompt_replacement,
):
    assert isinstance(prompt_dict, dict)
    negative_prompt = prompt_dict.get("negative_prompt")
    sample_steps = prompt_dict.get("sample_steps", 28)
    width = prompt_dict.get("width", 1024)
    height = prompt_dict.get("height", 1024)
    cfg_scale = prompt_dict.get("scale", 5.5)  # K2 convention: cfg_scale = guidance + 1 (official default guidance 4.5)
    seed = prompt_dict.get("seed")
    prompt: str = prompt_dict.get("prompt", "")

    if prompt_replacement is not None:
        prompt = prompt.replace(prompt_replacement[0], prompt_replacement[1])
        if negative_prompt is not None:
            negative_prompt = negative_prompt.replace(prompt_replacement[0], prompt_replacement[1])

    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
    else:
        torch.seed()
        torch.cuda.seed()

    if negative_prompt is None:
        negative_prompt = ""

    device = accelerator.device
    patch = dit.config.patch
    compression = 2 ** len(vae.temperal_downsample)  # Qwen-Image VAE: 8x

    logger.info(f"prompt: {prompt}")
    if cfg_scale != 1.0:
        logger.info(f"negative_prompt: {negative_prompt}")
    logger.info(f"height: {height}, width: {width}, sample_steps: {sample_steps}, cfg_scale: {cfg_scale}")
    if seed is not None:
        logger.info(f"seed: {seed}")

    tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
    encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

    def encode_prompt(prpt):
        if sample_prompts_te_outputs and prpt in sample_prompts_te_outputs:
            hiddens, mask = sample_prompts_te_outputs[prpt]
            return hiddens.unsqueeze(0).to(device), mask.unsqueeze(0).to(device)
        tokens = tokenize_strategy.tokenize(prpt)
        hiddens, mask = encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens)
        return hiddens, mask

    do_cfg = cfg_scale > 1.0
    hiddens, mask = encode_prompt(prompt)
    txt, txtmask = krea2_sampling.gather_valid_text(hiddens.to(device=device, dtype=torch.bfloat16), mask.to(device).bool())
    if do_cfg:
        un_hiddens, un_mask = encode_prompt(negative_prompt)
        untxt, untxtmask = krea2_sampling.gather_valid_text(un_hiddens.to(device=device, dtype=torch.bfloat16), un_mask.to(device).bool())

    align = compression * patch
    width = krea2_sampling.roundup(width, align, "width")
    height = krea2_sampling.roundup(height, align, "height")

    generator = torch.Generator(device=device)
    if seed is not None:
        generator.manual_seed(seed)
    noise = torch.randn(
        1, dit.config.channels, height // compression, width // compression, device=device, dtype=torch.bfloat16, generator=generator
    )

    img, pos, mask_ = krea2_sampling.prepare(noise, txt.shape[1], patch, txtmask)
    if do_cfg:
        _, unpos, unmask = krea2_sampling.prepare(noise, untxt.shape[1], patch, untxtmask)

    x1 = (256 // align) ** 2
    x2 = (1280 // align) ** 2
    ts = krea2_sampling.krea2_timesteps(img.shape[1], sample_steps, x1, x2, y1=0.5, y2=1.15)

    dit_is_training = dit.training
    dit.eval()
    with torch.no_grad(), accelerator.autocast():
        for tcurr, tprev in zip(ts[:-1], ts[1:]):
            t = torch.full((1,), tcurr, dtype=img.dtype, device=device)
            cond = dit(img=img, context=txt, t=t, pos=pos, mask=mask_)
            if do_cfg:
                uncond = dit(img=img, context=untxt, t=t, pos=unpos, mask=unmask)
                v = uncond + cfg_scale * (cond - uncond)
            else:
                v = cond
            img = img + (tprev - tcurr) * v
    if dit_is_training:
        dit.train()

    from einops import rearrange

    latent = rearrange(
        img, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=(height // compression) // patch, w=(width // compression) // patch
    )

    org_vae_device = vae.device
    vae.to(device)
    with torch.no_grad():
        pixels = vae.decode_to_pixels(latent.to(vae.dtype))  # (1, C, H, W) in [-1, 1]
    vae.to(org_vae_device)
    clean_memory_on_device(device)

    x = pixels.clamp(-1, 1).permute(0, 2, 3, 1)
    image = Image.fromarray((127.5 * (x + 1.0)).float().cpu().numpy().astype(np.uint8)[0])

    ts_str = time.strftime("%Y%m%d%H%M%S", time.localtime())
    num_suffix = f"e{epoch:06d}" if epoch is not None else f"{steps:06d}"
    seed_suffix = "" if seed is None else f"_{seed}"
    i: int = prompt_dict["enum"]
    img_filename = f"{'' if args.output_name is None else args.output_name + '_'}{num_suffix}_{i:02d}_{ts_str}{seed_suffix}.png"
    image.save(os.path.join(save_dir, img_filename))

    if "wandb" in [tracker.name for tracker in accelerator.trackers]:
        wandb_tracker = accelerator.get_tracker("wandb")
        import wandb

        wandb_tracker.log({f"sample_{i}": wandb.Image(image, caption=prompt)}, commit=False)


# endregion


class Krea2NetworkTrainer(train_network.NetworkTrainer):
    def __init__(self):
        super().__init__()
        self.sample_prompts_te_outputs = None
        self.is_swapping_blocks: bool = False

    def assert_extra_args(
        self,
        args,
        train_dataset_group: Union[DatasetGroup, MinimalDataset],
        val_dataset_group: Optional[DatasetGroup],
    ):
        super().assert_extra_args(args, train_dataset_group, val_dataset_group)

        if args.mixed_precision == "fp16":
            logger.warning("mixed_precision bf16 is recommended for Krea 2 / Krea 2ではmixed_precision bf16が推奨されます")

        # K2 fp8 supports only the scaled (dynamic) path; plain --fp8_base alone would cast the
        # whole DiT (incl. norms/modulation) to fp8, which breaks it.
        if (args.fp8_base or args.fp8_base_unet) and not args.fp8_scaled:
            logger.warning(
                "fp8_base / fp8_base_unet are not supported for Krea 2, use --fp8_scaled instead"
                " / Krea 2ではfp8_base/fp8_base_unetはサポートされていません。--fp8_scaledを使用してください"
            )
        if args.fp8_scaled and (args.fp8_base or args.fp8_base_unet):
            args.fp8_base = False
            args.fp8_base_unet = False

        if args.cache_text_encoder_outputs_to_disk and not args.cache_text_encoder_outputs:
            args.cache_text_encoder_outputs = True

        if args.cache_text_encoder_outputs:
            assert train_dataset_group.is_text_encoder_output_cacheable(), (
                "when caching Text Encoder output, either caption_dropout_rate, shuffle_caption, token_warmup_step or "
                "caption_tag_dropout_rate cannot be used"
            )

        train_dataset_group.verify_bucket_reso_steps(16)
        if val_dataset_group is not None:
            val_dataset_group.verify_bucket_reso_steps(16)

    def load_target_model(self, args, weight_dtype, accelerator):
        self.is_swapping_blocks = args.blocks_to_swap is not None and args.blocks_to_swap > 0

        te_dtype = torch.bfloat16
        te_device = "cpu"  # moved to accelerator.device later in cache_text_encoder_outputs_if_needed
        text_encoder = None
        if args.text_encoder is not None:
            text_encoder = krea2_utils.load_krea2_text_encoder(args.text_encoder, dtype=te_dtype, device=te_device)

        vae = qwen_image_autoencoder_kl.load_vae(args.vae, 3, device="cpu", disable_mmap=args.disable_mmap_load_safetensors)
        vae.to(dtype=torch.bfloat16)
        vae.eval()

        return "krea2", [text_encoder], vae, None  # unet (DiT) is loaded lazily

    def load_unet_lazily(self, args, weight_dtype, accelerator, text_encoders) -> tuple:
        loading_dtype = None if args.fp8_scaled else weight_dtype
        loading_device = "cpu" if self.is_swapping_blocks else accelerator.device

        attn_mode = "torch"
        if args.xformers:
            attn_mode = "xformers"
        if args.attn_mode is not None:
            attn_mode = args.attn_mode

        logger.info(f"Loading Krea 2 DiT with attn_mode: {attn_mode}, split_attn: {args.split_attn}, fp8_scaled: {args.fp8_scaled}")
        model = krea2_utils.load_krea2_dit(
            args.pretrained_model_name_or_path,
            device=accelerator.device,
            dtype=loading_dtype if loading_dtype is not None else torch.bfloat16,
            fp8_scaled=args.fp8_scaled,
            loading_device=loading_device,
            attn_mode=attn_mode,
            split_attn=args.split_attn,
        )

        if self.is_swapping_blocks:
            logger.info(f"enable block swap: blocks_to_swap={args.blocks_to_swap}")
            model.enable_block_swap(args.blocks_to_swap, accelerator.device, supports_backward=True)

        return model, text_encoders

    def get_tokenize_strategy(self, args):
        return strategy_krea2.Krea2TokenizeStrategy(args.tokenizer_cache_dir)

    def get_tokenizers(self, tokenize_strategy: strategy_krea2.Krea2TokenizeStrategy):
        return []  # K2's conditioner owns its own tokenizer/processor internally

    def get_latents_caching_strategy(self, args):
        return strategy_krea2.Krea2LatentsCachingStrategy(args.cache_latents_to_disk, args.vae_batch_size, False)

    def get_text_encoding_strategy(self, args):
        return strategy_krea2.Krea2TextEncodingStrategy()

    def post_process_network(self, args, accelerator, network, text_encoders, unet):
        pass

    def get_models_for_text_encoding(self, args, accelerator, text_encoders):
        if args.cache_text_encoder_outputs:
            return None
        return text_encoders

    def get_text_encoders_train_flags(self, args, text_encoders):
        return [False]  # K2 does not train the Qwen3-VL text encoder

    def get_text_encoder_outputs_caching_strategy(self, args):
        if args.cache_text_encoder_outputs:
            return strategy_krea2.Krea2TextEncoderOutputsCachingStrategy(
                args.cache_text_encoder_outputs_to_disk, args.text_encoder_batch_size, args.skip_cache_check, False
            )
        return None

    def cache_text_encoder_outputs_if_needed(
        self, args, accelerator: Accelerator, unet, vae, text_encoders, dataset: DatasetGroup, weight_dtype
    ):
        te_device = accelerator.device
        if args.cache_text_encoder_outputs:
            if not args.lowram:
                logger.info("move vae to cpu to save memory")
                org_vae_device = vae.device
                vae.to("cpu")
                clean_memory_on_device(accelerator.device)

            logger.info(f"move text encoder to {te_device} to encode and cache text encoder outputs")
            text_encoders[0].to(te_device)

            dataset.new_cache_text_encoder_outputs(text_encoders, accelerator)

            if args.sample_prompts is not None:
                logger.info(f"cache Text Encoder outputs for sample prompt: {args.sample_prompts}")

                tokenize_strategy = strategy_base.TokenizeStrategy.get_strategy()
                text_encoding_strategy = strategy_base.TextEncodingStrategy.get_strategy()

                prompts = sampling.load_prompts(args.sample_prompts)
                sample_prompts_te_outputs = {}  # prompt -> (hiddens, mask) gathered (varlen), on cpu
                with torch.no_grad():
                    for prompt_dict in prompts:
                        for p in [prompt_dict.get("prompt", ""), prompt_dict.get("negative_prompt", "")]:
                            if p not in sample_prompts_te_outputs:
                                logger.info(f"cache Text Encoder outputs for prompt: {p}")
                                tokens = tokenize_strategy.tokenize(p)
                                hiddens, mask = text_encoding_strategy.encode_tokens(tokenize_strategy, text_encoders, tokens)
                                # gather to valid-only (drop padding), matching the on-disk cache format
                                valid = mask[0].bool()
                                sample_prompts_te_outputs[p] = (hiddens[0][valid].to("cpu"), mask[0][valid].to("cpu"))
                self.sample_prompts_te_outputs = sample_prompts_te_outputs

            accelerator.wait_for_everyone()

            logger.info("move text encoder to meta device to save memory")
            text_encoders[0] = text_encoders[0].to("meta") if text_encoders[0] is not None else None
            clean_memory_on_device(accelerator.device)

            if not args.lowram:
                logger.info("move vae back to original device")
                vae.to(org_vae_device)
        else:
            if text_encoders[0] is not None:
                text_encoders[0].to(te_device)

    def sample_images(self, accelerator, args, epoch, global_step, device, ae, tokenizer, text_encoder, flux):
        text_encoders = self.get_models_for_text_encoding(args, accelerator, text_encoder)
        sample_images(accelerator, args, epoch, global_step, flux, ae, text_encoders, self.sample_prompts_te_outputs)

    def get_noise_scheduler(self, args: argparse.Namespace, device: torch.device) -> Any:
        noise_scheduler = sd3_train_utils.FlowMatchEulerDiscreteScheduler(num_train_timesteps=1000, shift=args.discrete_flow_shift)
        self.noise_scheduler_copy = copy.deepcopy(noise_scheduler)
        return noise_scheduler

    def encode_images_to_latents(self, args, vae: qwen_image_autoencoder_kl.AutoencoderKLQwenImage, images):
        return vae.encode_pixels_to_latents(images)

    def shift_scale_latents(self, args, latents):
        # K2 latents are already normalized by the Qwen-Image VAE caching ((raw - mean) / std).
        return latents

    def get_noise_pred_and_target(
        self,
        args,
        accelerator,
        noise_scheduler,
        latents,
        batch,
        text_encoder_conds,
        unet: krea2_models.SingleStreamDiT,
        network,
        weight_dtype,
        train_unet,
        is_train=True,
    ):
        device = accelerator.device
        patch = unet.config.patch
        bsz = latents.shape[0]

        noise = torch.randn_like(latents)
        noisy_model_input, _, sigmas = flux_train_utils.get_noisy_model_input_and_timesteps(
            args, noise_scheduler, latents, noise, device, weight_dtype
        )
        timesteps = (sigmas[:, 0, 0, 0] * 1000).to(torch.int64)

        # --- image tokens / pos / mask ---
        _, _, lat_h, lat_w = noisy_model_input.shape
        h_, w_ = lat_h // patch, lat_w // patch

        from einops import rearrange, repeat

        img_tokens = rearrange(noisy_model_input, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

        imgids = torch.zeros((h_, w_, 3), device=device)
        imgids[..., 1] = torch.arange(h_, device=device)[:, None]
        imgids[..., 2] = torch.arange(w_, device=device)[None, :]
        imgpos = repeat(imgids, "h w three -> b (h w) three", b=bsz, three=3)
        imgmask = torch.ones(bsz, h_ * w_, device=device, dtype=torch.bool)

        # --- text tokens / mask: cached as (valid_len, num_layers*dim) + (valid_len,) mask,
        # padded to the batch max by sd-scripts' generic collate (library.dataset.none_or_stack_elements).
        hiddens_flat, txtmask = text_encoder_conds  # (B, max_len, L*D), (B, max_len)
        num_layers, txtdim = len(krea2_text_encoder.TextEncoderConfig.select_layers), unet.config.txtdim
        context = hiddens_flat.reshape(bsz, hiddens_flat.shape[1], num_layers, txtdim).to(device=device, dtype=weight_dtype)
        txtmask = txtmask.to(device=device).bool()
        txtpos = torch.zeros(bsz, context.shape[1], 3, device=device)

        mask = torch.cat((imgmask, txtmask), dim=1)
        pos = torch.cat((imgpos, txtpos), dim=1)

        img_tokens = img_tokens.to(device=device, dtype=weight_dtype)
        t = (timesteps.float() / 1000.0).to(device=device)

        if args.gradient_checkpointing:
            img_tokens.requires_grad_(True)
            context.requires_grad_(True)

        with torch.set_grad_enabled(is_train), accelerator.autocast():
            model_pred = unet(img=img_tokens, context=context, t=t, pos=pos, mask=mask)  # (B, h*w, c*ph*pw)

        model_pred = rearrange(model_pred, "b (h w) (c ph pw) -> b c (h ph) (w pw)", ph=patch, pw=patch, h=h_, w=w_)

        target = noise - latents
        weighting = None
        return model_pred, target, timesteps, weighting

    def post_process_loss(self, loss, args, timesteps, noise_scheduler):
        return loss

    def get_sai_model_spec(self, args):
        return model_io.get_sai_model_spec_dataclass(None, args, False, True, False).to_metadata_dict()

    def update_metadata(self, metadata, args):
        metadata["ss_timestep_sampling"] = args.timestep_sampling
        metadata["ss_discrete_flow_shift"] = args.discrete_flow_shift
        metadata["ss_model_prediction_type"] = "raw"  # K2 always predicts flow-matching velocity directly

    def is_text_encoder_not_needed_for_training(self, args):
        return args.cache_text_encoder_outputs and not self.is_train_text_encoder(args)

    def prepare_text_encoder_grad_ckpt_workaround(self, index, text_encoder):
        pass  # K2 does not support training the Qwen3-VL text encoder

    def cast_text_encoder(self, args):
        return False  # Qwen3-VL is loaded directly in bf16

    def cast_vae(self, args):
        return False  # VAE is loaded directly in bf16

    def cast_unet(self, args):
        return not args.fp8_scaled

    def prepare_text_encoder_fp8(self, index, text_encoder, te_weight_dtype, weight_dtype):
        pass  # fp8 text encoder for K2 is not supported currently

    def on_validation_step_end(self, args, accelerator, network, text_encoders, unet, batch, weight_dtype):
        if self.is_swapping_blocks:
            accelerator.unwrap_model(unet).prepare_block_swap_before_forward()

    def prepare_unet_with_accelerator(self, args: argparse.Namespace, accelerator: Accelerator, unet: nn.Module) -> nn.Module:
        if not self.is_swapping_blocks:
            return super().prepare_unet_with_accelerator(args, accelerator, unet)

        model: krea2_models.SingleStreamDiT = unet
        model = accelerator.prepare(model, device_placement=[not self.is_swapping_blocks])
        accelerator.unwrap_model(model).move_to_device_except_swap_blocks(accelerator.device)
        accelerator.unwrap_model(model).prepare_block_swap_before_forward()

        return model


def setup_parser() -> argparse.ArgumentParser:
    parser = train_network.setup_parser()
    args_util.add_dit_training_arguments(parser)

    parser.add_argument(
        "--text_encoder",
        type=str,
        default=None,
        help="Qwen3-VL-4B-Instruct text encoder path (safetensors, official or ComfyUI key layout). Required unless "
        "--cache_text_encoder_outputs is used with a fully pre-cached dataset and no sample prompts."
        " / Qwen3-VL-4B-Instructテキストエンコーダのパス（safetensors、公式またはComfyUI形式）。",
    )
    parser.add_argument(
        "--timestep_sampling",
        choices=["sigma", "uniform", "sigmoid", "shift", "flux_shift"],
        default="shift",
        help="Method to sample timesteps. K2 docs recommend 'shift' with --discrete_flow_shift ~2.5 for 1024x1024."
        " / タイムステップのサンプリング方法。K2では--discrete_flow_shift 2.5前後の'shift'が1024x1024の目安として推奨されています。",
    )
    parser.add_argument("--sigmoid_scale", type=float, default=1.0, help="Scale factor for sigmoid timestep sampling.")
    parser.add_argument(
        "--discrete_flow_shift",
        type=float,
        default=2.5,
        help="Discrete flow shift for the flow-matching schedule. K2 docs: ~2.5 matches K2's inference time-shift at "
        "1024x1024 (resolution-aware range ~1.6 at 256x256 to ~3.2 at 1280x1280)."
        " / フローマッチングのシフト値。K2では1024x1024で約2.5が目安。",
    )
    parser.add_argument(
        "--model_prediction_type",
        choices=["raw"],
        default="raw",
        help="K2 always predicts flow-matching velocity directly (raw). Kept for CLI compatibility with other architectures.",
    )
    parser.add_argument("--fp8_scaled", action="store_true", help="use dynamic scaled fp8 for the DiT (requires no --fp8_base).")
    parser.add_argument(
        "--attn_mode",
        choices=["torch", "xformers", "flash", "sageattn", "sdpa"],
        default=None,
        help="Attention implementation to use. Default is None (torch).",
    )
    parser.add_argument("--split_attn", action="store_true", help="split attention computation to reduce memory usage")

    return parser


if __name__ == "__main__":
    parser = setup_parser()

    args = parser.parse_args()
    args_util.verify_command_line_training_args(args)
    args = args_util.read_config_from_file(args, parser)

    if args.attn_mode == "sdpa":
        args.attn_mode = "torch"

    trainer = Krea2NetworkTrainer()
    trainer.train(args)
