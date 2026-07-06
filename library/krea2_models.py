# Krea 2 (K2) single-stream MMDiT.
#
# Original work: https://github.com/krea-ai/krea-2
# Ported from kohya-ss/musubi-tuner (src/musubi_tuner/krea2/krea2_mmdit.py) to sd-scripts
# conventions: attention -> library.attention (same API as musubi's modules.attention),
# block swap -> library.custom_offloading_utils.ModelOffloader (musubi's create_offloader /
# BlockSwapConfig wrapper does not exist here; ModelOffloader is used directly, matching the
# pattern already used by library/hunyuan_image_models.py).
#
# Krea 2 is a text-to-image single-stream MMDiT: image and (fused) text tokens are
# concatenated and processed by a stack of SingleStreamBlocks (bidirectional attention,
# AdaLN-Zero-style modulation). Text conditioning comes from Qwen3-VL-4B hidden states,
# fused by a small TextFusionTransformer before being concatenated with the image tokens.
# The VAE is the Qwen-Image VAE (library.qwen_image_autoencoder_kl.AutoencoderKLQwenImage),
# already present in this codebase and reused as-is.

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from einops import rearrange
from torch import Tensor

from library import custom_offloading_utils
from library.attention import AttentionParams, attention as common_attention

from library.utils import setup_logging

setup_logging()
import logging

logger = logging.getLogger(__name__)


def rope(pos: Tensor, dim: int, theta: float = 1e4, ntk: float = 1.0) -> Tensor:
    scale = torch.arange(0, dim, 2, dtype=torch.float64, device=pos.device) / dim
    omega = 1.0 / ((theta * ntk) ** scale)
    out = torch.einsum("...n,d->...nd", pos, omega)
    out = torch.stack([torch.cos(out), -torch.sin(out), torch.sin(out), torch.cos(out)], dim=-1)
    out = rearrange(out, "b n d (i j) -> b n d i j", i=2, j=2)
    return out.float()


def ropeapply(xq: Tensor, xk: Tensor, freqs: Tensor) -> tuple[Tensor, Tensor]:
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 1, 2)
    xk_ = xk.float().reshape(*xk.shape[:-1], -1, 1, 2)
    freqs = freqs[:, None, :, :, :]
    xq_ = freqs[..., 0] * xq_[..., 0] + freqs[..., 1] * xq_[..., 1]
    xk_ = freqs[..., 0] * xk_[..., 0] + freqs[..., 1] * xk_[..., 1]
    return xq_.reshape(*xq.shape).to(xq.dtype), xk_.reshape(*xk.shape).to(xk.dtype)


def temb(
    t: Tensor,
    dim: int,
    period: float = 1e4,
    tfactor: float = 1e3,
    device: torch.device = None,
    dtype: torch.dtype = None,
) -> Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(period) * torch.arange(half, dtype=torch.float32, device=device) / half)
    # t: (B,) -> args: (B, 1, half), so the embedding broadcasts as a per-sample vec.
    args = (t.float() * tfactor)[:, None, None] * freqs
    sin, cos = torch.sin(args), torch.cos(args)
    return torch.cat((cos, sin), dim=-1).to(dtype=dtype)


@dataclass
class SingleMMDiTConfig:
    features: int
    tdim: int
    txtdim: int
    heads: int
    multiplier: int
    layers: int
    patch: int
    channels: int
    bias: bool = False
    theta: float = 1e3
    kvheads: int | None = None
    txtlayers: int = 1
    txtheads: int = 20
    txtkvheads: int = 20


# The single config shipped with the OSS checkpoints (single_mmdit_large_wide).
single_mmdit_large_wide = SingleMMDiTConfig(
    features=6144,
    tdim=256,
    txtdim=2560,
    heads=48,
    kvheads=12,
    multiplier=4,
    layers=28,
    patch=2,
    channels=16,
    txtheads=20,
    txtkvheads=20,
    txtlayers=12,
)


class SimpleModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(2, dim))
        self.multiplier = 2

    # vec (b d)
    def forward(self, vec: Tensor):
        out = vec + rearrange(self.lin, "two d -> 1 two d")
        scale, shift = out.chunk(self.multiplier, dim=1)
        return scale, shift


class DoubleSharedModulation(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.lin = nn.Parameter(torch.zeros(6 * dim))

    # vec (b (6 d))
    def forward(self, vec: Tensor):
        out = vec + self.lin
        prescale, preshift, pregate, postscale, postshift, postgate = out.chunk(6, dim=-1)
        return prescale, preshift, pregate, postscale, postshift, postgate


class PositionalEncoding(nn.Module):
    def __init__(self, dim, axdims: list[int], theta: float = 1e2, ntk: float = 1.0):
        super().__init__()
        self.axdims = axdims  # how to split the head dimension across the position axes
        self.theta = theta
        self.ntk = ntk

    def forward(self, pos: Tensor) -> Tensor:
        return torch.cat(
            [rope(pos[..., i], d, self.theta, self.ntk) for i, d in enumerate(self.axdims)],
            dim=-3,
        )


class RMSNorm(nn.Module):
    def __init__(self, features: int, eps: float = 1e-05, device: torch.device = None):
        super().__init__()
        self.features = features
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(features, device=device, dtype=torch.float32))

    def forward(self, x: Tensor) -> Tensor:
        t, dtype = x.float(), x.dtype
        t = F.rms_norm(t, (self.features,), eps=self.eps, weight=(self.scale.float() + 1.0))
        return t.to(dtype)


class QKNorm(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.qnorm = RMSNorm(dim)
        self.knorm = RMSNorm(dim)

    def forward(self, q: Tensor, k: Tensor, v: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        return self.qnorm(q), self.knorm(k), v


class SwiGLU(nn.Module):
    def __init__(self, features: int, multiplier: int, bias: bool = False, multiple: int = 128):
        super().__init__()

        mlpdim = int(2 * features / 3) * multiplier
        mlpdim = multiple * ((mlpdim + multiple - 1) // multiple)

        self.gate = nn.Linear(features, mlpdim, bias=bias)
        self.up = nn.Linear(features, mlpdim, bias=bias)
        self.down = nn.Linear(mlpdim, features, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int, kvheads: int = None, bias: bool = False):
        super().__init__()
        self.heads = heads
        self.kvheads = kvheads if kvheads is not None else heads
        self.headdim = dim // self.heads

        self.wq = nn.Linear(dim, self.headdim * self.heads, bias=bias)
        self.wk = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.wv = nn.Linear(dim, self.headdim * self.kvheads, bias=bias)
        self.gate = nn.Linear(dim, dim, bias=bias)
        self.qknorm = QKNorm(self.headdim)
        self.wo = nn.Linear(dim, dim, bias=bias)

    def forward(self, qkv: Tensor, freqs: Tensor | None = None, attn_params: AttentionParams | None = None) -> Tensor:
        q, k, v, gate = self.wq(qkv), self.wk(qkv), self.wv(qkv), self.gate(qkv)

        # QKNorm + RoPE run in [B, H, L, D] (K2-native layout) to preserve the reference numerics.
        q, k, v = (
            rearrange(q, "B L (H D) -> B H L D", H=self.heads),
            rearrange(k, "B L (H D) -> B H L D", H=self.kvheads),
            rearrange(v, "B L (H D) -> B H L D", H=self.kvheads),
        )

        q, k, v = self.qknorm(q, k, v)
        if freqs is not None:
            q, k = ropeapply(q, k, freqs)

        # library.attention's shared attention() does NOT handle GQA itself (its "torch" path calls
        # SDPA without enable_gqa, and none of the other backends expand kv either) -- k/v must
        # already have the same head count as q. K2 uses GQA (kvheads < heads) for wk/wv, so expand
        # them here before calling into the shared attention function.
        if self.kvheads != self.heads:
            rep = self.heads // self.kvheads
            k = k.repeat_interleave(rep, dim=1)
            v = v.repeat_interleave(rep, dim=1)

        # library.attention expects [B, L, H, D] and returns [B, L, H*D].
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        x = common_attention([q, k, v], attn_params=attn_params)
        out = self.wo(x * F.sigmoid(gate))

        return out


class LastLayer(nn.Module):
    def __init__(self, features: int, patch: int, channels: int):
        super().__init__()
        self.norm = RMSNorm(features)
        self.linear = nn.Linear(features, patch * patch * channels, bias=True)
        self.modulation = SimpleModulation(features)

    def forward(self, x: Tensor, tvec: Tensor) -> Tensor:
        scale, shift = self.modulation(tvec)
        x = (1 + scale) * self.norm(x) + shift
        x = self.linear(x)
        return x


class TextFusionBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, attn_params: AttentionParams | None = None) -> Tensor:
        x = x + self.attn(self.prenorm(x), attn_params=attn_params)
        x = x + self.mlp(self.postnorm(x))
        return x


class TextFusionTransformer(nn.Module):
    # num_txt_layers is the number of selected encoder hidden-state layers fed in
    # (projected down to 1), NOT the transformer depth -- that's fixed at 2 + 2 blocks.
    def __init__(self, num_txt_layers: int, txt_dim: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.layerwise_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])
        self.projector = nn.Linear(num_txt_layers, 1, bias=False)
        self.refiner_blocks = nn.ModuleList([TextFusionBlock(txt_dim, heads, multiplier, bias, kvheads) for _ in range(2)])

    def forward(
        self,
        x: Tensor,
        attn_params_nomask: AttentionParams | None = None,
        attn_params: AttentionParams | None = None,
    ) -> Tensor:
        b, l, n, d = x.shape
        x = x.reshape(b * l, n, d)
        for block in self.layerwise_blocks:
            x = block(x.contiguous(), attn_params=attn_params_nomask)
        x = rearrange(x, "(b l) n d -> b l d n", b=b, l=l)
        x = self.projector(x)
        x = x.squeeze(-1)

        for block in self.refiner_blocks:
            x = block(x, attn_params=attn_params)

        return x


class SingleStreamBlock(nn.Module):
    def __init__(self, features: int, heads: int, multiplier: int, bias: bool = False, kvheads: int = None):
        super().__init__()
        self.mod = DoubleSharedModulation(features)
        self.prenorm = RMSNorm(features)
        self.postnorm = RMSNorm(features)
        self.attn = Attention(dim=features, heads=heads, bias=bias, kvheads=kvheads)
        self.mlp = SwiGLU(features, multiplier, bias)

    def forward(self, x: Tensor, vec: Tensor, freqs: Tensor, attn_params: AttentionParams | None = None) -> Tensor:
        prescale, preshift, pregate, postscale, postshift, postgate = self.mod(vec)
        x = x + pregate * self.attn((1 + prescale) * self.prenorm(x) + preshift, freqs, attn_params)
        x = x + postgate * self.mlp((1 + postscale) * self.postnorm(x) + postshift)
        return x


class SingleStreamDiT(nn.Module):
    def __init__(self, config: SingleMMDiTConfig, attn_mode: str = "torch", split_attn: bool = False):
        super().__init__()
        self.config = config
        # Backend for the shared attention ("torch"=SDPA, "flash", "sageattn", "xformers").
        self.attn_mode = attn_mode
        self.split_attn = split_attn

        headdim = config.features // config.heads
        axes = [
            headdim - 12 * (headdim // 16),
            6 * (headdim // 16),
            6 * (headdim // 16),
        ]
        assert sum(axes) == headdim, f"sum(axes) = {sum(axes)}, headdim = {headdim}"
        assert all(a % 2 == 0 for a in axes), f"axes = {axes}"

        self.posemb = PositionalEncoding(config.features, axes, theta=config.theta, ntk=1.0)
        self.first = nn.Linear(config.channels * config.patch**2, config.features, bias=True)

        self.blocks = nn.ModuleList(
            [
                SingleStreamBlock(config.features, config.heads, config.multiplier, config.bias, config.kvheads)
                for _ in range(config.layers)
            ]
        )
        self.tmlp = nn.Sequential(
            nn.Linear(config.tdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.txtfusion = TextFusionTransformer(
            config.txtlayers, config.txtdim, config.txtheads, config.multiplier, config.bias, config.txtkvheads
        )
        self.txtmlp = nn.Sequential(
            RMSNorm(config.txtdim),
            nn.Linear(config.txtdim, config.features),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.features, config.features),
        )
        self.last = LastLayer(config.features, config.patch, config.channels)

        self.tproj = nn.Sequential(nn.GELU(approximate="tanh"), nn.Linear(config.features, config.features * 6))

        # sd-scripts training hooks
        self.gradient_checkpointing = False
        self.blocks_to_swap = None
        self.offloader = None
        self.num_blocks = len(self.blocks)

    @property
    def device(self):
        return next(self.parameters()).device

    @property
    def dtype(self):
        return next(self.parameters()).dtype

    def enable_gradient_checkpointing(self, cpu_offload: bool = False):
        # cpu_offload is accepted for interface parity; not implemented for K2 yet.
        self.gradient_checkpointing = True
        logger.info("Krea 2: Gradient checkpointing enabled.")

    def disable_gradient_checkpointing(self):
        self.gradient_checkpointing = False

    # Block swap (CPU offloading of the main SingleStreamBlocks). Follows the same pattern as
    # library/hunyuan_image_models.py: a single ModelOffloader over self.blocks.
    def enable_block_swap(self, num_blocks: int, device: torch.device, supports_backward: bool = False):
        self.blocks_to_swap = num_blocks
        assert num_blocks <= self.num_blocks - 2, f"Cannot swap more than {self.num_blocks - 2} blocks. Requested {num_blocks}."
        self.offloader = custom_offloading_utils.ModelOffloader(self.blocks, num_blocks, device, supports_backward=supports_backward)
        logger.info(f"Krea 2: Block swap enabled. Swapping {num_blocks} of {self.num_blocks} blocks to device {device}.")

    def switch_block_swap_for_inference(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(True)
            self.prepare_block_swap_before_forward()

    def switch_block_swap_for_training(self):
        if self.blocks_to_swap:
            self.offloader.set_forward_only(False)
            self.prepare_block_swap_before_forward()

    def move_to_device_except_swap_blocks(self, device: torch.device):
        # assume model is on cpu. do not move blocks to device to reduce temporary memory usage
        if self.blocks_to_swap:
            save_blocks = self.blocks
            self.blocks = nn.ModuleList()
        self.to(device)
        if self.blocks_to_swap:
            self.blocks = save_blocks

    def prepare_block_swap_before_forward(self):
        if self.blocks_to_swap is None or self.blocks_to_swap == 0:
            return
        self.offloader.prepare_block_devices_before_forward(self.blocks)

    def forward(self, img: Tensor, context: Tensor, t: Tensor, pos: Tensor, mask: Tensor | None = None) -> Tensor:
        img = self.first(img)
        t = self.tmlp(temb(t, self.config.tdim, device=img.device, dtype=img.dtype))
        tvec = self.tproj(t)

        # `mask`/`pos` arrive in image-first order: [img (all valid), text (valid prefix + pad)].
        # The text-only key-padding mask is therefore the tail beyond the image tokens.
        imglen = img.shape[1]
        txtmask = mask[:, imglen:]  # (B, txt_len) bool

        # Text fusion is a self-attention over text tokens only (img_len=0). The per-layer
        # blocks see every token (no mask); the refiner masks padding via txtmask.
        txt_attn_params_nomask = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, 0, None)
        txt_attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, 0, txtmask)
        context = self.txtfusion(context, txt_attn_params_nomask, txt_attn_params)
        context = self.txtmlp(context)

        combined = torch.cat((img, context), dim=1)  # image first, then text

        # Pad the combined sequence to a multiple of 256 to keep compiled kernel shapes stable.
        fulllen = combined.shape[1]
        padlen = (-fulllen) % 256
        if padlen > 0:
            combined = F.pad(combined, (0, 0, 0, padlen))
            pos = F.pad(pos, (0, 0, 0, padlen))
            txtmask = F.pad(txtmask, (0, padlen), value=False)

        # Main blocks: bidirectional attention over [image (img_len, all valid) + text (padded)].
        attn_params = AttentionParams.create_attention_params_from_mask(self.attn_mode, self.split_attn, imglen, txtmask)

        freqs = self.posemb(pos)

        for index, block in enumerate(self.blocks):
            if self.blocks_to_swap:
                self.offloader.wait_for_block(index)

            if self.gradient_checkpointing and self.training:
                combined = torch.utils.checkpoint.checkpoint(block, combined, tvec, freqs, attn_params, use_reentrant=False)
            else:
                combined = block(combined, tvec, freqs, attn_params)

            if self.blocks_to_swap:
                self.offloader.submit_move_blocks(self.blocks, index)

        final = self.last(combined, t)
        output = final[:, :imglen, :]  # image tokens are the leading slice now

        return output
