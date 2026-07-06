# Krea 2 (K2) sampling helpers: position/mask preparation and the resolution-aware
# flow-matching timestep schedule. Ported from kohya-ss/musubi-tuner
# (src/musubi_tuner/krea2/krea2_sampling.py), trimmed to the pieces shared between
# training-time image token prep and the sample-image-during-training Euler sampler.
#
# Unlike musubi (video-first, latents are (B,C,F,H,W)), this port keeps latents as plain
# (B,C,H,W) throughout -- sd-scripts' image datasets/VAE caching never carry a frame axis.

import math

import torch
from einops import rearrange, repeat


def roundup(value: int, multiple: int, name: str) -> int:
    """Round `value` up to the nearest multiple, logging when padding is applied."""
    aligned = ((value + multiple - 1) // multiple) * multiple
    if aligned != value:
        print(f"[krea2 sample] {name}={value} is not a multiple of {multiple}; padding to {aligned}")
    return aligned


def gather_valid_text(txt: torch.Tensor, mask: torch.Tensor):
    """Drop masked (invalid) text tokens so the valid ones form a contiguous prefix, then
    right-pad to the batch maximum.

    The Qwen3-VL conditioner pads the prompt to max_length and appends the template suffix,
    so its mask is [valid prompt, pad, valid suffix] -- valid tokens are NOT a prefix. K2's
    attention assumes valid == leading prefix, so the interior padding must be removed first.
    Dropping it is lossless: text tokens get zero RoPE position and padding is masked out, so
    only the set/order of valid tokens matters.

    txt: (B, seq, L, D), mask: (B, seq) bool -> (B, max_valid, L, D), (B, max_valid) bool.
    """
    valid = [txt[i][mask[i]] for i in range(txt.shape[0])]  # list of (n_i, L, D)
    max_len = max(v.shape[0] for v in valid)
    out = txt.new_zeros(txt.shape[0], max_len, txt.shape[2], txt.shape[3])
    newmask = torch.zeros(txt.shape[0], max_len, device=txt.device, dtype=torch.bool)
    for i, v in enumerate(valid):
        out[i, : v.shape[0]] = v
        newmask[i, : v.shape[0]] = True
    return out, newmask


def prepare(img: torch.Tensor, txtlen: int, patch: int, txtmask: torch.Tensor):
    """Patchify the latent and build the combined image+text position / mask tensors.

    Image tokens lead the sequence so each sample's valid tokens form a contiguous prefix
    ([img (all valid), text (valid prefix + padding)]), matching what SingleStreamDiT.forward
    expects. Returns (img_tokens, pos, mask). ``img`` is (B, C, H, W).
    """
    b, _, h, w = img.shape
    h_, w_ = h // patch, w // patch
    imgids = torch.zeros((h_, w_, 3), device=img.device)
    imgids[..., 1] = torch.arange(h_, device=img.device)[:, None]
    imgids[..., 2] = torch.arange(w_, device=img.device)[None, :]
    imgpos = repeat(imgids, "h w three -> b (h w) three", b=b, three=3)
    imgmask = torch.ones(b, h_ * w_, device=img.device, dtype=torch.bool)
    img_tokens = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=patch, pw=patch)

    txtpos = torch.zeros(b, txtlen, 3, device=img.device)
    mask = torch.cat((imgmask, txtmask), dim=1)
    pos = torch.cat((imgpos, txtpos), dim=1)
    return img_tokens, pos, mask


def krea2_timesteps(seq_len: int, steps: int, x1: float, x2: float, y1: float = 0.5, y2: float = 1.15, sigma: float = 1.0, mu=None):
    """Resolution-aware flow-matching timestep schedule (t: 1 -> 0).

    `mu` is interpolated linearly in image-sequence length between (x1,y1) and (x2,y2), then
    used to time-shift a uniform 1->0 grid. Pass an explicit `mu` to pin a constant shift
    regardless of resolution (used by the distilled Turbo checkpoint, trained at a fixed
    mu=1.15).
    """
    ts = torch.linspace(1, 0, steps + 1)
    if mu is None:
        slope = (y2 - y1) / (x2 - x1)
        mu = slope * seq_len + (y1 - slope * x1)
    ts = math.exp(mu) / (math.exp(mu) + (1.0 / ts - 1.0) ** sigma)
    return ts.tolist()
