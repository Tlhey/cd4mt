#!/usr/bin/env python
"""
ct_infer.py - Sample with a CT-trained student (and optional EMA weights)

Example:
  python ct_infer.py --ckpt checkpoints/ct_run/ct_step_10000.pt \
                     --config configs/test_ct_cfg.yaml --num_samples 4
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import soundfile as sf
import sys

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from music2latent import EncoderDecoder
from score.modules import UNet1d
from score.diffusion import Diffusion, LogNormalDistribution, KarrasSchedule, KarrasSampler, DiffusionSampler


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_unet(cfg: dict, in_channels: int) -> UNet1d:
    c = cfg["model"]["unet_1d"]
    return UNet1d(
        in_channels=in_channels,
        channels=c["base_channels"],
        patch_blocks=c["patch_blocks"],
        patch_factor=c["patch_factor"],
        kernel_sizes_init=c["kernel_sizes_init"],
        multipliers=c["multipliers"],
        factors=c["factors"],
        num_blocks=c["num_blocks"],
        attentions=[bool(a) for a in c["attentions"]],
        attention_heads=c["attention_heads"],
        attention_features=c["attention_features"],
        attention_multiplier=c["attention_multiplier"],
        resnet_groups=c["resnet_groups"],
        kernel_multiplier_downsample=c["kernel_multiplier_downsample"],
        use_nearest_upsample=c["use_nearest_upsample"],
        use_skip_scale=c["use_skip_scale"],
        use_attention_bottleneck=c["use_attention_bottleneck"],
        use_context_time=c["use_context_time"],
        time_emb_type=c.get("time_emb_type", "LearnedPositional"),
    )


def build_diffusion(cfg: dict, net: UNet1d) -> Diffusion:
    c = cfg["model"]["diffusion"]
    sigma_dist = LogNormalDistribution(
        mean=c["sigma_distribution"]["mean"],
        std=c["sigma_distribution"]["std"],
    )
    return Diffusion(
        net=net,
        sigma_distribution=sigma_dist,
        sigma_data=c["sigma_data"],
        dynamic_threshold=c.get("dynamic_threshold", 0.0),
    )


def load_ct_checkpoint(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    step = ckpt.get("step", 0)
    model = ckpt.get("model") or ckpt
    ema = ckpt.get("ema")
    return model, ema, int(step)


def decode_latents(ae: EncoderDecoder, latents: torch.Tensor, num_stems: int, latent_dim: int) -> torch.Tensor:
    B, SC, L = latents.shape
    S = num_stems
    C = latent_dim
    latents = latents.view(B, S, C, L)
    wav_list = []
    for s in range(S):
        stem_latent = latents[:, s].cpu().numpy()
        stem_wav = ae.decode(stem_latent)
        if isinstance(stem_wav, np.ndarray):
            stem_wav = torch.from_numpy(stem_wav)
        wav_list.append(stem_wav.float())
    return torch.stack(wav_list, dim=1)


@torch.no_grad()
def sample_from_student(diffusion: Diffusion, shape: tuple, device: torch.device, steps: int, sigma_min: float, sigma_max: float):
    schedule = KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0)
    sigmas = schedule(steps, device)
    sampler = KarrasSampler()
    noise = torch.randn(shape, device=device)
    return sampler(noise, lambda x, sigma: diffusion.denoise_fn(x, sigma=sigma), sigmas, steps)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/test_ct_cfg.yaml")
    parser.add_argument("--num_samples", type=int, default=4)
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=5.0)
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    num_stems = cfg["model"]["cae"]["num_stems"]
    latent_dim = cfg["model"]["cae"]["latent_dim"]
    in_channels = num_stems * latent_dim

    unet = build_unet(cfg, in_channels)
    student = build_diffusion(cfg, unet).to(device)

    model_state, ema_state, step = load_ct_checkpoint(args.ckpt)
    model_state = {k.replace("module.", ""): v for k, v in model_state.items()}
    missing, unexpected = student.load_state_dict(model_state, strict=False)
    print(f"Loaded CT ckpt (step {step}): missing={len(missing)}, unexpected={len(unexpected)}")

    # If EMA present, apply it
    if isinstance(ema_state, dict):
        with torch.no_grad():
            for k, p in student.named_parameters():
                if k in ema_state:
                    p.copy_(ema_state[k].to(p.device, dtype=p.dtype))
        print("Applied EMA weights to student for inference.")

    student.eval()

    # Music2latent decoder
    ae = EncoderDecoder(device=device)

    # Sampling
    latent_len = 128
    shape = (args.num_samples, in_channels, latent_len)
    latents = sample_from_student(student, shape, device, args.steps, args.sigma_min, args.sigma_max)

    # Decode
    wavs = decode_latents(ae, latents, num_stems, latent_dim)  # [B, S, T]
    sr = cfg["data"]["params"]["preprocessing"]["audio"]["sampling_rate"]

    # Save
    ckpt_name = Path(args.ckpt).stem
    run_name = Path(args.ckpt).parent.name
    out_dir = Path(args.output_dir) if args.output_dir else ROOT / run_name / (ckpt_name + "_ct")
    out_dir.mkdir(parents=True, exist_ok=True)

    stem_names = ["bass", "drums", "guitar", "piano"]
    for i in range(args.num_samples):
        d = out_dir / str(i)
        d.mkdir(parents=True, exist_ok=True)
        mix = 0
        for s, stem in enumerate(stem_names):
            w = wavs[i, s].cpu().numpy()
            sf.write(d / f"{stem}.wav", w, sr)
            mix = mix + w
        sf.write(d / "mix.wav", mix, sr)
    print(f"Saved to {out_dir}")


if __name__ == "__main__":
    main()

