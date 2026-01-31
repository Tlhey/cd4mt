#!/usr/bin/env python

# python infer.py --ckpt checkpoints/test_run/step_500.pt --config configs/test_cfg.yaml
"""
infer.py - Generate audio samples from trained diffusion model
"""

import sys
import argparse
import yaml
import numpy as np
import torch
import soundfile as sf
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from music2latent import EncoderDecoder
from score.diffusion import Diffusion, LogNormalDistribution, KarrasSchedule, KarrasSampler, DiffusionSampler
from score.modules import UNet1d


def load_config(path: str) -> dict:
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
        dynamic_threshold=c["dynamic_threshold"],
    )


def decode_latents(ae: EncoderDecoder, latents: torch.Tensor, num_stems: int, latent_dim: int) -> torch.Tensor:
    """Decode [B, S*C, L] latents to [B, S, T] waveforms."""
    B, SC, L = latents.shape
    S = num_stems
    C = latent_dim

    # [B, S*C, L] -> [B, S, C, L]
    latents = latents.view(B, S, C, L)

    wav_list = []
    for s in range(S):
        stem_latent = latents[:, s].cpu().numpy()  # [B, C, L]
        stem_wav = ae.decode(stem_latent)  # [B, T]
        if isinstance(stem_wav, np.ndarray):
            stem_wav = torch.from_numpy(stem_wav)
        wav_list.append(stem_wav.float())

    # [B, S, T]
    return torch.stack(wav_list, dim=1)


@torch.no_grad()
def sample_from_diffusion(diffusion: Diffusion, shape: tuple, device: torch.device, num_steps: int = 50,
                          sigma_min: float = 0.002, sigma_max: float = 5.0) -> torch.Tensor:
    """Sample from diffusion model using Karras sampler.

    Note: sigma_max should match training distribution. For LogNormal(mean=-1.2, std=1.2),
    the 95th percentile is ~2.1 and 99th percentile is ~4.6, so sigma_max=5.0 is reasonable.
    Using sigma_max=10.0 may cause issues if the model hasn't seen enough high-noise samples.
    """
    sampler = DiffusionSampler(
        diffusion=diffusion,
        sampler=KarrasSampler(),
        sigma_schedule=KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0),
        num_steps=num_steps,
    )
    noise = torch.randn(shape, device=device)
    generated = sampler(noise)

    # Print stats for debugging
    print(f"Generated latent stats: mean={generated.mean():.4f}, std={generated.std():.4f}, "
          f"min={generated.min():.4f}, max={generated.max():.4f}")

    return generated


def main():
    parser = argparse.ArgumentParser(description="Generate audio from diffusion model")
    parser.add_argument("--ckpt", type=str, default="/data1/yuchen/cd4mt/checkpoints/test_run/step_5000.pt")
    parser.add_argument("--config", type=str, default="configs/test_cfg.yaml")
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--sigma_max", type=float, default=5.0, help="Max sigma for sampling (default: 5.0)")
    parser.add_argument("--output_dir", type=str, default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load config
    cfg = load_config(args.config)
    num_stems = cfg["model"]["cae"]["num_stems"]
    latent_dim = cfg["model"]["cae"]["latent_dim"]
    stem_names = ["bass", "drums", "guitar", "piano"]
    sr = cfg["data"]["params"]["preprocessing"]["audio"]["sampling_rate"]

    # Build model
    in_channels = num_stems * latent_dim
    unet = build_unet(cfg, in_channels)
    diffusion = build_diffusion(cfg, unet).to(device)

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location=device)
    diffusion.load_state_dict(ckpt["model"])
    diffusion.eval()
    print(f"Loaded checkpoint: {args.ckpt} (step {ckpt['step']})")

    # Load CAE decoder
    ae = EncoderDecoder(device=device)

    # Output directory
    ckpt_name = Path(args.ckpt).stem  # e.g., "step_5000"
    run_name = Path(args.ckpt).parent.name  # e.g., "test_run"
    if args.output_dir:
        out_dir = Path(args.output_dir)
    else:
        out_dir = ROOT / run_name / ckpt_name
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {out_dir}")

    # Latent shape: [B, S*C, L]
    # L=128 is typical for 12s audio at 44100Hz with music2latent
    latent_length = 128
    latent_shape = (args.num_samples, in_channels, latent_length)

    print(f"Generating {args.num_samples} samples with {args.num_steps} diffusion steps (sigma_max={args.sigma_max})...")

    # Sample
    gen_latents = sample_from_diffusion(diffusion, latent_shape, device, num_steps=args.num_steps, sigma_max=args.sigma_max)
    print(f"Generated latents shape: {gen_latents.shape}")

    # Decode to waveforms
    print("Decoding to waveforms...")
    gen_wavs = decode_latents(ae, gen_latents, num_stems, latent_dim)  # [B, S, T]
    print(f"Generated waveforms shape: {gen_wavs.shape}")

    # Save each sample
    for i in range(args.num_samples):
        sample_dir = out_dir / str(i)
        sample_dir.mkdir(parents=True, exist_ok=True)

        # Save each stem
        for s, stem_name in enumerate(stem_names):
            stem_wav = gen_wavs[i, s].cpu().numpy()
            stem_path = sample_dir / f"{stem_name}.wav"
            sf.write(stem_path, stem_wav, sr)

        # Save mix
        mix_wav = gen_wavs[i].sum(dim=0).cpu().numpy()
        mix_path = sample_dir / "mix.wav"
        sf.write(mix_path, mix_wav, sr)

        print(f"Saved sample {i} to {sample_dir}")

    print("Done!")


if __name__ == "__main__":
    main()
