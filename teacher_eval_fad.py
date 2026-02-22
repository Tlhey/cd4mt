#!/usr/bin/env python
"""
teacher_eval_fad.py - Evaluate teacher diffusion model with lightweight FAD on val set.

Example:
  python teacher_eval_fad.py \
    --ckpt checkpoints/test_run/step_1500.pt \
    --config configs/test_cfg.yaml \
    --num_eval 8 --steps 40 --sigma_min 0.002 --sigma_max 5.0
"""

from __future__ import annotations

import argparse
from pathlib import Path
import yaml
import numpy as np
import torch
import sys
import time

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from music2latent import EncoderDecoder
from DataLoader.multitrack_datamodule import DataModuleFromConfig
from score.modules import UNet1d
from score.diffusion import Diffusion, LogNormalDistribution, KarrasSchedule, KarrasSampler, DiffusionSampler
from metrics.fad import compute_fad


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


def load_checkpoint_state(path: str):
    ckpt = torch.load(path, map_location="cpu", weights_only=True)
    ema_state = None
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        ema_state = ckpt.get("ema")
        step = ckpt.get("step") or ckpt.get("global_step") or ckpt.get("epoch") or 0
    else:
        state, step = ckpt, 0
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    if isinstance(ema_state, dict):
        ema_state = {k.replace("module.", ""): v for k, v in ema_state.items()}
    return state, int(step), ema_state


def encode_stems(ae: EncoderDecoder, wav_stems: torch.Tensor) -> torch.Tensor:
    bsz, num_stems, _ = wav_stems.shape
    latents_list = []
    for stem_id in range(num_stems):
        stem_audio = wav_stems[:, stem_id].cpu().numpy()
        latent = ae.encode(stem_audio)
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent)
        latents_list.append(latent.float())
    stacked = torch.stack(latents_list, dim=1)
    bsz, num_stems, latent_dim, latent_len = stacked.shape
    return stacked.view(bsz, num_stems * latent_dim, latent_len)


@torch.no_grad()
def decode_latents_to_mix(ae: EncoderDecoder, latents: torch.Tensor, num_stems: int, latent_dim: int) -> list[torch.Tensor]:
    bsz, _, latent_len = latents.shape
    latents = latents.view(bsz, num_stems, latent_dim, latent_len)
    mix = None
    for stem_id in range(num_stems):
        stem_latent = latents[:, stem_id].cpu().numpy()
        stem_wav = ae.decode(stem_latent)
        if isinstance(stem_wav, np.ndarray):
            stem_wav = torch.from_numpy(stem_wav)
        stem_wav = stem_wav.float()
        mix = stem_wav if mix is None else (mix + stem_wav)
    assert mix is not None
    return [mix[i].detach().cpu() for i in range(mix.shape[0])]


@torch.no_grad()
def sample_from_diffusion(
    diffusion: Diffusion,
    shape: tuple[int, int, int],
    device: torch.device,
    steps: int,
    sigma_min: float,
    sigma_max: float,
) -> torch.Tensor:
    sampler = DiffusionSampler(
        diffusion=diffusion,
        sampler=KarrasSampler(),
        sigma_schedule=KarrasSchedule(sigma_min=sigma_min, sigma_max=sigma_max, rho=7.0),
        num_steps=steps,
    )
    noise = torch.randn(shape, device=device)
    return sampler(noise)


def collect_references(val_loader, num_eval: int) -> list[torch.Tensor]:
    refs: list[torch.Tensor] = []
    for batch in val_loader:
        if "waveform_stems" in batch:
            wav_stems = batch["waveform_stems"]
            if not isinstance(wav_stems, torch.Tensor):
                wav_stems = torch.tensor(wav_stems)
            mix = wav_stems.sum(dim=1)
            refs.extend([mix[i].detach().cpu().float() for i in range(mix.shape[0])])
        else:
            wav = batch["waveform"]
            if not isinstance(wav, torch.Tensor):
                wav = torch.tensor(wav)
            refs.extend([wav[i].detach().cpu().float() for i in range(wav.shape[0])])
        if len(refs) >= num_eval:
            break
    return refs[:num_eval]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, required=True, help="Teacher checkpoint path")
    parser.add_argument("--config", type=str, default="configs/test_cfg.yaml")
    parser.add_argument("--num_eval", type=int, default=8, help="Number of generated/reference samples")
    parser.add_argument("--steps", type=int, default=40, help="Sampling steps for teacher diffusion")
    parser.add_argument("--sigma_min", type=float, default=0.002)
    parser.add_argument("--sigma_max", type=float, default=5.0)
    parser.add_argument("--no_ema", action="store_true", help="Do not apply EMA even if checkpoint includes it")
    parser.add_argument("--swanlab_project", type=str, default="", help="SwanLab project name (empty disables logging)")
    parser.add_argument("--swanlab_run_name", type=str, default="", help="Optional SwanLab run name")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Optional SwanLab logging
    swanlab = None
    swan_project = args.swanlab_project or cfg.get("train", {}).get("swanlab", {}).get("project", "")
    swan_run_name = args.swanlab_run_name or cfg.get("train", {}).get("swanlab", {}).get("run_name", "")
    if swan_project:
        try:
            import swanlab as _swanlab
            run_cfg = {
                "ckpt": args.ckpt,
                "config": args.config,
                "num_eval": args.num_eval,
                "steps": args.steps,
                "sigma_min": args.sigma_min,
                "sigma_max": args.sigma_max,
                "ema_applied": (not args.no_ema),
            }
            if swan_run_name:
                _swanlab.init(project=swan_project, run_name=swan_run_name, config=run_cfg)
            else:
                _swanlab.init(project=swan_project, config=run_cfg)
            swanlab = _swanlab
            print(f"SwanLab enabled: project={swan_project}")
        except Exception as e:
            print(f"SwanLab init failed, continue without logging: {e}")
            swanlab = None

    dm = DataModuleFromConfig(**cfg["data"]["params"])
    dm.prepare_data()
    dm.setup(stage="validate")
    val_loader = dm.val_dataloader()

    ae = EncoderDecoder(device=device)

    num_stems = cfg["model"]["cae"]["num_stems"]
    latent_dim = cfg["model"]["cae"]["latent_dim"]
    in_channels = num_stems * latent_dim
    sr = cfg["data"]["params"]["preprocessing"]["audio"]["sampling_rate"]

    unet = build_unet(cfg, in_channels)
    diffusion = build_diffusion(cfg, unet).to(device)
    state, step, ema_state = load_checkpoint_state(args.ckpt)
    missing, unexpected = diffusion.load_state_dict(state, strict=False)
    print(f"Loaded ckpt step={step} | missing={len(missing)} unexpected={len(unexpected)}")

    if (not args.no_ema) and isinstance(ema_state, dict):
        with torch.no_grad():
            named_params = dict(diffusion.named_parameters())
            used = 0
            for k, v in ema_state.items():
                if k in named_params:
                    named_params[k].data.copy_(v.to(named_params[k].device, dtype=named_params[k].dtype))
                    used += 1
        print(f"Applied EMA params: {used}")
    else:
        print("EMA not applied.")

    diffusion.eval()

    first_batch = next(iter(val_loader))
    if "waveform_stems" in first_batch:
        ex_latent = encode_stems(ae, first_batch["waveform_stems"][:1])
        latent_len = int(ex_latent.shape[-1])
    else:
        latent_len = 128
    print(f"Latent length: {latent_len}")

    refs = collect_references(val_loader, args.num_eval)
    if len(refs) == 0:
        raise RuntimeError("No validation references found for FAD computation.")
    print(f"Collected refs: {len(refs)}")

    shape = (len(refs), in_channels, latent_len)
    t0 = time.time()
    gen_latents = sample_from_diffusion(
        diffusion=diffusion,
        shape=shape,
        device=device,
        steps=args.steps,
        sigma_min=args.sigma_min,
        sigma_max=args.sigma_max,
    )
    gens = decode_latents_to_mix(ae, gen_latents, num_stems=num_stems, latent_dim=latent_dim)
    fad = compute_fad(refs, gens, sr)
    print(f"Teacher FAD: {fad:.6f}")
    elapsed = time.time() - t0
    print(f"Eval elapsed: {elapsed:.2f}s")

    if swanlab:
        try:
            swanlab.log(
                {
                    "teacher_fad": float(fad),
                    "step": int(step),
                    "eval_num": int(len(refs)),
                    "sample_steps": int(args.steps),
                    "sigma_min": float(args.sigma_min),
                    "sigma_max": float(args.sigma_max),
                    "lat_len": int(latent_len),
                    "eval_elapsed_sec": float(elapsed),
                }
            )
        except Exception as e:
            print(f"SwanLab log failed: {e}")


if __name__ == "__main__":
    main()
