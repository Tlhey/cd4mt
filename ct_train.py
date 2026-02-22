#!/usr/bin/env python
"""
ct_train.py - Consistency Training (and Distillation) for 1D latent audio

Usage:
  python ct_train.py --config configs/test_ct_cfg.yaml

This script:
  - Loads data via DataModuleFromConfig (same as train.py)
  - Builds a student UNet1d + Diffusion wrapper (denoiser interface)
  - Optionally loads a teacher Diffusion from a checkpoint
  - Maintains EMA of student as the target model for CT loss
  - Optimizes the CM-style consistency objective (ct.consistency_loss)

We intentionally keep the network interface identical to src/score/diffusion.Diffusion
so that inference can use the same sampler utilities.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import copy
import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from typing import Optional

ROOT = Path(__file__).parent.resolve()
import sys
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from DataLoader.multitrack_datamodule import DataModuleFromConfig
from music2latent import EncoderDecoder
from score.modules import UNet1d
from score.diffusion import Diffusion, LogNormalDistribution, KarrasSchedule, KarrasSampler
from metrics.fad import compute_fad

from ct import KarrasScheduleSpec, consistency_loss


def load_yaml(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _dtype_from_str(name: str | None):
    if not name or str(name).lower() in {"auto", "keep", "none"}:
        return None
    name = str(name).lower()
    if name in {"fp32", "float32"}:
        return torch.float32
    if name in {"fp16", "float16", "half"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    return None


def _tensor_dict_cpu_cast(d: dict[str, torch.Tensor], dtype_name: str | None, to_cpu: bool = True) -> dict:
    target_dtype = _dtype_from_str(dtype_name)
    out = {}
    for k, v in d.items():
        t = v.detach()
        if to_cpu:
            if target_dtype is not None and t.is_floating_point():
                t = t.to("cpu", dtype=target_dtype)
            else:
                t = t.to("cpu")
        else:
            if target_dtype is not None and t.is_floating_point():
                t = t.to(dtype=target_dtype)
        out[k] = t
    return out


def _safe_save(obj, path: Path) -> None:
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, tmp, _use_new_zipfile_serialization=False)
    os.replace(tmp, path)


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.9999):
        self.decay = float(decay)
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if k not in self.shadow:
                self.shadow[k] = p.detach().clone()
                continue
            self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    @torch.no_grad()
    def copy_to(self, model: nn.Module):
        for k, p in model.named_parameters():
            if k in self.shadow:
                p.data.copy_(self.shadow[k].to(p.device, dtype=p.dtype))


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


def encode_stems(ae: EncoderDecoder, wav_stems: torch.Tensor) -> torch.Tensor:
    B, S, T = wav_stems.shape
    latents_list = []
    for s in range(S):
        stem_audio = wav_stems[:, s].cpu().numpy()
        latent = ae.encode(stem_audio)
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent)
        latents_list.append(latent.float())
    stacked = torch.stack(latents_list, dim=1)
    B, S, C, L = stacked.shape
    return stacked.view(B, S * C, L)


@torch.no_grad()
def _decode_latents_to_mix(ae: EncoderDecoder, latents: torch.Tensor, num_stems: int, latent_dim: int) -> list[torch.Tensor]:
    import numpy as np
    B, SC, L = latents.shape
    S = num_stems
    C = latent_dim
    latents = latents.view(B, S, C, L)
    mix: torch.Tensor | None = None
    for s in range(S):
        stem_latent = latents[:, s].cpu().numpy()
        stem_wav = ae.decode(stem_latent)
        if isinstance(stem_wav, np.ndarray):
            stem_wav = torch.from_numpy(stem_wav)
        stem_wav = stem_wav.float()
        mix = stem_wav if mix is None else (mix + stem_wav)
    return [mix[i].detach().cpu() for i in range(mix.shape[0])]


def load_teacher(ckpt_path: str, teacher_cfg_path: str, device: torch.device) -> Diffusion:
    base_cfg = load_yaml(teacher_cfg_path)
    num_stems = base_cfg["model"]["cae"]["num_stems"]
    latent_dim = base_cfg["model"]["cae"]["latent_dim"]
    in_channels = num_stems * latent_dim
    teacher_unet = build_unet(base_cfg, in_channels)
    teacher_diffusion = build_diffusion(base_cfg, teacher_unet).to(device)

    # robust checkpoint load
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state = ckpt.get("model") if isinstance(ckpt, dict) else ckpt
    if isinstance(state, dict):
        state = {k.replace("module.", ""): v for k, v in state.items()}
    missing, unexpected = teacher_diffusion.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[Teacher] Non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")
    teacher_diffusion.eval()
    for p in teacher_diffusion.parameters():
        p.requires_grad_(False)
    return teacher_diffusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/test_ct_cfg.yaml")
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()

    cfg = load_yaml(args.config)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(cfg.get("seed", 42))

    # Data
    dm = DataModuleFromConfig(**cfg["data"]["params"])
    dm.prepare_data()
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()

    # CAE for encode
    ae = EncoderDecoder(device=device)

    # Student model
    num_stems = cfg["model"]["cae"]["num_stems"]
    latent_dim = cfg["model"]["cae"]["latent_dim"]
    in_channels = num_stems * latent_dim
    student_unet = build_unet(cfg, in_channels)
    student = build_diffusion(cfg, student_unet).to(device)

    # Optim
    train_cfg = cfg["train"]
    lr = float(train_cfg.get("lr", 1e-4))
    wd = float(train_cfg.get("weight_decay", 0.0))
    optimizer = optim.AdamW(student.parameters(), lr=lr, weight_decay=wd)
    grad_clip = float(train_cfg.get("grad_clip_norm", 1.0))
    grad_accum = int(train_cfg.get("grad_accumulation_steps", train_cfg.get("grad_accum_steps", 1)))

    # EMA target of student
    ema_decay = float(cfg["ct"].get("ema_decay", 0.999))
    ema = EMA(student, decay=ema_decay)

    # Optional teacher (for consistency distillation)
    teacher_ckpt = cfg["ct"].get("teacher_ckpt", "") or ""
    if teacher_ckpt:
        teacher_cfg_path = cfg["ct"].get("teacher_config", args.config)
        teacher = load_teacher(teacher_ckpt, teacher_cfg_path, device)
        def teacher_denoise_fn(x, sigmas):
            return teacher.denoise_fn(x, sigmas=sigmas)
    else:
        teacher = None
        teacher_denoise_fn = None

    # Target denoiser comes from EMA of the student
    target = copy.deepcopy(student).to(device)
    for p in target.parameters():
        p.requires_grad_(False)

    # Karras spec
    spec = KarrasScheduleSpec(
        sigma_min=float(cfg["ct"].get("sigma_min", 0.002)),
        sigma_max=float(cfg["ct"].get("sigma_max", 5.0)),
        rho=float(cfg["ct"].get("rho", 7.0)),
        num_scales=int(cfg["ct"].get("num_scales", 64)),
    )
    weight_schedule = str(cfg["ct"].get("weight_schedule", "karras"))
    loss_norm = str(cfg["ct"].get("loss_norm", "l2"))
    sigma_data = float(cfg["model"]["diffusion"]["sigma_data"])

    # I/O
    out_dir = Path(train_cfg.get("out_dir", "checkpoints/ct_run"))
    out_dir.mkdir(parents=True, exist_ok=True)
    save_every = int(train_cfg.get("save_every", 1000))
    log_every = int(train_cfg.get("log_every", 10))
    state_dtype = train_cfg.get("state_dtype", "fp32")

    max_steps = int(train_cfg.get("max_steps", 100000))
    step = 0
    accum_loss = 0.0

    student.train()

    while step < max_steps:
        for batch in train_loader:
            if step >= max_steps:
                break

            wav_stems = batch["waveform_stems"].to(device)
            with torch.no_grad():
                latents = encode_stems(ae, wav_stems).to(device)

            # Define denoise fns using Diffusion.denoise_fn for correct scaling
            def student_denoise_fn(x, sigmas):
                return student.denoise_fn(x, sigmas=sigmas)

            def target_denoise_fn(x, sigmas):
                return target.denoise_fn(x, sigmas=sigmas)

            loss = consistency_loss(
                student_denoise_fn=student_denoise_fn,
                target_denoise_fn=target_denoise_fn,
                x_start=latents,
                spec=spec,
                sigma_data=sigma_data,
                weight_schedule=weight_schedule,
                loss_norm=loss_norm,
                teacher_denoise_fn=teacher_denoise_fn,
            ) / grad_accum

            accum_loss += float(loss.detach().cpu())
            loss.backward()
            step += 1

            if step % grad_accum == 0:
                nn.utils.clip_grad_norm_(student.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)

                # Update EMA and copy to target
                ema.update(student)
                ema.copy_to(target)

                opt_step = step // grad_accum
                if opt_step % log_every == 0:
                    print(f"[CT] step {step} | loss {accum_loss:.6f}")
                    accum_loss = 0.0

            if step % save_every == 0:
                payload = {
                    "step": step,
                    "model": _tensor_dict_cpu_cast(student.state_dict(), state_dtype, to_cpu=True),
                    "optimizer": optimizer.state_dict(),
                    "ema": _tensor_dict_cpu_cast(ema.state_dict(), state_dtype, to_cpu=True),
                    "ema_decay": ema.decay,
                }
                ckpt = out_dir / f"ct_step_{step}.pt"
                _safe_save(payload, ckpt)
                print(f"[CT] saved: {ckpt}")

                # Optional FAD evaluation (light). We compare synthetic samples to a small slice of val set.
                fad_cfg = cfg.get("fad", {"enabled": True})
                eval_every = int(cfg["train"].get("eval_every", 0))
                if eval_every and (opt_step % eval_every == 0) and fad_cfg.get("enabled", True):
                    try:
                        # Build a tiny val loader from the same datamodule
                        dm.setup(stage="validate")
                        val_loader = dm.val_dataloader()
                        sr = cfg["data"]["params"]["preprocessing"]["audio"]["sampling_rate"]
                        num_eval = int(fad_cfg.get("num_eval", 8))
                        steps_eval = int(fad_cfg.get("steps", 40))

                        # Determine latent length from one encode
                        first = next(iter(val_loader))
                        if "waveform_stems" in first:
                            ex_lat = encode_stems(ae, first["waveform_stems"][:1])
                            latent_len = int(ex_lat.shape[-1])
                        else:
                            latent_len = 128

                        # References from val set
                        refs = []
                        for b in val_loader:
                            if "waveform_stems" in b:
                                wav = torch.tensor(b["waveform_stems"]).float()
                                mix = wav.sum(dim=1)  # [B, T]
                                for i in range(mix.shape[0]):
                                    refs.append(mix[i].cpu())
                            else:
                                wav = b["waveform"].float()
                                for i in range(wav.shape[0]):
                                    refs.append(wav[i].cpu())
                            if len(refs) >= num_eval:
                                break
                        refs = refs[:num_eval]

                        # Generate samples
                        shape = (num_eval, in_channels, latent_len)
                        schedule = KarrasSchedule(sigma_min=spec.sigma_min, sigma_max=spec.sigma_max, rho=spec.rho)
                        sigmas = schedule(steps_eval, device)
                        sampler = KarrasSampler()
                        noise = torch.randn(shape, device=device)
                        latents = sampler(noise, lambda x, sigma: student.denoise_fn(x, sigma=sigma), sigmas, steps_eval)
                        gens = _decode_latents_to_mix(ae, latents, num_stems, latent_dim)

                        fad_val = compute_fad(refs, gens, sr)
                        print(f"[CT][Eval] FAD: {fad_val:.4f}")
                    except Exception as e:
                        print(f"[CT][Eval] FAD failed: {e}")

    # Final save
    final = out_dir / "ct_final.pt"
    payload = {
        "step": step,
        "model": _tensor_dict_cpu_cast(student.state_dict(), state_dtype, to_cpu=True),
        "ema": _tensor_dict_cpu_cast(ema.state_dict(), state_dtype, to_cpu=True),
        "ema_decay": ema.decay,
    }
    _safe_save(payload, final)
    print(f"[CT] done. saved: {final}")


if __name__ == "__main__":
    main()
