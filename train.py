#!/usr/bin/env python
#  conda activate /root/micromamba/envs/cdp10
# python train.py --config configs/test_cfg.yaml
"""
train.py - CD4MT Training Script
Uses 1D diffusion with UNet1d.
Input shape: [B, S*C, L] where S=num_stems, C=latent_channels, L=latent_length.
"""

import sys
import os
import argparse
import yaml
import numpy as np
import torch
# Allow PyTorch 2.6+ safe unpickler to load older checkpoints
try:
    torch.serialization.add_safe_globals([np.core.multiarray.scalar])
except Exception:
    pass
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from music2latent import EncoderDecoder
from DataLoader.multitrack_datamodule import DataModuleFromConfig
from score.diffusion import Diffusion, LogNormalDistribution, KarrasSchedule, KarrasSampler, DiffusionSampler
from score.modules import UNet1d


def _strip_module_prefix(state_dict: dict) -> dict:
    if not isinstance(state_dict, dict):
        return state_dict
    return {k.replace("module.", ""): v for k, v in state_dict.items()}


def _load_resume_ckpt(path: str, device: torch.device):
    """Load checkpoint for resume training.
    Returns: (state_dict, step, optim_state or None, ema_state or None)
    - Forces weights_only=False for compatibility with older pickles.
    - Accepts {model: ...} / {state_dict: ...} / raw dict.
    - Strips DataParallel prefixes.
    """
    ckpt = torch.load(path, map_location=device, weights_only=False)
    ema_state = ckpt.get("ema") if isinstance(ckpt, dict) else None
    if isinstance(ckpt, dict):
        state = ckpt.get("model") or ckpt.get("state_dict") or ckpt
        step = ckpt.get("step") or ckpt.get("global_step") or ckpt.get("epoch") or 0
        opt_state = ckpt.get("optimizer") or ckpt.get("optim")
    else:
        state, step, opt_state = ckpt, 0, None
    state = _strip_module_prefix(state)
    try:
        step = int(step)
    except Exception:
        step = 0
    return state, step, opt_state, ema_state


class EMA:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = float(decay)
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: torch.nn.Module):
        for k, p in model.named_parameters():
            if not p.requires_grad:
                continue
            if k not in self.shadow:
                self.shadow[k] = p.detach().clone()
                continue
            self.shadow[k].mul_(self.decay).add_(p.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return {k: v.clone() for k, v in self.shadow.items()}

    def load_state_dict(self, state: dict):
        for k, v in state.items():
            if k in self.shadow:
                self.shadow[k].copy_(v)

    @torch.no_grad()
    def apply_to(self, model: torch.nn.Module):
        for k, p in model.named_parameters():
            if k in self.shadow:
                p.data.copy_(self.shadow[k])


@torch.no_grad()
def _param_delta_and_update(prev: list | None, model: torch.nn.Module):
    """Compute mean |Δparam| across all parameters and refresh snapshots.
    Returns (delta, new_prev_list).
    """
    delta = float("nan")
    if prev is not None:
        vals = []
        for p_prev, p in zip(prev, model.parameters()):
            vals.append((p - p_prev).abs().mean().item())
        if vals:
            delta = float(np.mean(vals))
    new_prev = [p.detach().clone() for p in model.parameters()]
    return delta, new_prev


def load_config(path: str) -> dict:
    with open(path, "r") as f:
        return yaml.safe_load(f)


def _safe_save(obj, path: Path) -> None:
    """Robust checkpoint save to avoid inline_container pos errors.
    - Writes to a temporary file first, then atomically renames.
    - Uses legacy zipfile serialization to avoid backend assert on some FS.
    """
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    # Ensure directory exists
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(obj, tmp, _use_new_zipfile_serialization=False)
    os.replace(tmp, path)


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


def _model_state_for_save(model: torch.nn.Module, dtype_name: str | None, to_cpu: bool = True) -> dict:
    return _tensor_dict_cpu_cast(model.state_dict(), dtype_name, to_cpu)


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


def encode_stems(ae: EncoderDecoder, wav_stems: torch.Tensor) -> torch.Tensor:
    """Encode [B, S, T] waveforms to [B, S*C, L] latents."""
    B, S, T = wav_stems.shape
    latents_list = []

    for s in range(S):
        stem_audio = wav_stems[:, s].cpu().numpy()
        latent = ae.encode(stem_audio)
        if isinstance(latent, np.ndarray):
            latent = torch.from_numpy(latent)
        latents_list.append(latent.float())

    # [B, S, C, L] -> [B, S*C, L]
    stacked = torch.stack(latents_list, dim=1)
    B, S, C, L = stacked.shape
    return stacked.view(B, S * C, L)


# (Previous helper definitions replaced by the extended versions above.)


def train(cfg: dict, resume: str | None = None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(cfg["train"].get("seed", 42))

    # Data
    dm = DataModuleFromConfig(**cfg["data"]["params"])
    dm.prepare_data()
    dm.setup(stage="fit")
    train_loader = dm.train_dataloader()
    val_loader = dm.val_dataloader()
    print(f"Train batches: {len(train_loader)}")

    # CAE
    ae = EncoderDecoder(device=device)

    # Model
    in_channels = cfg["model"]["cae"]["num_stems"] * cfg["model"]["cae"]["latent_dim"]
    unet = build_unet(cfg, in_channels)
    diffusion = build_diffusion(cfg, unet).to(device)

    num_params = sum(p.numel() for p in diffusion.parameters() if p.requires_grad)
    print(f"Parameters: {num_params / 1e6:.2f}M")

    # Optimizer
    optimizer = torch.optim.AdamW(diffusion.parameters(), lr=cfg["train"]["lr"])

    # Training config
    max_steps = cfg["train"]["max_steps"]
    log_every = cfg["train"]["log_every"]
    save_every = cfg["train"]["save_every"]
    grad_clip = cfg["train"].get("grad_clip_norm", 1.0)
    grad_accum = cfg["train"].get("grad_accumulation_steps", 1)
    out_dir = Path(cfg["train"]["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    # SwanLab
    swanlab_cfg = cfg["train"].get("swanlab", {})
    if swanlab_cfg.get("project"):
        try:
            import swanlab
            swanlab.init(project=swanlab_cfg["project"], config=cfg)
        except:
            swanlab = None
    else:
        swanlab = None

    # EMA
    ema_cfg = cfg["train"].get("ema", {})
    ema_enabled = bool(ema_cfg.get("enabled", False))
    ema_decay = float(ema_cfg.get("decay", 0.999))
    ema = EMA(diffusion, decay=ema_decay) if ema_enabled else None
    # Checkpoint size controls
    ckpt_cfg = cfg["train"].get("checkpoint", {})
    save_optimizer = bool(ckpt_cfg.get("save_optimizer", True))
    save_ema = bool(ckpt_cfg.get("save_ema", True))
    state_dtype = ckpt_cfg.get("state_dtype", "fp32")  # fp32|fp16|bf16|auto

    # Optional resume
    resume_path = resume or cfg.get("train", {}).get("resume_ckpt")
    step = 0
    if resume_path:
        try:
            state, prev_step, opt_state, ema_state = _load_resume_ckpt(resume_path, device)
            missing, unexpected = diffusion.load_state_dict(state, strict=False)
            if len(missing) or len(unexpected):
                print(f"[Resume] Non-strict load: missing={len(missing)}, unexpected={len(unexpected)}")
            if opt_state:
                try:
                    optimizer.load_state_dict(opt_state)
                except Exception as e:
                    print(f"[Resume] Optimizer state not loaded: {e}")
            if ema and ema_state:
                try:
                    ema.load_state_dict(ema_state)
                except Exception as e:
                    print(f"[Resume] EMA state not loaded: {e}")
            step = prev_step
            print(f"[Resume] Loaded '{resume_path}' at step {prev_step}.")
        except Exception as e:
            print(f"[Resume] Failed to load '{resume_path}': {e}")

    # Train loop
    diffusion.train()
    epoch = 0
    accum_loss = 0.0
    # Track parameter snapshots and stall threshold for optimizer update check
    prev_params = [p.detach().clone() for p in diffusion.parameters()]
    param_delta_warn_threshold = float(cfg["train"].get("param_delta_warn_threshold", 1e-12))

    print(f"Gradient accumulation steps: {grad_accum}")
    print(f"Effective batch size: {cfg['data']['params']['batch_size'] * grad_accum}")

    while step < max_steps:
        epoch += 1
        for batch in train_loader:
            if step >= max_steps:
                break

            # Encode
            wav_stems = batch["waveform_stems"]
            with torch.no_grad():
                latents = encode_stems(ae, wav_stems).to(device)

            # Debug: print latent stats on first step and validate sigma_data
            if step == 0:
                latent_std = latents.std().item()
                latent_mean = latents.mean().item()
                print(f"Latent stats: mean={latent_mean:.4f}, std={latent_std:.4f}, "
                      f"min={latents.min():.4f}, max={latents.max():.4f}")

                # Validate sigma_data
                sigma_data = cfg["model"]["diffusion"]["sigma_data"]
                if abs(latent_std - sigma_data) > 0.3 * sigma_data:
                    print(f"WARNING: sigma_data={sigma_data} but latent std={latent_std:.4f}!")
                    print(f"         Consider setting sigma_data to {latent_std:.2f} in config for better training.")

            # Forward (scale loss for gradient accumulation)
            loss = diffusion(latents) / grad_accum
            accum_loss += loss.item()

            # Backward (accumulate gradients)
            loss.backward()

            # Step optimizer every grad_accum batches
            step += 1
            if step % grad_accum == 0:
                grad_total_norm = float(torch.nn.utils.clip_grad_norm_(diffusion.parameters(), grad_clip))
                optimizer.step()
                optimizer.zero_grad()

                # EMA update
                if ema:
                    ema.update(diffusion)

                # Parameter delta logging
                with torch.no_grad():
                    deltas = []
                    for p, p_prev in zip(diffusion.parameters(), prev_params):
                        deltas.append((p - p_prev).abs().mean().item())
                        p_prev.copy_(p)
                    param_delta = float(np.mean(deltas)) if deltas else float('nan')

                # Log (use accumulated loss)
                if (step // grad_accum) % log_every == 0:
                    stalled = (not np.isnan(param_delta)) and (param_delta < param_delta_warn_threshold)
                    print(f"[Epoch {epoch}] Step {step}/{max_steps} | Loss: {accum_loss:.6f} | grad_norm: {grad_total_norm:.4g} | param_delta: {param_delta:.3e}{' STALLED' if stalled else ''}")
                    if swanlab:
                        try:
                            swanlab.log({
                                "loss": accum_loss,
                                "step": step,
                                "grad_norm": grad_total_norm,
                                "param_delta": None if np.isnan(param_delta) else param_delta,
                                "optimizer_stalled": bool(stalled),
                            })
                        except Exception:
                            pass
                accum_loss = 0.0

            # Save
            if step % save_every == 0:
                ckpt = out_dir / f"step_{step}.pt"
                payload = {"step": step}
                payload["model"] = _model_state_for_save(diffusion, state_dtype, to_cpu=True)
                if save_optimizer:
                    payload["optimizer"] = optimizer.state_dict()
                if ema and save_ema:
                    payload["ema"] = _tensor_dict_cpu_cast(ema.state_dict(), state_dtype, to_cpu=True)
                    payload["ema_decay"] = ema_decay
                _safe_save(payload, ckpt)
                print(f"Saved: {ckpt}")

    # Final save
    payload = {"step": step}
    payload["model"] = _model_state_for_save(diffusion, state_dtype, to_cpu=True)
    if save_optimizer:
        payload["optimizer"] = optimizer.state_dict()
    if ema and save_ema:
        payload["ema"] = _tensor_dict_cpu_cast(ema.state_dict(), state_dtype, to_cpu=True)
        payload["ema_decay"] = ema_decay
    _safe_save(payload, out_dir / "final.pt")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test_cfg.yaml")
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, resume=args.resume)
