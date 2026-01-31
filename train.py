#!/usr/bin/env python
#  conda activate  /home/yuchen/.local/share/mamba/envs/cdp10 
# python train.py --config configs/test_cfg.yaml
"""
train.py - CD4MT Training Script
Uses 1D diffusion with UNet1d.
Input shape: [B, S*C, L] where S=num_stems, C=latent_channels, L=latent_length.
"""

import sys
import argparse
import yaml
import numpy as np
import torch
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from music2latent import EncoderDecoder
from DataLoader.multitrack_datamodule import DataModuleFromConfig
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


def train(cfg: dict):
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

    # Train loop
    diffusion.train()
    step = 0
    epoch = 0
    accum_loss = 0.0

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
                torch.nn.utils.clip_grad_norm_(diffusion.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

                # Log (use accumulated loss)
                if (step // grad_accum) % log_every == 0:
                    print(f"[Epoch {epoch}] Step {step}/{max_steps} | Loss: {accum_loss:.6f}")
                    if swanlab:
                        swanlab.log({"loss": accum_loss, "step": step})
                accum_loss = 0.0

            # Save
            if step % save_every == 0:
                ckpt = out_dir / f"step_{step}.pt"
                torch.save({"step": step, "model": diffusion.state_dict()}, ckpt)
                print(f"Saved: {ckpt}")

    # Final save
    torch.save({"step": step, "model": diffusion.state_dict()}, out_dir / "final.pt")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/test_cfg.yaml")
    args = parser.parse_args()
    train(load_config(args.config))
