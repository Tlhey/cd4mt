"""CTM (Consistency Trajectory Model) for multitrack audio generation with CAE"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from einops import rearrange, reduce

# Add CTM path to sys.path
CTM_PATH = "/data1/yuchen/music/consistency_models"
if CTM_PATH not in sys.path:
    sys.path.append(CTM_PATH)

# CAE codec
from music2latent import EncoderDecoder

# CTM imports
from cm import dist_util, logger
from cm.script_util import (
    create_model_and_diffusion,
    model_and_diffusion_defaults,
    cm_train_defaults,
)
from cm.train_util import CMTrainLoop

# utils
from ldm.modules.util import count_params, summarize_params


class CTMMultitrackModel(pl.LightningModule):
    """CTM (Consistency Trajectory Model) for multitrack audio generation"""

    def __init__(
        self,
        # CTM config
        image_size: int = 32,
        num_channels: int = 128,
        num_res_blocks: int = 2,
        channel_mult: str = "1,2,4",
        attention_resolutions: str = "16,8",
        dropout: float = 0.0,
        class_cond: bool = False,
        use_checkpoint: bool = False,
        use_scale_shift_norm: bool = True,
        resblock_updown: bool = False,
        use_fp16: bool = False,
        use_new_attention_order: bool = False,

        # Diffusion parameters
        sigma_min: float = 0.002,
        sigma_max: float = 80.0,
        weight_schedule: str = "karras",

        # CTM specific
        training_mode: str = "ctm",
        teacher_dropout: float = 0.1,
        target_ema_mode: str = "fixed",
        scale_mode: str = "fixed",
        start_ema: float = 0.0,
        start_scales: int = 40,
        end_scales: int = 40,

        # CAE config
        cae_latent_dim: int = 64,
        cae_z_channels: int = 64,
        sample_rate: int = 44100,

        # Multitrack setup
        num_stems: int = 4,
        stem_names: List[str] = ["bass", "drums", "guitar", "piano"],
        support_mixture: bool = True,

        # Training
        base_learning_rate: float = 1e-4,

        # Sampling
        sampling_steps: int = 1,  # CTM supports single-step sampling

        # Monitoring
        monitor: str = "val/loss",

        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()

        # =========================
        # 1. Audio Auto-Encoder (CAE)
        # =========================
        # CAE 在首次使用时根据张量设备来初始化，避免默认 cuda:0
        self.autoencoder = None
        self.cae_latent_dim = cae_latent_dim
        self.cae_z_channels = cae_z_channels
        self.sample_rate = sample_rate

        # =========================
        # 2. Multi-track Configuration
        # =========================
        self.num_stems = num_stems
        self.stem_names = stem_names
        self.support_mixture = support_mixture

        # =========================
        # 3. CTM Model Setup
        # =========================

        # 3.1 Calculate input channels
        if support_mixture:
            # Support mixture + stems: (1+S)*C_lat
            diffusion_in_channels = num_stems * cae_z_channels
        else:
            # Only stems: S*C_lat
            diffusion_in_channels = num_stems * cae_z_channels

        # 3.2 Create CTM model arguments
        self.ctm_args = type('Args', (), {})()

        # Model arguments
        self.ctm_args.image_size = image_size
        self.ctm_args.num_channels = num_channels
        self.ctm_args.num_res_blocks = num_res_blocks
        self.ctm_args.channel_mult = channel_mult
        self.ctm_args.attention_resolutions = attention_resolutions
        self.ctm_args.dropout = dropout
        self.ctm_args.class_cond = class_cond
        self.ctm_args.use_checkpoint = use_checkpoint
        self.ctm_args.use_scale_shift_norm = use_scale_shift_norm
        self.ctm_args.resblock_updown = resblock_updown
        self.ctm_args.use_fp16 = use_fp16
        self.ctm_args.use_new_attention_order = use_new_attention_order
        self.ctm_args.learn_sigma = False

        # Diffusion arguments
        self.ctm_args.sigma_min = sigma_min
        self.ctm_args.sigma_max = sigma_max
        self.ctm_args.weight_schedule = weight_schedule

        # CTM specific arguments
        self.ctm_args.training_mode = training_mode
        self.ctm_args.teacher_dropout = teacher_dropout
        self.ctm_args.target_ema_mode = target_ema_mode
        self.ctm_args.scale_mode = scale_mode
        self.ctm_args.start_ema = start_ema
        self.ctm_args.start_scales = start_scales
        self.ctm_args.end_scales = end_scales

        # 3.3 Create CTM model and diffusion
        print(f"🏗️ Creating CTM model with {diffusion_in_channels} channels")

        # We need to modify the UNet to accept our multitrack input channels
        # Temporarily set in_channels to our multitrack dimension
        original_create_model = create_model_and_diffusion

        def custom_create_model_and_diffusion(*args, **kwargs):
            # Override in_channels for multitrack audio
            model_kwargs = dict(vars(self.ctm_args))
            # CTM expects 3 channels for RGB images, but we need multitrack channels
            # We'll need to create a custom UNet or modify the existing one
            return original_create_model(
                image_size=self.ctm_args.image_size,
                class_cond=self.ctm_args.class_cond,
                learn_sigma=self.ctm_args.learn_sigma,
                num_channels=self.ctm_args.num_channels,
                num_res_blocks=self.ctm_args.num_res_blocks,
                channel_mult=self.ctm_args.channel_mult,
                num_heads=4,
                num_head_channels=-1,
                num_heads_upsample=-1,
                attention_resolutions=self.ctm_args.attention_resolutions,
                dropout=self.ctm_args.dropout,
                use_checkpoint=self.ctm_args.use_checkpoint,
                use_scale_shift_norm=self.ctm_args.use_scale_shift_norm,
                resblock_updown=self.ctm_args.resblock_updown,
                use_fp16=self.ctm_args.use_fp16,
                use_new_attention_order=self.ctm_args.use_new_attention_order,
                sigma_min=self.ctm_args.sigma_min,
                sigma_max=self.ctm_args.sigma_max,
                weight_schedule=self.ctm_args.weight_schedule,
                distillation=(training_mode == "ctm")
            )

        self.model, self.diffusion = custom_create_model_and_diffusion()

        # Modify the UNet input/output channels for multitrack audio
        if hasattr(self.model, 'input_blocks') and hasattr(self.model.input_blocks[0][0], 'weight'):
            # Modify first conv layer to accept multitrack input
            original_conv = self.model.input_blocks[0][0]
            new_conv = nn.Conv2d(
                diffusion_in_channels,
                original_conv.out_channels,
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )

            # Initialize new conv layer
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            if new_conv.bias is not None:
                nn.init.constant_(new_conv.bias, 0)

            self.model.input_blocks[0][0] = new_conv

        if hasattr(self.model, 'out') and hasattr(self.model.out[-1], 'weight'):
            # Modify output layer
            original_conv = self.model.out[-1]
            new_conv = nn.Conv2d(
                original_conv.in_channels,
                diffusion_in_channels,  # Output should match input channels
                kernel_size=original_conv.kernel_size,
                stride=original_conv.stride,
                padding=original_conv.padding,
                bias=original_conv.bias is not None
            )

            # Initialize new conv layer
            nn.init.kaiming_normal_(new_conv.weight, mode='fan_out', nonlinearity='relu')
            if new_conv.bias is not None:
                nn.init.constant_(new_conv.bias, 0)

            self.model.out[-1] = new_conv

        # =========================
        # 4. Training Configuration
        # =========================
        self.learning_rate = float(base_learning_rate)
        self.monitor = monitor
        self.sampling_steps = sampling_steps

        # =========================
        # 5. Logging
        # =========================
        print(f"✅ CTM Multitrack Model initialized:")
        print(f"   - CAE latent dim: {cae_latent_dim}")
        print(f"   - CAE z channels: {cae_z_channels}")
        print(f"   - Num stems: {num_stems}")
        print(f"   - Support mixture: {support_mixture}")
        print(f"   - Diffusion in channels: {diffusion_in_channels}")
        print(f"   - Learning rate: {self.learning_rate}")
        print(f"   - Sampling steps: {sampling_steps}")
        print(f"   - Training mode: {training_mode}")

        # 参数规模与内存占用
        try:
            summarize_params(self.model, name="CTM.UNet")
        except Exception:
            total_params = sum(p.numel() for p in self.model.parameters())
            print(f"CTM Model has {total_params:,} parameters.")

    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        Use CAE to encode audio to latent space

        Args:
            audio: (B, S, T) - multi-stem audio waveforms

        Returns:
            latents: (B, S, C_lat, L) - CAE latent representation
        """
        # Ensure input is 3D: (B, S, T)
        if audio.dim() == 2:
            # (B, T) -> (B, 1, T)
            audio = audio.unsqueeze(1)

        batch_size, num_stems, audio_length = audio.shape

        # Ensure CAE exists on the right device
        if self.autoencoder is None:
            dev = getattr(self, 'device', None) or audio.device
            self.autoencoder = EncoderDecoder(device=dev)

        # Check stems count
        if num_stems != self.num_stems:
            print(f"Warning: Expected {self.num_stems} stems, got {num_stems}. Using available stems.")

        # Encode each stem separately
        latents_list = []
        for b in range(batch_size):
            batch_latents = []
            for s in range(num_stems):
                try:
                    # Get single sample's single stem: [T]
                    stem_audio = audio[b, s].cpu().numpy()  # [T]

                    # Convert to CAE expected format: [1, T] (channels, samples)
                    stem_audio = stem_audio.reshape(1, -1)  # [1, T]

                    # CAE encode: [1, T] -> [1, 64, L]
                    stem_latents = self.autoencoder.encode(
                        stem_audio,
                        max_waveform_length=32768,
                        max_batch_size=1
                    )

                    # Convert to torch tensor
                    if isinstance(stem_latents, np.ndarray):
                        stem_latents = torch.from_numpy(stem_latents).to(audio.device)

                    # Convert to float32
                    stem_latents = stem_latents.to(dtype=torch.float32)

                    # Ensure correct dimensions [C, L] format
                    if stem_latents.dim() == 3 and stem_latents.size(0) == 1:
                        stem_latents = stem_latents.squeeze(0)  # [1, 64, L] -> [64, L]
                    elif stem_latents.dim() == 2:
                        pass  # Already [64, L] format
                    else:
                        raise ValueError(f"Unexpected CAE output shape: {stem_latents.shape}")

                    batch_latents.append(stem_latents)

                except Exception as e:
                    print(f"Error encoding stem {s} of batch {b}: {e}")
                    raise e

            # Stack all stems for current batch: [S, 64, L]
            batch_latents = torch.stack(batch_latents, dim=0)
            latents_list.append(batch_latents)

        # Stack all batches: [B, S, 64, L]
        latents = torch.stack(latents_list, dim=0)

        return latents

    def decode_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Use CAE to decode latent representation to audio

        Args:
            latents: (B, S, C_lat, L) - CAE latent representation

        Returns:
            audio: (B, S, T) - audio waveforms
        """
        # Ensure CAE exists on the right device
        if self.autoencoder is None:
            dev = getattr(self, 'device', None) or latents.device
            self.autoencoder = EncoderDecoder(device=dev)

        batch_size, num_channels, latent_channels, latent_length = latents.shape

        # Decode each channel separately
        audio_list = []
        for i in range(num_channels):
            channel_latents = latents[:, i]  # (B, C_lat, L)
            channel_audio = self.autoencoder.decode(channel_latents.cpu().numpy())

            # Convert back to tensor
            if isinstance(channel_audio, np.ndarray):
                channel_audio = torch.from_numpy(channel_audio).to(latents.device)
            elif isinstance(channel_audio, torch.Tensor):
                channel_audio = channel_audio.to(latents.device)

            audio_list.append(channel_audio.unsqueeze(1))  # (B, 1, T)

        # Concatenate all channels
        audio = torch.cat(audio_list, dim=1)  # (B, S, T)

        return audio

    def prepare_diffusion_input(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Convert CAE latent representation to diffusion model input format

        Args:
            latents: (B, S, C_lat, L) - CAE latent representation

        Returns:
            diffusion_input: (B, S*C_lat, H, W) - diffusion model input
        """
        B, S, C, L = latents.shape

        # Reshape to 2D input: ensure H and W dimensions are both > 1
        total_channels = S * C

        # Convert sequence length to approximately square H×W
        import math
        side_length = int(math.sqrt(L))
        if side_length * side_length < L:
            side_length += 1

        # Pad to square
        padded_length = side_length * side_length
        padding = padded_length - L

        # (B, S, C, L) -> (B, S*C, L) -> pad -> (B, S*C, H, W)
        flat_latents = latents.view(B, total_channels, L)
        if padding > 0:
            flat_latents = torch.nn.functional.pad(flat_latents, (0, padding))

        diffusion_input = flat_latents.view(B, total_channels, side_length, side_length)

        print(f"Diffusion input shape: {diffusion_input.shape}, dtype: {diffusion_input.dtype}")

        return diffusion_input

    def restore_diffusion_output(self, diffusion_output: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        Convert diffusion model output back to CAE latent representation format

        Args:
            diffusion_output: (B, S*C_lat, H, W) - diffusion model output
            target_shape: (B, S, C_lat, L) - target shape

        Returns:
            latents: (B, S, C_lat, L) - CAE latent representation
        """
        B, S, C, L = target_shape

        print(f"🔄 Restoring diffusion output: {diffusion_output.shape} -> {target_shape}")

        # Handle 4D to target shape conversion
        if diffusion_output.dim() == 4:
            batch_size, total_channels, H, W = diffusion_output.shape

            # Verify channel count matches
            expected_channels = S * C
            if total_channels != expected_channels:
                raise ValueError(f"Channel mismatch: expected {expected_channels}, got {total_channels}")

            # Flatten H and W dimensions: (B, S*C, H, W) -> (B, S*C, H*W)
            flat_output = diffusion_output.view(batch_size, total_channels, H * W)

            # Take first L elements: (B, S*C, H*W) -> (B, S*C, L)
            if H * W >= L:
                flat_output = flat_output[:, :, :L]
            else:
                # If generated length is not enough, pad with zeros
                padding = L - (H * W)
                flat_output = torch.nn.functional.pad(flat_output, (0, padding))

        elif diffusion_output.dim() == 3:
            # Already in (B, S*C, L) format
            flat_output = diffusion_output
        else:
            raise ValueError(f"Unsupported diffusion output dimensions: {diffusion_output.dim()}")

        # Reshape to target shape: (B, S*C, L) -> (B, S, C, L)
        latents = flat_output.view(B, S, C, L)

        print(f"✅ Restoration complete: {latents.shape}")
        return latents

    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Forward pass - compute diffusion loss

        Args:
            x: (B, S, C_lat, L) - CAE latent representation

        Returns:
            loss: diffusion loss
        """
        # Prepare diffusion input
        diffusion_input = self.prepare_diffusion_input(x)

        # Sample timesteps
        batch_size = diffusion_input.shape[0]
        timesteps = torch.randint(0, 1000, (batch_size,), device=diffusion_input.device)

        # Add noise
        noise = torch.randn_like(diffusion_input)
        noisy_input = self.diffusion.q_sample(diffusion_input, timesteps, noise=noise)

        # Predict noise
        predicted = self.model(noisy_input, timesteps)

        # Compute loss (MSE between predicted and actual noise)
        loss = F.mse_loss(predicted, noise)

        return loss

    def sample(
        self,
        shape: Tuple[int, ...],
        num_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        Sample to generate new audio

        Args:
            shape: (B, S, C_lat, L) - generation shape
            num_steps: sampling steps (CTM can do single-step)

        Returns:
            generated_audio: (B, S, T) - generated audio
        """
        if num_steps is None:
            num_steps = self.sampling_steps

        B, S, C, L = shape

        # Create noise input - must match training prepare_diffusion_input output shape
        total_channels = S * C

        # Calculate same 2D dimensions as prepare_diffusion_input
        import math
        side_length = int(math.sqrt(L))
        if side_length * side_length < L:
            side_length += 1

        # Noise shape must match training (B, S*C, H, W) format
        noise_shape = (B, total_channels, side_length, side_length)
        noise = torch.randn(noise_shape, device=self.device)

        print(f"✅ Noise shape: {noise_shape} (matches training prepare_diffusion_input output)")

        # CTM sampling - single step or few steps
        with torch.no_grad():
            if num_steps == 1:
                # Single-step CTM sampling
                generated_latents = self.model(noise, torch.zeros(B, device=self.device, dtype=torch.long))
            else:
                # Multi-step sampling (use DDIM or similar)
                sample = noise
                timesteps = torch.linspace(999, 0, num_steps, dtype=torch.long, device=self.device)

                for i, t in enumerate(timesteps):
                    t_batch = t.repeat(B)
                    with torch.no_grad():
                        predicted = self.model(sample, t_batch)

                    # Simple Euler step (can be replaced with more sophisticated samplers)
                    if i < len(timesteps) - 1:
                        dt = timesteps[i] - timesteps[i+1]
                        sample = sample - dt * predicted / 1000.0
                    else:
                        sample = predicted

                generated_latents = sample

        # Restore latent representation shape
        generated_latents = self.restore_diffusion_output(generated_latents, shape)

        # Decode to audio
        generated_audio = self.decode_audio(generated_latents)

        return generated_audio

    def training_step(self, batch, batch_idx):
        """Training step"""
        try:
            # Get audio data - prioritize stems data
            if isinstance(batch, dict):
                audio = batch.get('waveform_stems', batch.get('waveform', batch.get('audio', None)))
            else:
                audio = batch

            if audio is None:
                raise ValueError("Cannot get audio data from batch")

            # Encode audio
            latents = self.encode_audio(audio)

            # Compute diffusion loss
            loss = self.forward(latents)

            # Log loss
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)

            return loss

        except Exception as e:
            print(f"❌ Error in training_step: {e}")
            raise e

    def validation_step(self, batch, batch_idx):
        """Validation step"""
        try:
            # Get audio data - prioritize stems data
            if isinstance(batch, dict):
                audio = batch.get('waveform_stems', batch.get('waveform', batch.get('audio', None)))
            else:
                audio = batch

            if audio is None:
                raise ValueError("Cannot get audio data from batch")

            # Encode audio
            latents = self.encode_audio(audio)

            # Compute diffusion loss
            val_loss = self.forward(latents)

            # Log loss
            self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)

            return val_loss

        except Exception as e:
            print(f"❌ Error in validation_step: {e}")
            raise e

    def configure_optimizers(self):
        """Configure optimizers"""
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=1000,
            eta_min=self.learning_rate * 0.1
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.monitor,
                "interval": "epoch",
                "frequency": 1,
            }
        }


# Compatibility functions
def create_ctm_model_from_config(config: Dict[str, Any]) -> CTMMultitrackModel:
    """Create CTM model from configuration"""
    return CTMMultitrackModel(**config)


# Legacy aliases for backwards compatibility
MyModelCTMMultitrack = CTMMultitrackModel
        # 确保已初始化 CAE 到正确设备
        if self.autoencoder is None:
            self.autoencoder = EncoderDecoder()
            # 该 EncoderDecoder 需要显式 device；若其实现要求 device，则在调用处按需调整
