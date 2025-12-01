"""
MyModelScoreDiffusion: Score-based Diffusion Model for Multi-track Music Generation
====================================================================================

重构版本：继承AudioDiffusionModel_2d，集成CAE音频编解码器

设计理念:
1. 继承AudioDiffusionModel_2d获得完整的扩散功能
2. 集成music2latent CAE替代传统的VAE
3. 支持multi-track (stems)音乐生成
4. 使用EDM/VP-SDE噪声调度和Karras采样

Author: Based on MusicLDM architecture with CAE integration
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from einops import rearrange, reduce

# Music2Latent CAE
from music2latent import EncoderDecoder

# Audio Diffusion Components
from ctm_pl.audio_diffusion_pytorch_.model import AudioDiffusionModel_2d
from ctm_pl.audio_diffusion_pytorch_.diffusion import (
    LogNormalDistribution,
    KarrasSchedule,
    ADPM2Sampler
)

# OpenAI UNet
from ldm.modules.modules.diffusionmodules.openaimodel import UNetModel

# Utilities
from ldm.modules.util import instantiate_from_config, count_params


class MyModelScoreDiffusion(pl.LightningModule):
    """
    Score-based Diffusion Model for Multi-track Music Generation
    继承AudioDiffusionModel_2d，集成CAE音频编解码器
    
    Pipeline: Waveform → CAE Encode → Score Diffusion → CAE Decode → Waveform
    """
    
    def __init__(
        self,
        # UNet configuration
        unet_config: Dict[str, Any],
        
        # CAE configuration  
        cae_latent_dim: int = 64,
        cae_z_channels: int = 64,
        sample_rate: int = 44100,
        
        # Diffusion configuration
        diffusion_sigma_distribution: Optional[Dict] = None,
        diffusion_sigma_data: float = 0.5,
        diffusion_dynamic_threshold: float = 0.0,
        lambda_perceptual: float = 0.0,
        
        # Multi-track configuration
        num_stems: int = 4,
        stem_names: List[str] = ["bass", "drums", "guitar", "piano"],
        support_mixture: bool = True,
        
        # Training configuration
        base_learning_rate: float = 1e-4,
        
        # Sampling configuration
        sampling_steps: int = 50,
        sigma_min: float = 0.0001,
        sigma_max: float = 3.0,
        rho: float = 7.0,
        
        # Monitoring
        monitor: str = "val/loss",
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # =========================
        # 1. Audio Auto-Encoder (CAE)
        # =========================
        self.autoencoder = EncoderDecoder()
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
        # 3. Diffusion Model Setup
        # =========================
        
        # 3.1 计算UNet输入通道数
        if support_mixture:
            # 支持 mixture + stems: (1+S)*C_lat
            diffusion_in_channels = num_stems * cae_z_channels
        else:
            # 仅支持 stems: S*C_lat
            diffusion_in_channels = num_stems * cae_z_channels
            
        # 3.2 设置扩散参数
        if diffusion_sigma_distribution is None:
            diffusion_sigma_distribution = LogNormalDistribution(mean=-1.2, std=1.2)
        else:
            diffusion_sigma_distribution = instantiate_from_config(diffusion_sigma_distribution)
        
        # 3.3 更新UNet配置
        unet_config = unet_config.copy()
        unet_config["params"] = unet_config["params"].copy()
        unet_config["params"]["in_channels"] = diffusion_in_channels
        unet_config["params"]["out_channels"] = diffusion_in_channels
        
        print(f"🏗️ Creating UNet with {diffusion_in_channels} channels")
        print(f"   UNet config: {unet_config}")
        
        # 3.4 创建AudioDiffusionModel_2d
        self.audio_diffusion = AudioDiffusionModel_2d(
            diffusion_sigma_distribution=diffusion_sigma_distribution,
            diffusion_sigma_data=diffusion_sigma_data,
            diffusion_dynamic_threshold=diffusion_dynamic_threshold,
            lambda_perceptual=lambda_perceptual,
            unet_type='openAI',  # 使用OpenAI UNet
            **unet_config["params"]
        )
        
        # 3.5 采样组件
        self.sigma_schedule = KarrasSchedule(
            sigma_min=sigma_min, 
            sigma_max=sigma_max, 
            rho=rho
        )
        
        self.sampler = ADPM2Sampler(rho=1.0)
        self.sampling_steps = sampling_steps
        
        # =========================
        # 4. Training Configuration
        # =========================
        self.learning_rate = float(base_learning_rate)
        self.monitor = monitor
        
        # =========================
        # 5. Logging
        # =========================
        print(f"✅ MyModelScoreDiffusion initialized:")
        print(f"   - CAE latent dim: {cae_latent_dim}")
        print(f"   - CAE z channels: {cae_z_channels}")
        print(f"   - Num stems: {num_stems}")
        print(f"   - Support mixture: {support_mixture}")
        print(f"   - Diffusion in channels: {diffusion_in_channels}")
        print(f"   - Learning rate: {self.learning_rate}")
        print(f"   - Sampling steps: {sampling_steps}")
        
        # 计算参数数量
        count_params(self.audio_diffusion, verbose=True)
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        使用CAE编码音频到潜在空间
        
        Args:
            audio: (B, S, T) - 多stem音频波形
            
        Returns:
            latents: (B, S, C_lat, L) - CAE潜在表示
        """
        # 确保输入是3D: (B, S, T)
        if audio.dim() == 2:
            # (B, T) -> (B, 1, T)
            audio = audio.unsqueeze(1)
        
        batch_size, num_stems, audio_length = audio.shape
        
        # 检查stems数量
        if num_stems != self.num_stems:
            print(f"Warning: Expected {self.num_stems} stems, got {num_stems}. Using available stems.")
        
        # 分别编码每个stem
        latents_list = []
        for b in range(batch_size):
            batch_latents = []
            for s in range(num_stems):
                try:
                    # 获取单个样本的单个stem: [T]
                    stem_audio = audio[b, s].cpu().numpy()  # [T]
                    
                    # 转换为CAE期望的格式: [1, T] (channels, samples)
                    stem_audio = stem_audio.reshape(1, -1)  # [1, T]
                    
                    # CAE编码: [1, T] -> [1, 64, L]
                    # 使用更安全的参数设置
                    stem_latents = self.autoencoder.encode(
                        stem_audio, 
                        max_waveform_length=32768,  # 限制最大长度以避免内存问题
                        max_batch_size=1            # 单个样本处理
                    )
                    if isinstance(stem_latents, np.ndarray):
                        stem_latents = torch.from_numpy(stem_latents).to(audio.device)
                    stem_latents = stem_latents.to(dtype=torch.float32)
                    
                    # 确保维度正确 [C, L] 格式
                    if stem_latents.dim() == 3 and stem_latents.size(0) == 1:
                        stem_latents = stem_latents.squeeze(0)  # [1, 64, L] -> [64, L]
                    elif stem_latents.dim() == 2:
                        pass  # 已经是 [64, L] 格式
                    else:
                        raise ValueError(f"Unexpected CAE output shape: {stem_latents.shape}")
                    
                    batch_latents.append(stem_latents)
                    
                except Exception as e:
                    print(f"Error encoding stem {s} of batch {b}: {e}")
                    raise e
            
            # 堆叠当前batch的所有stems: [S, 64, L]
            batch_latents = torch.stack(batch_latents, dim=0)
            latents_list.append(batch_latents)
        
        # 堆叠所有batch: [B, S, 64, L]
        latents = torch.stack(latents_list, dim=0)
        
        return latents
    
    def decode_audio(self, latents: torch.Tensor) -> torch.Tensor:
        """
        使用CAE解码潜在表示到音频
        
        Args:
            latents: (B, S, C_lat, L) - CAE潜在表示
            
        Returns:
            audio: (B, S, T) - 音频波形
        """
        batch_size, num_channels, latent_channels, latent_length = latents.shape
        
        # 分别解码每个通道
        audio_list = []
        for i in range(num_channels):
            channel_latents = latents[:, i]  # (B, C_lat, L)
            channel_audio = self.autoencoder.decode(channel_latents)  # (B, T)
            audio_list.append(channel_audio.unsqueeze(1))  # (B, 1, T)
        
        # 合并所有通道
        audio = torch.cat(audio_list, dim=1)  # (B, S, T)
        
        return audio
    
    def prepare_diffusion_input(self, latents: torch.Tensor) -> torch.Tensor:
        """
        将CAE潜在表示转换为扩散模型输入格式
        
        Args:
            latents: (B, S, C_lat, L) - CAE潜在表示
            
        Returns:
            diffusion_input: (B, S*C_lat, H, W) - 扩散模型输入
        """
        B, S, C, L = latents.shape
        
        # 重塑为2D输入：需要确保H和W维度都大于1
        # 将(B, S, C, L)转换为方形的2D表示
        total_channels = S * C
        
        # 将序列长度转换为接近正方形的H×W
        import math
        side_length = int(math.sqrt(L))
        if side_length * side_length < L:
            side_length += 1
        
        # Pad到正方形
        padded_length = side_length * side_length
        padding = padded_length - L
        
        # (B, S, C, L) -> (B, S*C, L) -> pad -> (B, S*C, H, W)
        flat_latents = latents.view(B, total_channels, L)
        if padding > 0:
            flat_latents = torch.nn.functional.pad(flat_latents, (0, padding))
        
        diffusion_input = flat_latents.view(B, total_channels, side_length, side_length)
        
        # 确保数据类型一致
        print(f"Diffusion input shape: {diffusion_input.shape}, dtype: {diffusion_input.dtype}")
        
        return diffusion_input
    
    def restore_diffusion_output(self, diffusion_output: torch.Tensor, target_shape: torch.Size) -> torch.Tensor:
        """
        将扩散模型输出转换回CAE潜在表示格式
        
        Args:
            diffusion_output: (B, S*C_lat, H, W) - 扩散模型输出
            target_shape: (B, S, C_lat, L) - 目标形状
            
        Returns:
            latents: (B, S, C_lat, L) - CAE潜在表示
        """
        B, S, C, L = target_shape
        
        # 检查维度匹配
        if diffusion_output.dim() == 4:
            diffusion_output = diffusion_output.squeeze(2)  # 移除H维度
        
        # 重塑回原始形状
        latents = diffusion_output.view(B, S, C, L)
        
        return latents
    
    def forward(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        前向传播 - 计算扩散损失
        
        Args:
            x: (B, S, C_lat, L) - CAE潜在表示
            
        Returns:
            loss: 扩散损失
        """
        # 准备扩散输入
        diffusion_input = self.prepare_diffusion_input(x)
        
        # 计算扩散损失
        loss = self.audio_diffusion(diffusion_input, **kwargs)
        
        return loss
    
    def sample(
        self, 
        shape: Tuple[int, ...], 
        num_steps: Optional[int] = None,
        **kwargs
    ) -> torch.Tensor:
        """
        采样生成新的音频
        
        Args:
            shape: (B, S, C_lat, L) - 生成形状
            num_steps: 采样步数
            
        Returns:
            generated_audio: (B, S, T) - 生成的音频
        """
        if num_steps is None:
            num_steps = self.sampling_steps
        
        B, S, C, L = shape
        
        # 创建噪声输入
        noise_shape = (B, S * C, 1, L)
        noise = torch.randn(noise_shape, device=self.device)
        
        # 使用扩散模型采样
        generated_latents = self.audio_diffusion.sample(
            noise=noise,
            num_steps=num_steps,
            sigma_schedule=self.sigma_schedule,
            sampler=self.sampler,
            **kwargs
        )
        
        # 恢复潜在表示形状
        generated_latents = self.restore_diffusion_output(generated_latents, shape)
        
        # 解码为音频
        generated_audio = self.decode_audio(generated_latents)
        
        return generated_audio
    
    def training_step(self, batch, batch_idx):
        """训练步骤"""
        try:
            # 获取音频数据 - 优先获取stems数据
            if isinstance(batch, dict):
                audio = batch.get('waveform_stems', batch.get('waveform', batch.get('audio', None)))
            else:
                audio = batch
            
            if audio is None:
                raise ValueError("无法从batch中获取音频数据")
            
            # 编码音频
            latents = self.encode_audio(audio)
            
            # 计算扩散损失
            loss = self.forward(latents)
            
            # 记录损失
            self.log("train/loss", loss, prog_bar=True, on_step=True, on_epoch=True)
            
            return loss
            
        except Exception as e:
            print(f"❌ Error in training_step: {e}")
            raise e
    
    def validation_step(self, batch, batch_idx):
        """验证步骤"""
        try:
            # 获取音频数据 - 优先获取stems数据
            if isinstance(batch, dict):
                audio = batch.get('waveform_stems', batch.get('waveform', batch.get('audio', None)))
            else:
                audio = batch
            
            if audio is None:
                raise ValueError("无法从batch中获取音频数据")
            
            # 编码音频
            latents = self.encode_audio(audio)
            
            # 计算扩散损失 (forward函数内部会调用prepare_diffusion_input)
            val_loss = self.forward(latents)
            
            # 记录损失
            self.log("val/loss", val_loss, prog_bar=True, on_step=False, on_epoch=True)
            
            return val_loss
            
        except Exception as e:
            print(f"❌ Error in validation_step: {e}")
            raise e
    
    def configure_optimizers(self):
        """配置优化器"""
        optimizer = torch.optim.AdamW(
            self.parameters(), 
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            weight_decay=0.01
        )

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


# 兼容性函数
def create_model_from_config(config: Dict[str, Any]) -> MyModelScoreDiffusion:
    return MyModelScoreDiffusion(**config)

