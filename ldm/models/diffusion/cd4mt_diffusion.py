"""cd4mt score diffusion model - multitrack audio generation with cae"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from typing import Dict, Any, Optional, Union, List, Tuple
from pathlib import Path
from einops import rearrange, reduce

# cae codec
from src.music2latent.music2latent import EncoderDecoder

# diffusion core
from ldm.models.diffusion.ctm.audio_diffusion_pytorch_.model import AudioDiffusionModel_2d
from ldm.models.diffusion.ctm.audio_diffusion_pytorch_.diffusion import (
    LogNormalDistribution,
    KarrasSchedule,
    ADPM2Sampler
)

# unet
from ldm.modules.modules.diffusionmodules.openaimodel import UNetModel

# utils
from ldm.modules.util import instantiate_from_config, count_params, summarize_params


class ScoreDiffusionModel(pl.LightningModule):
    """score diffusion with cae codec for multitrack gen"""
    
    def __init__(
        self,
        # unet config
        unet_config: Dict[str, Any],
        
        # cae config  
        cae_latent_dim: int = 64,
        cae_z_channels: int = 64,
        sample_rate: int = 44100,
        
        # diffusion params
        diffusion_sigma_distribution: Optional[Dict] = None,
        diffusion_sigma_data: float = 0.5,
        diffusion_dynamic_threshold: float = 0.0,
        lambda_perceptual: float = 0.0,
        
        # multitrack setup
        num_stems: int = 4,
        stem_names: List[str] = ["bass", "drums", "guitar", "piano"],
        support_mixture: bool = True,
        
        # training
        base_learning_rate: float = 1e-4,
        
        # sampling
        sampling_steps: int = 50,
        sigma_min: float = 0.0001,
        sigma_max: float = 3.0,
        rho: float = 7.0,
        
        # monitoring
        monitor: str = "val/loss",
        
        **kwargs
    ):
        super().__init__()
        self.save_hyperparameters()
        
        # =========================
        # 1. Audio Auto-Encoder (CAE)
        # =========================
        # Lazily construct CAE on the correct device when first used
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
        
        print(f"[Init] Creating UNet with in/out channels = {diffusion_in_channels}")
        print(f"        UNet config(model_channels={unet_config['params'].get('model_channels')}, "
              f"num_res_blocks={unet_config['params'].get('num_res_blocks')}, "
              f"channel_mult={unet_config['params'].get('channel_mult')}, "
              f"num_head_channels={unet_config['params'].get('num_head_channels')})")
        
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
        print(f"✅ ScoreDiffusionModel initialized:")
        print(f"   - CAE latent dim: {cae_latent_dim}")
        print(f"   - CAE z channels: {cae_z_channels}")
        print(f"   - Num stems: {num_stems}")
        print(f"   - Support mixture: {support_mixture}")
        print(f"   - Diffusion in channels: {diffusion_in_channels}")
        print(f"   - Learning rate: {self.learning_rate}")
        print(f"   - Sampling steps: {sampling_steps}")

        # 参数规模与内存占用（更详细）
        try:
            summarize_params(self.audio_diffusion, name="AudioDiffusionModel_2d")
            if hasattr(self.audio_diffusion, 'unet'):
                summarize_params(self.audio_diffusion.unet, name="UNet (inner)")
        except Exception as _e:
            # 回退到简单统计，避免因 dtype 等导致初始化报错
            count_params(self.audio_diffusion, verbose=True)

        # 形状/超参安全检查（提前发现常见断言）
        try:
            mc = int(unet_config['params'].get('model_channels'))
            cmult = list(unet_config['params'].get('channel_mult'))
            head_dim = int(unet_config['params'].get('num_head_channels', -1))
            if head_dim != -1:
                bad = [mc * int(m) for m in cmult if (mc * int(m)) % head_dim != 0]
                if bad:
                    print(f"[Warn] num_head_channels={head_dim} does not divide channels at levels: {bad}")
        except Exception:
            pass
    
    def encode_audio(self, audio: torch.Tensor) -> torch.Tensor:
        """
        使用CAE编码音频到潜在空间
        
        Args:
            audio: (B, S, T) - 多stem音频波形
            
        Returns:
            latents: (B, S, C_lat, L) - CAE潜在表示
        """
        # 确保已初始化 CAE 到正确设备
        if self.autoencoder is None:
            dev = getattr(self, 'device', None) or audio.device
            self.autoencoder = EncoderDecoder(device=dev)

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
                    
                    # 转换为torch tensor并确保数据类型一致
                    if isinstance(stem_latents, np.ndarray):
                        stem_latents = torch.from_numpy(stem_latents).to(audio.device)
                    
                    # 强制转换为float32以避免mixed precision问题
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
        # 确保已初始化 CAE 到正确设备
        if self.autoencoder is None:
            dev = getattr(self, 'device', None) or latents.device
            self.autoencoder = EncoderDecoder(device=dev)

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
        
        print(f"🔄 恢复扩散输出: {diffusion_output.shape} -> {target_shape}")
        
        # 正确处理4D到目标形状的转换
        if diffusion_output.dim() == 4:
            batch_size, total_channels, H, W = diffusion_output.shape
            
            # 验证通道数匹配
            expected_channels = S * C
            if total_channels != expected_channels:
                raise ValueError(f"通道数不匹配: 期望 {expected_channels}, 得到 {total_channels}")
            
            # 展平H和W维度: (B, S*C, H, W) -> (B, S*C, H*W)
            flat_output = diffusion_output.view(batch_size, total_channels, H * W)
            
            # 截取前L个元素: (B, S*C, H*W) -> (B, S*C, L)
            if H * W >= L:
                flat_output = flat_output[:, :, :L]
            else:
                # 如果生成的长度不够，用零填充
                padding = L - (H * W)
                flat_output = torch.nn.functional.pad(flat_output, (0, padding))
            
        elif diffusion_output.dim() == 3:
            # 已经是(B, S*C, L)格式
            flat_output = diffusion_output
        else:
            raise ValueError(f"不支持的扩散输出维度: {diffusion_output.dim()}")
        
        # 重塑为目标形状: (B, S*C, L) -> (B, S, C, L)
        latents = flat_output.view(B, S, C, L)
        
        print(f"✅ 恢复完成: {latents.shape}")
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
        
        # 创建噪声输入 - 必须与训练时prepare_diffusion_input的输出形状一致
        total_channels = S * C  # 256 for 4 stems * 64 channels
        
        # 计算与prepare_diffusion_input相同的2D维度
        import math
        side_length = int(math.sqrt(L))
        if side_length * side_length < L:
            side_length += 1
        
        # 噪声形状必须匹配训练时的(B, S*C, H, W)格式
        noise_shape = (B, total_channels, side_length, side_length)
        noise = torch.randn(noise_shape, device=self.device)
        
        print(f"✅ 噪声形状: {noise_shape} (匹配训练时prepare_diffusion_input输出)")
        
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
        
        # 学习率调度器
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
def create_model_from_config(config: Dict[str, Any]) -> ScoreDiffusionModel:
    """Create model from configuration"""
    return ScoreDiffusionModel(**config)


# Legacy aliases for backwards compatibility
MyModelScoreDiffusion = ScoreDiffusionModel
MyModelScoreDiffusionSampler = ScoreDiffusionModel
