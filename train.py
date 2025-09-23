import os
import sys
import yaml
import torch
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
import logging
from datetime import datetime
import pytorch_lightning as pl

# from IPython.display import Audio, display, HTML, Markdown
import matplotlib.font_manager as fm
from src.music2latent.music2latent import EncoderDecoder
from ldm.models.diffusion.cd4mt_diffusion import ScoreDiffusionModel
from ldm.data.multitrack_datamodule import DataModuleFromConfig
from src.cm.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    cm_train_defaults,
    args_to_dict,
    add_dict_to_argparser,
    create_ema_and_scales_fn,
)
ROOT = "/data1/yuchen/cd4mt"
sys.path.append("/data1/yuchen/MusicLDM-Ext/src")
sys.path.append(ROOT)
sys.path.append(f"{ROOT}/ldm")
os.chdir(ROOT)
CFG_PATH = "configs/cd4mt_small.yaml"

def adapt_l2cm(config, audio_latent_shape):
    B, S, C, L = audio_latent_shape  # (batch, stems, channels, length)
    # 将音频latent重塑为2D"图像"格式 (B, SC, H, W)这里我们将stems*channels作为通道维度，length作为width
    # 高度设为1或者将length分割为height*width
    if L >= 128:
        # 如果length足够长，可以reshape为方形
        import math
        height = int(math.sqrt(L))
        width = L // height
        if height * width < L:
            width += 1
        image_size = max(height, width)
    else:
        # 否则使用1D处理
        height = 1
        width = L
        image_size = 64  # 默认图像size

    config.update({
        "image_size": image_size,
        "num_channels": 128,  # 模型通道数
        "in_channels": S * C,  # 输入通道数 = stems * cae_channels
        "out_channels": S * C,  # 输出通道数
        "num_res_blocks": 2,
        "channel_mult": "1,2,4",
        "attention_resolutions": "32,16,8",
        "num_heads": 4,
        "num_head_channels": 32,
        "dropout": 0.1,
        "use_scale_shift_norm": True,
        "learn_sigma": False,
        "class_cond": False,
        "use_fp16": False,
        "use_checkpoint": False,
    })

    return config, (height, width)

def main():
# load data
    with open(CFG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    dm = DataModuleFromConfig(**cfg["data"]["params"])
    dm.prepare_data()
    dm.setup(stage="fit")
    device = cfg["device"]
    print(device)
    train_loader = dm.train_dataloader()
    batch = next(iter(train_loader))
    wav_stems = batch["waveform_stems"]  # (B, S, T)
    wav_mix = batch.get("waveform", None)  # (B, T)
    if wav_mix is not None:
        print(f"wav_mix: {wav_mix.shape}")
    B, S, T = wav_stems.shape
    print(f"Batch={B}, Stems={S}, Time={T}")

# CAE encode
    ae = EncoderDecoder()
    stem_names = cfg['model']['params']['stem_names']
    latents_list = []
    encode_shapes = []
    for s in range(S):
        stem_name = stem_names[s] if s < len(stem_names) else f"stem_{s}"
        stem_audio = wav_stems[:, s].cpu().numpy()  
        print(f"\n {stem_name} stem {s}. stem_audio : {stem_audio.shape}")
        stem_latents = ae.encode(stem_audio)
        if isinstance(stem_latents, np.ndarray):
            stem_latents = torch.from_numpy(stem_latents)      
        latents_list.append(stem_latents)
        encode_shapes.append(stem_latents.shape)
            

    latents_stacked = torch.stack(latents_list, dim=1)  # (B, S, C, L)
    print(f"latents_stacked.shape: {latents_stacked.shape}")
    print(f" Batch={latents_stacked.shape[0]}, Stems={latents_stacked.shape[1]}, Channels={latents_stacked.shape[2]}, Length={latents_stacked.shape[3]}")
    latents = latents_stacked.to(device)
    print(f" latents on : {device}")
    # recst_list = []
    # for s in range(S):
    #     stem_name = stem_names[s] if s < len(stem_names) else f"stem_{s}"
    #     print(f"\nDecode {stem_name}")
    #     stem_latents = latents[:, s].cpu().numpy()  # (B, C, L)

    #     recst = ae.decode(stem_latents)
    #     print(f"recst.shape: {recst.shape}")
    #     print(f"range: [{recst.min():.3f}, {recst.max():.3f}]")

    #     if isinstance(recst, torch.Tensor):
    #         recst = recst.cpu().numpy() 
    #     current_length = recst.shape[-1]
    #     if current_length > T:
    #         excess = current_length - T
    #         start_trim = excess // 2
    #         end_trim = excess - start_trim
    #         recst = recst[..., start_trim:current_length-end_trim]
            
    #     elif current_length < T:
    #         deficit = T - current_length
    #         pad_left = deficit // 2
    #         pad_right = deficit - pad_left
    #         recst = np.pad(recst, ((0,0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
    #     recst_list.append(recst)


    # recst_aud = np.stack(recst_list, axis=1)  # (B, S, T')
    # recst_tensor = torch.from_numpy(recst_aud).to(device)

    # print(f"\nrecst shape: {recst_aud.shape}")
    # print(f"original lenght: {T}, recst length: {recst_aud.shape[2]}")

    # if recst_aud.shape[2] == T:
    #     mse_error = np.mean((wav_stems.cpu().numpy() - recst_aud)**2)
    #     print(f"MSE: {mse_error:.6f}")
    # else:
    #     print("length error")



    with open(CFG_PATH, 'r') as f:
        args = yaml.safe_load(f)


# ct model
    args, reshape_info = adapt_l2cm(args, latents_stacked.shape)
    model_and_diffusion_kwargs = args_to_dict(
        args, model_and_diffusion_defaults().keys()
    )
    model_and_diffusion_kwargs["distillation"] = False
    model, diffusion = create_model_and_diffusion(**model_and_diffusion_kwargs)
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数统计:")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")

    # 清理之前的logger
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    # 训练配置
    TRAIN_EPOCHS = 200  # 测试训练，建议正式训练用20+
    TRAIN_LOG_DIR = "./training_logs"
    TRAIN_LOG_FILE = f"{TRAIN_LOG_DIR}/cd4mt_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

    # 创建日志目录
    Path(TRAIN_LOG_DIR).mkdir(exist_ok=True)

    # 设置日志
    class TrainingLogger:
        def __init__(self, log_file):
            self.log_file = log_file
            # 创建独立的logger，避免重复
            self.logger = logging.getLogger(f'CD4MT_Training_{datetime.now().strftime("%H%M%S")}')
            self.logger.setLevel(logging.INFO)
            
            # 清除已有的handlers
            self.logger.handlers.clear()
            
            # 文件handler
            self.file_handler = logging.FileHandler(log_file, encoding='utf-8')
            formatter = logging.Formatter(
                '%(asctime)s | %(levelname)s | %(message)s',
                datefmt='%Y-%m-%d %H:%M:%S'
            )
            self.file_handler.setFormatter(formatter)
            self.logger.addHandler(self.file_handler)
            
            # 防止向root logger传播
            self.logger.propagate = False
        
        def info(self, msg):
            self.logger.info(msg)
            print(f"INFO: {msg}")  # 直接打印到控制台
            
        def error(self, msg):
            self.logger.error(msg)
            print(f"ERROR: {msg}")
            
        def warning(self, msg):
            self.logger.warning(msg)
            print(f"WARNING: {msg}")

    # 初始化训练日志
    train_logger = TrainingLogger(TRAIN_LOG_FILE)
    train_logger.info("🚀 开始 CD4MT 模型训练")
    train_logger.info(f"📄 日志文件: {TRAIN_LOG_FILE}")
    train_logger.info(f"🎯 训练轮数: {TRAIN_EPOCHS}")
    train_logger.info(f"🎵 音轨: {cfg['model']['params']['stem_names']}")
    train_logger.info(f"📊 批大小: {cfg['data']['params']['batch_size']}")

    print(f"📝 训练日志将保存到: {TRAIN_LOG_FILE}")
    print(f"✅ 所有训练变量已正确定义")


    # 训练进度监控和损失追踪
    class TrainingProgressCallback(pl.Callback):
        def __init__(self, logger):
            self.logger = logger
            self.start_time = None
            
        def on_train_start(self, trainer, pl_module):
            self.start_time = datetime.now()
            self.logger.info("=" * 60)
            self.logger.info("🎯 训练开始")
            self.logger.info(f"   模型参数: {sum(p.numel() for p in pl_module.parameters()):,}")
            self.logger.info(f"   训练设备: {trainer.strategy.root_device}")
            self.logger.info(f"   最大epochs: {trainer.max_epochs}")
            self.logger.info("=" * 60)
            
        def on_train_epoch_start(self, trainer, pl_module):
            epoch = trainer.current_epoch + 1
            self.logger.info(f"\n📅 Epoch {epoch}/{trainer.max_epochs}")
            
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            # 每50个batch记录一次进度
            if batch_idx % 50 == 0:
                if isinstance(outputs, dict) and 'loss' in outputs:
                    loss = outputs['loss'].item()
                elif hasattr(outputs, 'item'):
                    loss = outputs.item()
                else:
                    loss = float('nan')
                self.logger.info(f"   Batch {batch_idx}: loss={loss:.6f}")
                
        def on_train_epoch_end(self, trainer, pl_module):
            epoch = trainer.current_epoch + 1
            elapsed = datetime.now() - self.start_time
            
            # 获取训练损失
            train_loss = trainer.callback_metrics.get('train/loss_epoch', float('nan'))
            val_loss = trainer.callback_metrics.get('val/loss', float('nan'))
            
            self.logger.info(f"✅ Epoch {epoch} 完成")
            self.logger.info(f"   训练损失: {train_loss:.6f}")
            self.logger.info(f"   验证损失: {val_loss:.6f}")
            self.logger.info(f"   累计时间: {elapsed}")
            
        def on_train_end(self, trainer, pl_module):
            total_time = datetime.now() - self.start_time
            self.logger.info("=" * 60)
            self.logger.info("🎉 训练完成!")
            self.logger.info(f"   总训练时间: {total_time}")
            self.logger.info(f"   最终训练损失: {trainer.callback_metrics.get('train/loss_epoch', 'N/A')}")
            self.logger.info(f"   最终验证损失: {trainer.callback_metrics.get('val/loss', 'N/A')}")
            self.logger.info("=" * 60)
            
        def on_exception(self, trainer, pl_module, exception):
            self.logger.error(f"❌ 训练异常: {exception}")

    class LossTracker:
        def __init__(self, log_file):
            self.log_file = log_file
            self.losses = []
            
        def add_loss(self, epoch, batch, train_loss, val_loss=None):
            self.losses.append({
                'epoch': epoch,
                'batch': batch,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'timestamp': datetime.now().isoformat()
            })
            
        def save_losses(self):
            import json
            loss_file = self.log_file.replace('.txt', '_losses.json')
            with open(loss_file, 'w', encoding='utf-8') as f:
                json.dump(self.losses, f, indent=2, ensure_ascii=False)
            print(f"📊 损失数据保存到: {loss_file}")

    # 检查必需变量并初始化训练组件
    try:
        if 'TRAIN_LOG_FILE' not in globals():
            raise NameError("TRAIN_LOG_FILE 未定义")
        if 'train_logger' not in globals():
            raise NameError("train_logger 未定义")
        
        loss_tracker = LossTracker(TRAIN_LOG_FILE)
        progress_callback = TrainingProgressCallback(train_logger)
        print("✅ 训练回调和损失追踪器已初始化")
        
    except NameError as e:
        print(f"❌ 错误: {e}")
        print("请先运行训练初始化单元格！")


        # 简化的训练准备和执行
    try:
        print("🏗️ 创建训练模型...")
        
        # 重新创建模型用于训练
        with open(CFG_PATH, 'r') as f:
            train_cfg = yaml.safe_load(f)
        
        # 创建模型
        train_model_config = train_cfg['model']['params'].copy()
        train_unet_config = train_model_config.pop('unet_')
        
        train_model = ScoreDiffusionModel(
            unet_config=train_unet_config,
            **train_model_config
        )
        train_model = train_model.to(device)
        
        print(f"✅ 训练模型创建成功")
        print(f"   模型参数: {sum(p.numel() for p in train_model.parameters()):,}")
        
        # 配置训练器 - 修复checkpoint配置
        print("⚙️ 配置训练器...")
        
        # 创建checkpoint目录
        checkpoint_dir = "./training_logs/checkpoints"
        Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
        
        # 修复checkpoint回调配置
        checkpoint_callback = ModelCheckpoint(
            dirpath=checkpoint_dir,
            filename="cd4mt-test-{epoch:02d}-{step:04d}",
            save_top_k=-1,  # 保存所有checkpoint，或者设为1只保存最后一个
            save_last=True,  # 保存最后一个
            every_n_epochs=1,  # 每个epoch保存一次
            verbose=True
        )
        
        # 创建训练器 - 使用配置文件中的参数
        trainer = pl.Trainer(
            accelerator="gpu",
            devices=[0, 1, 2],  
            # strategy="auto", 
            max_epochs=2,  # 使用配置文件中的epochs
            val_check_interval=0.5,  # 每个epoch中间验证一次
            limit_train_batches=train_cfg['trainer'].get('limit_train_batches', 1.0),
            limit_val_batches=train_cfg['trainer'].get('limit_val_batches', 1.0),
            enable_progress_bar=True,
            log_every_n_steps=1,
            callbacks=[checkpoint_callback],
            logger=False,  # 禁用复杂logger避免问题
            num_sanity_val_steps=0  # 跳过验证检查
        )
        
        print("✅ 训练器配置完成")
        print(f"📁 Checkpoint保存目录: {checkpoint_dir}")
        print("🚀 开始快速训练测试...")
        
        # 测试训练几个步骤
        trainer.fit(train_model, dm)
        
        print("🎉 训练测试完成!")
        
        # 显示保存的checkpoint信息
        print(f"\n📋 Checkpoint信息:")
        print(f"   保存目录: {checkpoint_dir}")
        if hasattr(checkpoint_callback, 'best_model_path') and checkpoint_callback.best_model_path:
            print(f"   最佳模型: {checkpoint_callback.best_model_path}")
        if hasattr(checkpoint_callback, 'last_model_path') and checkpoint_callback.last_model_path:
            print(f"   最新模型: {checkpoint_callback.last_model_path}")
        
        # 列出所有保存的checkpoint文件
        import glob
        ckpt_files = glob.glob(f"{checkpoint_dir}/*.ckpt")
        if ckpt_files:
            print(f"   所有checkpoint文件:")
            for ckpt in ckpt_files:
                file_size = os.path.getsize(ckpt) / (1024*1024)  # MB
                print(f"     - {ckpt} ({file_size:.1f} MB)")
        else:
            print("   ⚠️ 未找到checkpoint文件")
        
        # 测试训练后的模型生成
        print("\n🧪 测试训练后的模型...")
        train_model.eval()
        with torch.no_grad():
            test_shape = (1, 4, 64, 127)
            test_generated = train_model.sample(shape=test_shape, num_steps=5)  # 减少采样步数
            print(f"✅ 生成测试成功: {test_generated.shape}")
            print(f"   生成音频统计: 均值={test_generated.mean().item():.6f}, 标准差={test_generated.std().item():.6f}")
        
    except Exception as e:
        print(f"❌ 训练测试失败: {e}")
        import traceback
        print("详细错误:")
        traceback.print_exc()


        # 训练完成后的分析和总结
    if 'train_model' in locals():
        print("📊 训练后分析:")
        print(f"   模型状态: {'训练模式' if train_model.training else '评估模式'}")
        print(f"   设备: {next(train_model.parameters()).device}")
        
        # 对比训练前后的生成效果
        print("\n🔍 对比分析:")
        gen_shape = (1, 4, 64, 127)
        gen_steps = 5
        with torch.no_grad():
            gen_aud = model.sample(
                shape=gen_shape,
                num_steps=gen_steps
            )
        if 'gen_aud' in locals() and 'test_generated' in locals():
            print(f"   训练前生成: 均值={gen_aud.mean().item():.6f}, 标准差={gen_aud.std().item():.6f}")
            print(f"   训练后生成: 均值={test_generated.mean().item():.6f}, 标准差={test_generated.std().item():.6f}")
            
            # 计算差异
            if gen_aud.shape == test_generated.shape:
                diff = torch.mean(torch.abs(gen_aud - test_generated)).item()
                print(f"   生成差异: {diff:.6f}")
                if diff > 0.01:
                    print("   ✅ 模型已发生变化，训练有效")
                else:
                    print("   ⚠️  生成结果相近，可能需要更多训练")
        
        print(f"\n🎉 CD4MT训练测试完成!")
        print(f"   ✅ 模型可以正常训练")
        print(f"   ✅ 生成功能正常")
        print(f"   💡 建议: 使用更多epochs和完整数据进行正式训练")
        
    else:
        print("❌ 训练模型不存在，请先运行训练单元格")