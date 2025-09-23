# %% [markdown]
# # CD4MT

# %% [markdown]
# ## 1. env

# %%
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %%
# %matplotlib inline
import os, sys, yaml, torch, numpy as np, matplotlib.pyplot as plt
import soundfile as sf
from pathlib import Path
from IPython.display import Audio, display, HTML, Markdown
import matplotlib.font_manager as fm

# 设置项目根路径
ROOT = "/data1/yuchen/cd4mt"
sys.path.append("/data1/yuchen/MusicLDM-Ext/src")
sys.path.append(ROOT)
sys.path.append(f"{ROOT}/ldm")
os.chdir(ROOT)

print(f"Working directory: {os.getcwd()}")
from src.music2latent.music2latent.inference import EncoderDecoder

from ldm.models.diffusion.cd4mt_diffusion import ScoreDiffusionModel
from ldm.data.multitrack_datamodule import DataModuleFromConfig

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# %%


# %% [markdown]
# ## 2. config and load_data

# %%
CFG_PATH = "configs/cd4mt_small.yaml"

with open(CFG_PATH, 'r') as f:
    cfg = yaml.safe_load(f)

print(f"📄 配置文件: {CFG_PATH}")
print(f"🎯 项目名称: {cfg['project_name']}")
print(f"🎵 音轨类型: {cfg['data']['params']['path']['stems']}")
print(f"📊 批大小: {cfg['data']['params']['batch_size']}")
print(f"🔊 采样率: {cfg['data']['params']['preprocessing']['audio']['sampling_rate']}Hz")

display(Markdown(f"""
### 🔧 关键配置参数
| 参数 | 值 | 说明 |
|------|----|----- |
| 音轨数量 | {cfg['model']['params']['num_stems']} | bass, drums, guitar, piano |
| CAE潜在维度 | {cfg['model']['params']['cae_latent_dim']} | CAE编码器输出通道数 |
| UNet通道数 | {cfg['model']['params']['unet_']['params']['model_channels']} | 扩散模型基础通道数 |
| 采样步数 | {cfg['model']['params']['sampling_steps']} | 扩散采样步数 |
| 学习率 | {cfg['model']['params']['base_learning_rate']} | 训练学习率 |
"""))

# %%
dm = DataModuleFromConfig(**cfg["data"]["params"])
dm.prepare_data()
dm.setup(stage="fit")

train_loader = dm.train_dataloader()
print(f"train_loader: {len(train_loader)}")
batch = next(iter(train_loader))
print(f"batch.keys(): {list(batch.keys())}")

wav_stems = batch["waveform_stems"]  # (B, S, T)
wav_mix = batch.get("waveform", None)  # (B, T)

print(f"wav_stems: {wav_stems.shape}")
if wav_mix is not None:
    print(f"wav_mix: {wav_mix.shape}")

B, S, T = wav_stems.shape
print(f"Batch={B}, Stems={S}, Time={T}")

# %% [markdown]
# ## 3. CAE 音频编码器测试

# %%


# %%
ae = EncoderDecoder()
print(f"\n stem_num {S} ")

stem_names = cfg['model']['params']['stem_names']
latents_list = []
encode_shapes = []

for s in range(S):
    stem_name = stem_names[s] if s < len(stem_names) else f"stem_{s}"
    print(f"\n {stem_name} stem {s}:")
    stem_audio = wav_stems[:, s].cpu().numpy()  
    print(f"stem_audio : {stem_audio.shape}")

    try:
        stem_latents = ae.encode(stem_audio)
        if isinstance(stem_latents, np.ndarray):
            stem_latents = torch.from_numpy(stem_latents)
        
        print(f"  stem_latents.shape {stem_latents.shape}")
        print(f"  tem_latents.dtype {stem_latents.dtype}")
        print(f"  stem_latents.min(), max() :.3f [{stem_latents.min():.3f}, {stem_latents.max():.3f}]")        
        latents_list.append(stem_latents)
        encode_shapes.append(stem_latents.shape)
        
    except Exception as e:
        print(f" error : {e}")

latents_stacked = torch.stack(latents_list, dim=1)  # (B, S, C, L)
print(f"latents_stacked.shape: {latents_stacked.shape}")
print(f" Batch={latents_stacked.shape[0]}, Stems={latents_stacked.shape[1]}, Channels={latents_stacked.shape[2]}, Length={latents_stacked.shape[3]}")
latents = latents_stacked.to(device)
print(f" latents on : {device}")

# %%
recst_list = []
for s in range(S):
    stem_name = stem_names[s] if s < len(stem_names) else f"stem_{s}"
    print(f"\nDecode {stem_name}")
    stem_latents = latents[:, s].cpu().numpy()  # (B, C, L)
    
    try:
        recst = ae.decode(stem_latents)
        print(f"recst.shape: {recst.shape}")
        print(f"range: [{recst.min():.3f}, {recst.max():.3f}]")

        if isinstance(recst, torch.Tensor):
            recst = recst.cpu().numpy() 
        current_length = recst.shape[-1]
        if current_length > T:
            excess = current_length - T
            start_trim = excess // 2
            end_trim = excess - start_trim
            recst = recst[..., start_trim:current_length-end_trim]
            
        elif current_length < T:
            deficit = T - current_length
            pad_left = deficit // 2
            pad_right = deficit - pad_left
            recst = np.pad(recst, ((0,0), (pad_left, pad_right)), mode='constant', constant_values=0)
        
        recst_list.append(recst)
        
    except Exception as e:
        raise e

recst_aud = np.stack(recst_list, axis=1)  # (B, S, T')
recst_tensor = torch.from_numpy(recst_aud).to(device)

print(f"\nrecst shape: {recst_aud.shape}")
print(f"original lenght: {T}, recst length: {recst_aud.shape[2]}")

if recst_aud.shape[2] == T:
    mse_error = np.mean((wav_stems.cpu().numpy() - recst_aud)**2)
    print(f"MSE: {mse_error:.6f}")
else:
    print("length error")

# %% [markdown]
# ## 4. CD4MT 扩散模型初始化

# %%
with open(CFG_PATH, 'r') as f:
    cfg_fresh = yaml.safe_load(f)

model_config = cfg_fresh['model']['params'].copy()
try:
    unet_config = model_config.pop('unet_')
    print(f"\nUNet配置: {unet_config}")
    model = ScoreDiffusionModel(
        unet_config=unet_config,
        **model_config  
    )
    model = model.to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"模型参数统计:")
    print(f"   总参数: {total_params:,}")
    print(f"   可训练参数: {trainable_params:,}")
    print(f"   模型大小: ~{total_params * 4 / 1024 / 1024:.1f} MB (FP32)")
    
    # 显示模型配置
    print(f"\n模型配置:")
    print(f"   音轨数量: {model.num_stems}")
    print(f"   音轨名称: {model.stem_names}")
    print(f"   CAE潜在维度: {model.cae_latent_dim}")
    print(f"   采样率: {model.sample_rate}Hz")
    print(f"   采样步数: {model.sampling_steps}")
    
except Exception as e:
    raise e

# %%
if model is not None:
    print("🔄 测试模型编码功能...")
    
    with torch.no_grad():
        try:
            model_latents = model.encode_audio(wav_stems.to(device))
            
            print(f"✅ 模型编码成功!")
            print(f"📊 编码结果形状: {model_latents.shape}")
            print(f"🔍 与直接CAE编码对比:")
            print(f"   直接CAE: {latents.shape}")
            print(f"   模型编码: {model_latents.shape}")
            
            if latents.shape == model_latents.shape:
                diff = torch.mean(torch.abs(latents - model_latents)).item()
                rel_diff = diff / (torch.mean(torch.abs(latents)).item() + 1e-8)
                
                print(f"📈 编码差异分析:")
                print(f"   绝对差异: {diff:.6f}")
                print(f"   相对差异: {rel_diff*100:.2f}%")
                print(f"   直接CAE范围: [{latents.min():.3f}, {latents.max():.3f}]")
                print(f"   模型编码范围: [{model_latents.min():.3f}, {model_latents.max():.3f}]")
                
                if diff < 1e-6:
                    print("✅ 编码结果一致!")
                elif rel_diff < 0.1:
                    print("✅ 编码结果基本一致 (可接受的小差异)")
                else:
                    print("⚠️  编码结果存在显著差异")
                    
                print(f"📋 数据类型检查:")
                print(f"   直接CAE: {latents.dtype}, {latents.device}")
                print(f"   模型编码: {model_latents.dtype}, {model_latents.device}")
            
        except Exception as e:
            print(f"❌ 模型编码测试失败: {e}")
            model_latents = latents
else:
    print("⚠️  跳过模型编码测试（模型未成功创建）")
    model_latents = latents

# %% [markdown]
# ## 5. 扩散模型数据准备

# %%
if model is not None:
    try:
        diffusion_input = model.prepare_diffusion_input(model_latents)

        print(f" 输入形状变化:")
        print(f"   CAE潜在: {model_latents.shape} → 扩散输入: {diffusion_input.shape}")
        print(f"详细分析:")
        print(f"   原始: (B={model_latents.shape[0]}, S={model_latents.shape[1]}, C={model_latents.shape[2]}, L={model_latents.shape[3]})")
        print(f"   扩散: (B={diffusion_input.shape[0]}, SC={diffusion_input.shape[1]}, H={diffusion_input.shape[2]}, W={diffusion_input.shape[3]})")
        print(f"   通道合并: {model_latents.shape[1]}×{model_latents.shape[2]} = {diffusion_input.shape[1]}")

        print(f"数值统计:")
        print(f"   均值: {diffusion_input.mean().item():.6f}")
        print(f"   标准差: {diffusion_input.std().item():.6f}")
        print(f"   范围: [{diffusion_input.min().item():.3f}, {diffusion_input.max().item():.3f}]")
        
    except Exception as e:
        raise e
else:
    print("no model")

# %% [markdown]
# ## 6. 音频生成测试

# %%
if model is not None:
    try:
        with torch.no_grad():
            gen_batch_size = 1
            gen_stems = 4
            gen_channels = 64
            gen_length = 127
            gen_steps = 10 
            gen_shape = (gen_batch_size, gen_stems, gen_channels, gen_length)
            
            print(f"生成参数:")
            print(f"   gen_shape: {gen_shape}, gen_steps: {gen_steps}")
            
            print(f"\nGen aud ing")
            gen_aud = model.sample(
                shape=gen_shape,
                num_steps=gen_steps
            )
            print(f"gen aud shape: {gen_aud.shape}, (Batch={gen_aud.shape[0]}, Stems={gen_aud.shape[1]}, Time={gen_aud.shape[2]})")
            
            print(f"\n gen aud:")
            for s in range(gen_aud.shape[1]):
                stem_name = stem_names[s] if s < len(stem_names) else f"stem_{s}"
                stem_aud = gen_aud[0, s]
                print(f"   {stem_name}: 均值={stem_aud.mean().item():.6f}, 标准差={stem_aud.std().item():.6f}, 范围=[{stem_aud.min().item():.3f}, {stem_aud.max().item():.3f}]")
            
            gen_mix = gen_aud.sum(dim=1)  # (B, T)
            print(f"混合aud形状: {gen_mix.shape}")
            
    except Exception as e:
        raise e
else:
    print("no model")

# %% [markdown]
# ## 7. 音频可视化与对比分析

# %%
vis_batch_idx = 0 # bass, frums, guitar, piano
vis_stem_idx = 1  
vis_length = 44100 * 3  
sample_rate = cfg['data']['params']['preprocessing']['audio']['sampling_rate']

orig_aud = wav_stems[vis_batch_idx, vis_stem_idx].cpu().numpy()[:vis_length]

if recst_aud.shape[2] >= vis_length:
    recst_stem = recst_aud[vis_batch_idx, vis_stem_idx, :vis_length]
else:
    recst_stem = np.pad(recst_aud[vis_batch_idx, vis_stem_idx], 
                               (0, max(0, vis_length - recst_aud.shape[2])), 'constant')
    recst_stem = recst_stem[:vis_length]

if gen_aud.shape[2] >= vis_length:
    gen_stem = gen_aud[0, vis_stem_idx].cpu().numpy()[:vis_length]
else:
    gen_stem = np.pad(gen_aud[0, vis_stem_idx].cpu().numpy(), 
                           (0, max(0, vis_length - gen_aud.shape[2])), 
                           'constant')
    gen_stem = gen_stem[:vis_length]

print(f"vis param:")
print(f"   track: {stem_names[vis_stem_idx]}")
print(f"   dur: {vis_length/sample_rate:.1f}s")
print(f"   sr: {sample_rate}Hz")

time_axis = np.linspace(0, vis_length/sample_rate, vis_length)

# %%
print(time_axis.shape)
print(orig_aud.shape)
print(recst_stem.shape)

# %%
plt.figure(figsize=(15, 10))

plt.subplot(3, 1, 1)
plt.plot(time_axis, orig_aud, color='blue', alpha=0.8, linewidth=0.5)
plt.title(f'Original - {stem_names[vis_stem_idx]} Track', fontsize=14, fontweight='bold')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.xlim(0, vis_length/sample_rate)

plt.subplot(3, 1, 2)
plt.plot(time_axis, recst_stem, color='green', alpha=0.8, linewidth=0.5)
plt.title(f'CAE Recst - {stem_names[vis_stem_idx]} Track', fontsize=14, fontweight='bold')
plt.ylabel('Amplitude')
plt.grid(True, alpha=0.3)
plt.xlim(0, vis_length/sample_rate)

plt.subplot(3, 1, 3)
plt.plot(time_axis, gen_stem, color='red', alpha=0.8, linewidth=0.5)
plt.title(f'CD4MT Gen - {stem_names[vis_stem_idx]} Track', fontsize=14, fontweight='bold')
plt.ylabel('Amplitude')
plt.xlabel('Time (seconds)')
plt.grid(True, alpha=0.3)
plt.xlim(0, vis_length/sample_rate)

plt.tight_layout()
plt.suptitle('CD4MT Aud Pipeline Comparison', fontsize=16, fontweight='bold', y=1.02)
plt.savefig(f'fig/wave_comp_{stem_names[vis_stem_idx]}.png',
            dpi=300, bbox_inches='tight', facecolor='white')
plt.show()

print("aud stats:")
print(f"{'type':<12} | {'mean':<10} | {'std':<10} | {'min':<10} | {'max':<10}")
print("-" * 65)
print(f"{'orig':<12} | {orig_aud.mean():<10.6f} | {orig_aud.std():<10.6f} | {orig_aud.min():<10.6f} | {orig_aud.max():<10.6f}")
print(f"{'recst':<12} | {recst_stem.mean():<10.6f} | {recst_stem.std():<10.6f} | {recst_stem.min():<10.6f} | {recst_stem.max():<10.6f}")
print(f"{'gen':<12} | {gen_stem.mean():<10.6f} | {gen_stem.std():<10.6f} | {gen_stem.min():<10.6f} | {gen_stem.max():<10.6f}")

# %%
from scipy import signal

plt.figure(figsize=(15, 12))

def compute_spectrogram(aud, sr, title):
    f, t, Sxx = signal.spectrogram(aud, sr, nperseg=1024, noverlap=512)
    return f, t, 10 * np.log10(Sxx + 1e-10)

plt.subplot(3, 1, 1)
f_orig, t_orig, Sxx_orig = compute_spectrogram(orig_aud, sample_rate, 'Original')
plt.pcolormesh(t_orig, f_orig[:200], Sxx_orig[:200], shading='gouraud', cmap='viridis')
plt.title(f'Orig Spec - {stem_names[vis_stem_idx]}', fontweight='bold')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power (dB)')

plt.subplot(3, 1, 2)
f_recon, t_recon, Sxx_recon = compute_spectrogram(recst_stem, sample_rate, 'Reconstructed')
plt.pcolormesh(t_recon, f_recon[:200], Sxx_recon[:200], shading='gouraud', cmap='viridis')
plt.title(f'CAE Recst Spect - {stem_names[vis_stem_idx]}', fontweight='bold')
plt.ylabel('Frequency (Hz)')
plt.colorbar(label='Power (dB)')

plt.subplot(3, 1, 3)
f_gen, t_gen, Sxx_gen = compute_spectrogram(gen_stem, sample_rate, 'Generated')
plt.pcolormesh(t_gen, f_gen[:200], Sxx_gen[:200], shading='gouraud', cmap='viridis')
plt.title(f'CD4MT Gen Spect - {stem_names[vis_stem_idx]}', fontweight='bold')
plt.ylabel('Frequency (Hz)')
plt.xlabel('Time (seconds)')
plt.colorbar(label='Power (dB)')

plt.tight_layout()
plt.suptitle('Frequency Domain Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.show()
plt.savefig(f'fig/mel_comp_{stem_names[vis_stem_idx]}.png',
            dpi=300, bbox_inches='tight', facecolor='white')


# %% [markdown]
# ## 8. 多音轨对比分析

# %%
plt.figure(figsize=(16, 12))

vis_duration = 2.0
vis_samples = int(vis_duration * sample_rate)
time_axis_short = np.linspace(0, vis_duration, vis_samples)

for s in range(S):
    stem_name = stem_names[s] if s < len(stem_names) else f"Stem {s}"

    plt.subplot(S, 3, s*3 + 1)
    orig_short = wav_stems[vis_batch_idx, s].cpu().numpy()[:vis_samples]
    plt.plot(time_axis_short, orig_short, color='blue', alpha=0.8, linewidth=0.5)
    plt.title(f'{stem_name} - Original')
    plt.ylabel('Amplitude')
    if s == 0:
        plt.text(0.5, 1.1, 'Original Audio', transform=plt.gca().transAxes, 
                ha='center', fontweight='bold', fontsize=12)
    plt.subplot(S, 3, s*3 + 2)
    if s < recst_aud.shape[1]:
        recon_short = recst_aud[vis_batch_idx, s, :vis_samples]
        if len(recon_short) < vis_samples:
            recon_short = np.pad(recon_short, (0, vis_samples - len(recon_short)), 'constant')
        plt.plot(time_axis_short, recon_short, color='green', alpha=0.8, linewidth=0.5)
    plt.title(f'{stem_name} - Reconstructed')
    if s == 0:
        plt.text(0.5, 1.1, 'CAE Reconstructed', transform=plt.gca().transAxes, 
                ha='center', fontweight='bold', fontsize=12)

    plt.subplot(S, 3, s*3 + 3)
    if s < gen_aud.shape[1]:
        gen_short = gen_aud[0, s].cpu().numpy()[:vis_samples]
        if len(gen_short) < vis_samples:
            gen_short = np.pad(gen_short, (0, vis_samples - len(gen_short)), 'constant')
        plt.plot(time_axis_short, gen_short, color='red', alpha=0.8, linewidth=0.5)
    plt.title(f'{stem_name} - Generated')
    if s == 0:
        plt.text(0.5, 1.1, 'CD4MT Generated', transform=plt.gca().transAxes, 
                ha='center', fontweight='bold', fontsize=12)
    
    if s == S-1: 
        plt.xlabel('Time (seconds)')

plt.tight_layout()
plt.suptitle('Multi-Track Comparison: All Instruments', fontsize=16, fontweight='bold', y=1.02)
plt.show()

# %% [markdown]
# ## 9. 交互式音频播放

# %%
# 音频播放功能
print("🎵 准备音频播放...")

# 播放参数
play_duration = 5.0  # 播放5秒
play_samples = int(play_duration * sample_rate)

display(HTML("<h3>🎵 Audio Playback Comparison</h3>"))

# 播放原始音频的每个音轨
display(HTML("<h4>📻 Original Audio Tracks</h4>"))
for s in range(S):
    stem_name = stem_names[s] if s < len(stem_names) else f"Stem {s}"
    audio_data = wav_stems[vis_batch_idx, s].cpu().numpy()[:play_samples]
    
    display(HTML(f"<p><strong>{stem_name} (Original):</strong></p>"))
    display(Audio(audio_data, rate=sample_rate))

# 播放重建音频的每个音轨
display(HTML("<h4>🔄 CAE Reconstructed Tracks</h4>"))
for s in range(min(S, recst_aud.shape[1])):
    stem_name = stem_names[s] if s < len(stem_names) else f"Stem {s}"
    audio_data = recst_aud[vis_batch_idx, s, :play_samples]
    if len(audio_data) < play_samples:
        audio_data = np.pad(audio_data, (0, play_samples - len(audio_data)), 'constant')
    
    display(HTML(f"<p><strong>{stem_name} (Reconstructed):</strong></p>"))
    display(Audio(audio_data, rate=sample_rate))

# 播放生成音频的每个音轨
display(HTML("<h4>🎼 CD4MT Generated Tracks</h4>"))
for s in range(min(S, gen_aud.shape[1])):
    stem_name = stem_names[s] if s < len(stem_names) else f"Stem {s}"
    audio_data = gen_aud[0, s].cpu().numpy()[:play_samples]
    if len(audio_data) < play_samples:
        audio_data = np.pad(audio_data, (0, play_samples - len(audio_data)), 'constant')
    
    display(HTML(f"<p><strong>{stem_name} (Generated):</strong></p>"))
    display(Audio(audio_data, rate=sample_rate))

# %%
# 混合音频播放
display(HTML("<h4>🎵 Mixed Audio Comparison</h4>"))

# 原始混合
original_mix = wav_stems[vis_batch_idx].sum(dim=0).cpu().numpy()[:play_samples]
display(HTML("<p><strong>Original Mix (All Tracks):</strong></p>"))
display(Audio(original_mix, rate=sample_rate))

# 重建混合
reconstructed_mix = recst_aud[vis_batch_idx].sum(axis=0)[:play_samples]
if len(reconstructed_mix) < play_samples:
    reconstructed_mix = np.pad(reconstructed_mix, (0, play_samples - len(reconstructed_mix)), 'constant')
display(HTML("<p><strong>CAE Reconstructed Mix:</strong></p>"))
display(Audio(reconstructed_mix, rate=sample_rate))

# 生成混合
generated_mix_audio = gen_aud[0].sum(dim=0).cpu().numpy()[:play_samples]
if len(generated_mix_audio) < play_samples:
    generated_mix_audio = np.pad(generated_mix_audio, (0, play_samples - len(generated_mix_audio)), 'constant')
display(HTML("<p><strong>CD4MT Generated Mix:</strong></p>"))
display(Audio(generated_mix_audio, rate=sample_rate))

# %% [markdown]
# ## 10. 潜在空间分析

# %%
# 潜在空间可视化
print("🔍 潜在空间分析...")

plt.figure(figsize=(16, 10))

# 显示每个音轨的潜在表示
for s in range(S):
    stem_name = stem_names[s] if s < len(stem_names) else f"Stem {s}"
    
    plt.subplot(2, S, s + 1)
    latent_data = latents[vis_batch_idx, s].cpu().numpy()  # (C, L)
    plt.imshow(latent_data, aspect='auto', cmap='viridis', interpolation='nearest')
    plt.title(f'{stem_name}\nLatent Space')
    plt.ylabel('Channels (64)')
    if s == 0:
        plt.colorbar(label='Latent Value')
    
    # 潜在表示的统计分布
    plt.subplot(2, S, s + S + 1)
    latent_flat = latent_data.flatten()
    plt.hist(latent_flat, bins=50, alpha=0.7, color=f'C{s}', density=True)
    plt.title(f'{stem_name}\nValue Distribution')
    plt.xlabel('Latent Value')
    plt.ylabel('Density')
    
    # 添加统计信息
    mean_val = latent_flat.mean()
    std_val = latent_flat.std()
    plt.axvline(mean_val, color='red', linestyle='--', alpha=0.8, label=f'Mean: {mean_val:.3f}')
    plt.legend()

plt.tight_layout()
plt.suptitle('CAE Latent Space Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.show()

# 潜在空间统计摘要
print("\n📊 潜在空间统计摘要:")
print(f"{'音轨':<10} | {'均值':<10} | {'标准差':<10} | {'最小值':<10} | {'最大值':<10} | {'范围':<10}")
print("-" * 75)

for s in range(S):
    stem_name = stem_names[s] if s < len(stem_names) else f"Stem_{s}"
    latent_data = latents[vis_batch_idx, s].cpu().numpy().flatten()
    
    mean_val = latent_data.mean()
    std_val = latent_data.std()
    min_val = latent_data.min()
    max_val = latent_data.max()
    range_val = max_val - min_val
    
    print(f"{stem_name:<10} | {mean_val:<10.4f} | {std_val:<10.4f} | {min_val:<10.4f} | {max_val:<10.4f} | {range_val:<10.4f}")

# %% [markdown]
# ## 11. 扩散过程可视化

# %%
# 扩散输入格式分析
print("🔄 扩散过程可视化...")

plt.figure(figsize=(16, 8))

# 显示扩散输入的2D格式
plt.subplot(2, 3, 1)
diffusion_sample = diffusion_input[0, :64]  # 显示前64个通道
plt.imshow(diffusion_sample.cpu().numpy(), aspect='auto', cmap='RdBu_r')
plt.title('Diffusion Input\n(First 64 Channels)')
plt.ylabel('Channels')
plt.colorbar()

plt.subplot(2, 3, 2)
diffusion_sample = diffusion_input[0, 64:128]  # 显示第65-128个通道
plt.imshow(diffusion_sample.cpu().numpy(), aspect='auto', cmap='RdBu_r')
plt.title('Diffusion Input\n(Channels 65-128)')
plt.colorbar()

plt.subplot(2, 3, 3)
diffusion_sample = diffusion_input[0, 128:192]  # 显示第129-192个通道
plt.imshow(diffusion_sample.cpu().numpy(), aspect='auto', cmap='RdBu_r')
plt.title('Diffusion Input\n(Channels 129-192)')
plt.colorbar()

# 通道维度的分布分析
plt.subplot(2, 3, 4)
channel_means = diffusion_input[0].mean(dim=(1, 2)).cpu().numpy()
plt.plot(channel_means, 'o-', alpha=0.7)
plt.title('Channel-wise Mean Values')
plt.xlabel('Channel Index')
plt.ylabel('Mean Value')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 5)
channel_stds = diffusion_input[0].std(dim=(1, 2)).cpu().numpy()
plt.plot(channel_stds, 'o-', alpha=0.7, color='orange')
plt.title('Channel-wise Standard Deviation')
plt.xlabel('Channel Index')
plt.ylabel('Std Value')
plt.grid(True, alpha=0.3)

plt.subplot(2, 3, 6)
# 显示每个音轨对应的通道范围
for s in range(S):
    start_ch = s * 64
    end_ch = (s + 1) * 64
    stem_name = stem_names[s] if s < len(stem_names) else f"Stem {s}"
    
    stem_channels = diffusion_input[0, start_ch:end_ch]
    stem_mean = stem_channels.mean().item()
    stem_std = stem_channels.std().item()
    
    plt.bar(s, stem_mean, yerr=stem_std, capsize=5, alpha=0.7, label=stem_name)

plt.title('Per-Stem Statistics in Diffusion Input')
plt.xlabel('Stem Index')
plt.ylabel('Mean ± Std')
plt.xticks(range(S), [stem_names[s] if s < len(stem_names) else f"S{s}" for s in range(S)])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Diffusion Model Input Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.show()

# 打印扩散输入的详细信息
print(f"\n📊 扩散输入详细信息:")
print(f"   形状: {diffusion_input.shape}")
print(f"   数据类型: {diffusion_input.dtype}")
print(f"   设备: {diffusion_input.device}")
print(f"   内存占用: {diffusion_input.numel() * 4 / 1024 / 1024:.2f} MB")
print(f"   数值范围: [{diffusion_input.min().item():.6f}, {diffusion_input.max().item():.6f}]")
print(f"   均值: {diffusion_input.mean().item():.6f}")
print(f"   标准差: {diffusion_input.std().item():.6f}")

# %% [markdown]
# ## 12. 性能和质量分析

# %%
# 音频质量指标计算
print("📈 音频质量分析...")

def calculate_snr(original, reconstructed):
    """计算信噪比"""
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)

def calculate_correlation(x, y):
    """计算相关系数"""
    return np.corrcoef(x.flatten(), y.flatten())[0, 1]

# 计算各项指标
quality_metrics = []

for s in range(S):
    stem_name = stem_names[s] if s < len(stem_names) else f"Stem {s}"
    
    # 获取音频数据
    original = wav_stems[vis_batch_idx, s].cpu().numpy()
    
    # CAE重建质量
    if s < recst_aud.shape[1]:
        reconstructed = recst_aud[vis_batch_idx, s]
        min_len = min(len(original), len(reconstructed))
        
        orig_crop = original[:min_len]
        recon_crop = reconstructed[:min_len]
        
        # 计算指标
        mse = np.mean((orig_crop - recon_crop) ** 2)
        snr = calculate_snr(orig_crop, recon_crop)
        corr = calculate_correlation(orig_crop, recon_crop)
        
        quality_metrics.append({
            'stem': stem_name,
            'type': 'CAE Reconstruction',
            'mse': mse,
            'snr': snr,
            'correlation': corr
        })
    
    # 生成音频质量（与原始对比）
    if s < gen_aud.shape[1]:
        generated = gen_aud[0, s].cpu().numpy()
        min_len = min(len(original), len(generated))
        
        orig_crop = original[:min_len]
        gen_crop = generated[:min_len]
        
        # 计算指标
        mse = np.mean((orig_crop - gen_crop) ** 2)
        snr = calculate_snr(orig_crop, gen_crop)
        corr = calculate_correlation(orig_crop, gen_crop)
        
        quality_metrics.append({
            'stem': stem_name,
            'type': 'CD4MT Generation',
            'mse': mse,
            'snr': snr,
            'correlation': corr
        })

# 显示质量指标表格
print("\n📊 音频质量指标:")
print(f"{'音轨':<10} | {'类型':<18} | {'MSE':<12} | {'SNR (dB)':<10} | {'相关系数':<10}")
print("-" * 75)

for metric in quality_metrics:
    snr_str = f"{metric['snr']:.2f}" if not np.isinf(metric['snr']) else "∞"
    corr_str = f"{metric['correlation']:.4f}" if not np.isnan(metric['correlation']) else "N/A"
    print(f"{metric['stem']:<10} | {metric['type']:<18} | {metric['mse']:<12.6f} | {snr_str:<10} | {corr_str:<10}")

# %%
# 质量指标可视化
plt.figure(figsize=(15, 10))

# 准备数据
stems = []
mse_recon = []
mse_gen = []
snr_recon = []
snr_gen = []
corr_recon = []
corr_gen = []

for s in range(S):
    stem_name = stem_names[s] if s < len(stem_names) else f"Stem {s}"
    stems.append(stem_name)
    
    # 查找对应的指标
    recon_metrics = next((m for m in quality_metrics if m['stem'] == stem_name and 'Reconstruction' in m['type']), None)
    gen_metrics = next((m for m in quality_metrics if m['stem'] == stem_name and 'Generation' in m['type']), None)
    
    mse_recon.append(recon_metrics['mse'] if recon_metrics else 0)
    mse_gen.append(gen_metrics['mse'] if gen_metrics else 0)
    
    snr_recon.append(recon_metrics['snr'] if recon_metrics and not np.isinf(recon_metrics['snr']) else 0)
    snr_gen.append(gen_metrics['snr'] if gen_metrics and not np.isinf(gen_metrics['snr']) else 0)
    
    corr_recon.append(recon_metrics['correlation'] if recon_metrics and not np.isnan(recon_metrics['correlation']) else 0)
    corr_gen.append(gen_metrics['correlation'] if gen_metrics and not np.isnan(gen_metrics['correlation']) else 0)

x = np.arange(len(stems))
width = 0.35

# MSE对比
plt.subplot(2, 2, 1)
plt.bar(x - width/2, mse_recon, width, label='CAE Reconstruction', alpha=0.8)
plt.bar(x + width/2, mse_gen, width, label='CD4MT Generation', alpha=0.8)
plt.title('Mean Squared Error Comparison')
plt.xlabel('Stems')
plt.ylabel('MSE')
plt.xticks(x, stems)
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)

# SNR对比
plt.subplot(2, 2, 2)
plt.bar(x - width/2, snr_recon, width, label='CAE Reconstruction', alpha=0.8)
plt.bar(x + width/2, snr_gen, width, label='CD4MT Generation', alpha=0.8)
plt.title('Signal-to-Noise Ratio Comparison')
plt.xlabel('Stems')
plt.ylabel('SNR (dB)')
plt.xticks(x, stems)
plt.legend()
plt.grid(True, alpha=0.3)

# 相关系数对比
plt.subplot(2, 2, 3)
plt.bar(x - width/2, corr_recon, width, label='CAE Reconstruction', alpha=0.8)
plt.bar(x + width/2, corr_gen, width, label='CD4MT Generation', alpha=0.8)
plt.title('Correlation Coefficient Comparison')
plt.xlabel('Stems')
plt.ylabel('Correlation')
plt.xticks(x, stems)
plt.legend()
plt.ylim(-1, 1)
plt.grid(True, alpha=0.3)

# 综合质量评分（基于多个指标的加权平均）
plt.subplot(2, 2, 4)
# 标准化指标并计算综合评分
quality_scores_recon = []
quality_scores_gen = []

for i in range(len(stems)):
    # 重建质量评分 (SNR高好，MSE低好，相关系数高好)
    if mse_recon[i] > 0:
        score_recon = (snr_recon[i] + abs(corr_recon[i]) * 50) / (1 + np.log10(mse_recon[i] + 1e-10))
    else:
        score_recon = 0
    quality_scores_recon.append(max(0, score_recon))
    
    # 生成质量评分
    if mse_gen[i] > 0:
        score_gen = (snr_gen[i] + abs(corr_gen[i]) * 50) / (1 + np.log10(mse_gen[i] + 1e-10))
    else:
        score_gen = 0
    quality_scores_gen.append(max(0, score_gen))

plt.bar(x - width/2, quality_scores_recon, width, label='CAE Reconstruction', alpha=0.8)
plt.bar(x + width/2, quality_scores_gen, width, label='CD4MT Generation', alpha=0.8)
plt.title('Composite Quality Score')
plt.xlabel('Stems')
plt.ylabel('Quality Score')
plt.xticks(x, stems)
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.suptitle('Audio Quality Metrics Analysis', fontsize=16, fontweight='bold', y=1.02)
plt.show()

# %% [markdown]
# ## 13. 总结和结论

# %%
# 生成分析报告
print("📋 CD4MT 系统分析报告")
print("=" * 50)

print(f"\n🎯 模型配置:")
print(f"   - 音轨数量: {S} ({', '.join(stem_names[:S])})")
print(f"   - 采样率: {sample_rate} Hz")
print(f"   - CAE潜在维度: 64")
if model is not None:
    print(f"   - 模型参数: {sum(p.numel() for p in model.parameters()):,}")
    print(f"   - 采样步数: {model.sampling_steps}")

print(f"\n📊 数据处理流程:")
print(f"   1. 原始音频: {wav_stems.shape}")
print(f"   2. CAE编码: {latents.shape}")
print(f"   3. 扩散输入: {diffusion_input.shape}")
print(f"   4. 生成音频: {gen_aud.shape}")

print(f"\n🔍 质量评估:")
avg_mse_recon = np.mean([m for m in mse_recon if m > 0])
avg_mse_gen = np.mean([m for m in mse_gen if m > 0])
avg_corr_recon = np.mean([c for c in corr_recon if c != 0])
avg_corr_gen = np.mean([c for c in corr_gen if c != 0])

print(f"   - CAE重建MSE: {avg_mse_recon:.6f}")
print(f"   - CD4MT生成MSE: {avg_mse_gen:.6f}")
print(f"   - CAE重建相关性: {avg_corr_recon:.4f}")
print(f"   - CD4MT生成相关性: {avg_corr_gen:.4f}")

print(f"\n✅ 系统状态:")
print(f"   - CAE编码器: {'✅ 正常' if 'ae' in locals() else '❌ 未加载'}")
print(f"   - CD4MT模型: {'✅ 正常' if model is not None else '❌ 未加载'}")
print(f"   - 数据加载: {'✅ 正常' if 'batch' in locals() else '❌ 失败'}")
print(f"   - GPU加速: {'✅ 可用' if torch.cuda.is_available() else '❌ 不可用'}")

print(f"\n🎵 音频特征:")
for s in range(S):
    stem_name = stem_names[s] if s < len(stem_names) else f"Stem {s}"
    original_energy = np.sqrt(np.mean(wav_stems[vis_batch_idx, s].cpu().numpy() ** 2))
    generated_energy = np.sqrt(np.mean(gen_aud[0, s].cpu().numpy() ** 2)) if s < gen_aud.shape[1] else 0
    print(f"   - {stem_name}: 原始能量={original_energy:.6f}, 生成能量={generated_energy:.6f}")

print(f"\n🎼 潜在空间统计:")
print(f"   - 维度: {latents.shape}")
print(f"   - 数值范围: [{latents.min().item():.3f}, {latents.max().item():.3f}]")
print(f"   - 平均值: {latents.mean().item():.6f}")
print(f"   - 标准差: {latents.std().item():.6f}")

print(f"\n💡 使用建议:")
if avg_mse_recon < 0.01:
    print(f"   ✅ CAE重建质量良好")
else:
    print(f"   ⚠️  CAE重建质量有待改善")

if model is not None:
    print(f"   ✅ 可以进行音频生成实验")
    print(f"   💡 建议使用30-50采样步数获得更好质量")
else:
    print(f"   ⚠️  需要训练好的模型checkpoint")

print(f"\n📝 注意事项:")
print(f"   - 本演示使用的是{'真实' if 'dm' in locals() and hasattr(dm, 'train_dataloader') else '模拟'}数据")
print(f"   - 生成质量取决于模型训练程度")
print(f"   - 实际应用建议使用完整训练的模型")

print(f"\n" + "=" * 50)
print(f"CD4MT 可视化分析完成! 🎉")


