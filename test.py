#!/usr/bin/env python3
"""
End-to-end CD4MT CT training script (from test.ipynb), with:
- Auto GPU selection (no hard-coded cuda:0)
- CAE encode per stem, reshape to (B,S*C,H,W)
- UNet (CT) with in/out conv resized to match S*C
- Karras consistency loss training + EMA target
- SwanLab logging
- Sanity check (single batch forward/backward) before long training
"""
import os
import sys
import time
import copy
import json
import math
import yaml
import numpy as np
import soundfile as sf
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

# Repo root and paths
ROOT = "/data1/yuchen/cd4mt"
os.environ.setdefault("NUMBA_CACHE_DIR", f"{ROOT}/.numba_cache")
os.environ.setdefault("NUMBA_DISABLE_JIT", "1")
os.makedirs(os.environ["NUMBA_CACHE_DIR"], exist_ok=True)
sys.path.append("/data1/yuchen/MusicLDM-Ext/src")
sys.path.append(ROOT)
sys.path.append(f"{ROOT}/ldm")
os.chdir(ROOT)

# Data & models
# Preload librosa.util and inject fallback for pad_center/tiny if missing (librosa 0.11 lazy-loader quirks)
try:
    import librosa
    import numpy as _np
    import torch as _th
    import librosa.util as _lutil
    if not hasattr(_lutil, 'pad_center'):
        def _pad_center(x, size, axis=-1, mode='constant', **kwargs):
            import numpy as np
            is_t = isinstance(x, _th.Tensor)
            length = x.shape[axis]
            if length >= size:
                return x
            pad = size - length
            left = pad // 2
            right = pad - left
            if is_t:
                pad_cfg = [(0,0)]*x.ndim
                pad_cfg[axis] = (left, right)
                # torch.nn.functional.pad expects reversed order for last dims; fallback to numpy then back
                xnp = x.detach().cpu().numpy()
                out = np.pad(xnp, pad_cfg, mode=mode, **kwargs)
                return _th.from_numpy(out).to(x.device)
            else:
                pad_cfg = [(0,0)]*x.ndim
                pad_cfg[axis] = (left, right)
                return np.pad(x, pad_cfg, mode=mode, **kwargs)
        _lutil.pad_center = _pad_center
    if not hasattr(_lutil, 'tiny'):
        def _tiny(x):
            try:
                dt = getattr(x, 'dtype', _np.float32)
                return _np.finfo(dt).tiny
            except Exception:
                return _np.finfo(_np.float32).tiny
        _lutil.tiny = _tiny
except Exception as _libe:
    print("[warn] librosa util preload failed:", _libe)
from src.music2latent.music2latent import EncoderDecoder
from ldm.data.multitrack_datamodule import DataModuleFromConfig
from src.cm.script_util import create_model
from src.cm.karras_diffusion import KarrasDenoiser, karras_sample
from src.cm.nn import update_ema

try:
    import swanlab
except Exception:
    swanlab = None


def auto_select_cuda(min_free_gb: float = 4.0) -> torch.device:
    """Pick the GPU with max free memory; fallback to CPU."""
    if not torch.cuda.is_available():
        print("CUDA not available -> cpu")
        return torch.device("cpu")
    best, best_free = None, -1.0
    for i in range(torch.cuda.device_count()):
        free_b, _ = torch.cuda.mem_get_info(i)
        free_gb = free_b / (1024**3)
        if free_gb > best_free and free_gb >= min_free_gb:
            best, best_free = i, free_gb
    if best is None:
        for i in range(torch.cuda.device_count()):
            free_b, _ = torch.cuda.mem_get_info(i)
            free_gb = free_b / (1024**3)
            if free_gb > best_free:
                best, best_free = i, free_gb
    print(f"Selected CUDA:{best} (free ~{best_free:.1f} GB)")
    return torch.device(f"cuda:{best}")


def encode_batch_to_imgs(ae: EncoderDecoder, batch: dict, to_float32: bool = True):
    """Encode (B,S,T) stems to CAE latents and reshape to (B, S*C, H, W)."""
    wav_stems = batch['waveform_stems']  # (B,S,T)
    if isinstance(wav_stems, np.ndarray):
        wav_stems = torch.from_numpy(wav_stems)
    assert wav_stems.dim() == 3, f"expected wav_stems (B,S,T), got {tuple(wav_stems.shape)}"
    B, S, T = wav_stems.shape
    latents_list = []
    for s in range(S):
        stem_audio = wav_stems[:, s].cpu().numpy()  # (B,T), B treated as audio_channels
        stem_lat = ae.encode(stem_audio)            # (B,C,L)
        if isinstance(stem_lat, np.ndarray):
            stem_lat = torch.from_numpy(stem_lat)
        if to_float32:
            stem_lat = stem_lat.to(torch.float32)
        latents_list.append(stem_lat)
    latents = torch.stack(latents_list, dim=1).contiguous()  # (B,S,C,L)

    B, S, C, L = latents.shape
    flat = latents.view(B, S * C, L)
    side = int(math.sqrt(L))
    if side * side < L:
        side += 1
    pad = side * side - L
    if pad > 0:
        flat = F.pad(flat, (0, pad))
    imgs = flat.view(B, S * C, side, side)
    return imgs, {"side": side, "pad": pad, "latent_len": L}


def imgs_to_latents(imgs: torch.Tensor, meta: dict, num_stems: int) -> torch.Tensor:
    """Invert encode_batch_to_imgs: (B, S*C, side, side) -> (B, S, C, L)."""
    B, SC, side, _ = imgs.shape
    L_pad = side * side
    C = SC // num_stems
    flat = imgs.view(B, SC, L_pad)
    if meta.get("pad", 0) > 0:
        flat = flat[..., : L_pad - meta["pad"]]
    L = flat.shape[-1]
    return flat.view(B, num_stems, C, L)


def main():
    # Config
    CFG_PATH = os.getenv("CFG_PATH", "configs/cd4mt_medium.yaml")
    with open(CFG_PATH, 'r') as f:
        cfg = yaml.safe_load(f)
    print(f"Using config: {CFG_PATH}")

    # Data
    dm = DataModuleFromConfig(**cfg["data"]["params"])
    dm.prepare_data(); dm.setup(stage="fit")

    # Hyperparams (CT)
    # Quick run knobs from env
    FAST_RUN = os.getenv("FAST_RUN", "1") == "1"
    MAX_STEPS_PER_EPOCH = int(os.getenv("MAX_STEPS", "5")) if FAST_RUN else None

    ct_hparams = {
        'batch_size': int(cfg['data']['params'].get('batch_size', 4)),
        'lr': float(cfg['model']['params'].get('base_learning_rate', 1e-4)),
        'ema_decay': 0.95,
        'num_scales': 16 if FAST_RUN else 32,
        'sigma_min': float(cfg['model']['params'].get('sigma_min', 0.0001)),
        'sigma_max': float(cfg['model']['params'].get('sigma_max', 3.0)),
        'sigma_data': float(cfg['model']['params'].get('diffusion_sigma_data', 0.5)),
        'epochs': 1 if FAST_RUN else 2,  # default; override below
        'grad_clip': 1.0,
        'log_interval': 20,
    }
    # Allow overriding epochs from env or trainer cfg
    try:
        epochs_env = os.getenv('EPOCHS')
        if epochs_env is not None:
            ct_hparams['epochs'] = int(epochs_env)
        else:
            ct_hparams['epochs'] = int(cfg.get('trainer', {}).get('max_epochs', ct_hparams['epochs']))
    except Exception:
        pass

    # Device
    device = auto_select_cuda(min_free_gb=4.0)

    # CAE
    ae = EncoderDecoder(device=device)

    # Build model (CT UNet) and diffusion
    # Make one batch to infer in_channels (S*C)
    first_batch = next(iter(dm.train_dataloader()))
    imgs0, meta0 = encode_batch_to_imgs(ae, first_batch)
    in_channels = imgs0.shape[1]

    cmov = cfg["model"]["params"].get("cm_model_override", {})
    def _get_cmov(k, default):
        v = cmov.get(k, default)
        # cast
        if isinstance(default, bool):
            return bool(v)
        if isinstance(default, int):
            try:
                return int(v)
            except Exception:
                return default
        if isinstance(default, float):
            try:
                return float(v)
            except Exception:
                return default
        return v

    model = create_model(
        image_size=_get_cmov('image_size', 32),
        num_channels=_get_cmov('num_channels', 192),
        num_res_blocks=_get_cmov('num_res_blocks', 2),
        channel_mult=str(_get_cmov('channel_mult', '1,2,4')),
        learn_sigma=_get_cmov('learn_sigma', False),
        class_cond=_get_cmov('class_cond', False),
        use_checkpoint=_get_cmov('use_checkpoint', False),
        attention_resolutions=str(_get_cmov('attention_resolutions', '16,8,4')),
        num_heads=_get_cmov('num_heads', 6),
        num_head_channels=_get_cmov('num_head_channels', 32),
        num_heads_upsample=_get_cmov('num_heads_upsample', -1),
        use_scale_shift_norm=_get_cmov('use_scale_shift_norm', True),
        dropout=_get_cmov('dropout', 0.1),
        resblock_updown=_get_cmov('resblock_updown', False),
        use_fp16=_get_cmov('use_fp16', False),
        use_new_attention_order=_get_cmov('use_new_attention_order', False),
    )

    # Report parameter size for CM/UNet
    try:
        from ldm.modules.util import summarize_params
        summarize_params(model, name="CM.UNet")
    except Exception as _e:
        total = sum(p.numel() for p in model.parameters())
        print(f"CM.UNet params: {total:,} (~{total*4/1024/1024:.1f} MB fp32)")

    # Replace first/last conv to match our in_channels
    model.input_blocks[0][0] = torch.nn.Conv2d(
        in_channels,
        model.input_blocks[0][0].out_channels,
        kernel_size=model.input_blocks[0][0].kernel_size,
        stride=model.input_blocks[0][0].stride,
        padding=model.input_blocks[0][0].padding,
        bias=model.input_blocks[0][0].bias is not None,
    )
    model.out[-1] = torch.nn.Conv2d(
        model.out[-1].in_channels,
        in_channels,
        kernel_size=model.out[-1].kernel_size,
        stride=model.out[-1].stride,
        padding=model.out[-1].padding,
        bias=model.out[-1].bias is not None,
    )

    diffusion = KarrasDenoiser(
        sigma_data=ct_hparams['sigma_data'],
        sigma_min=ct_hparams['sigma_min'],
        sigma_max=ct_hparams['sigma_max'],
        weight_schedule='karras',
        loss_norm='l2',
    )

    model.to(device)
    target_model = copy.deepcopy(model).to(device)
    for p in target_model.parameters():
        p.requires_grad_(False)
    target_model.eval()

    optimizer = torch.optim.Adam(model.parameters(), lr=ct_hparams['lr'])
    scaler = GradScaler()

    # Sanity check: single micro-batch forward/backward to avoid OOM
    micro_bsz = int(os.getenv('MICRO_BSZ', '8'))
    imgs0 = imgs0[:micro_bsz].to(device)
    model.train()
    with autocast(dtype=torch.float16):
        outs = diffusion.consistency_losses(
            model,
            imgs0.half(),
            num_scales=ct_hparams['num_scales'],
            target_model=target_model,
            teacher_model=None,
            teacher_diffusion=None,
        )
        loss0 = outs['loss'].mean()
    optimizer.zero_grad(); scaler.scale(loss0).backward(); scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), ct_hparams['grad_clip'])
    scaler.step(optimizer); scaler.update()
    print(f"Sanity check OK. loss={float(loss0):.6f} (micro_bsz={micro_bsz})")

    # SwanLab init
    run = None
    if swanlab is not None:
        run = swanlab.init(
            project="cd4mt",
            experiment_name=f"ct_unet_{time.strftime('%m%d_%H%M%S')}",
            config={**ct_hparams, "num_params": sum(p.numel() for p in model.parameters())},
        )

    # Train epochs
    best, best_path = float('inf'), None
    os.makedirs('checkpoints', exist_ok=True)
    total_step = 0

    for epoch in range(ct_hparams['epochs']):
        model.train();
        train_loader = dm.train_dataloader()
        run_avg, steps = 0.0, 0
        for step, batch in enumerate(train_loader):
            imgs, meta = encode_batch_to_imgs(ae, batch)
            imgs = imgs.to(device)
            B = imgs.shape[0]
            accum = max(1, (B + micro_bsz - 1) // micro_bsz)
            optimizer.zero_grad()
            loss_total = 0.0
            for i in range(0, B, micro_bsz):
                x = imgs[i:i+micro_bsz]
                with autocast(dtype=torch.float16):
                    losses = diffusion.consistency_losses(
                        model,
                        x.half(),
                        num_scales=ct_hparams['num_scales'],
                        target_model=target_model,
                        teacher_model=None,
                        teacher_diffusion=None,
                    )
                    loss = losses['loss'].mean()
                scaler.scale(loss / accum).backward()
                loss_total += float(loss.item())
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), ct_hparams['grad_clip'])
            scaler.step(optimizer); scaler.update()
            update_ema(target_model.parameters(), model.parameters(), rate=ct_hparams['ema_decay'])
            loss = torch.tensor(loss_total / accum)
            run_avg += float(loss.item()); steps += 1; total_step += 1
            if step % ct_hparams['log_interval'] == 0:
                if run is not None:
                    swanlab.log({"train/loss": float(loss.item()), "epoch": epoch, "step": total_step})
                print(f"[train] epoch={epoch} step={step}/{len(train_loader)} loss={loss.item():.4f}")
            if FAST_RUN and MAX_STEPS_PER_EPOCH is not None and step + 1 >= MAX_STEPS_PER_EPOCH:
                print("Early stop this epoch (FAST_RUN)")
                break
        avg_train = run_avg / max(1, steps)
        if run is not None:
            swanlab.log({"train/avg": float(avg_train), "epoch": epoch})
        print(f"[train] epoch={epoch} avg={avg_train:.4f}")

        # Val
        model.eval();
        val_loader = dm.val_dataloader()
        vrun, vsteps = 0.0, 0
        _cpu = torch.get_rng_state(); _cuda = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
        torch.manual_seed(12345); np.random.seed(12345)
        if torch.cuda.is_available(): torch.cuda.manual_seed_all(12345)
        with torch.no_grad():
            for vstep, vbatch in enumerate(val_loader):
                vimgs, vmeta = encode_batch_to_imgs(ae, vbatch)
                vimgs = vimgs.to(device)
                Bv = vimgs.shape[0]
                torch.manual_seed(12345 + vstep)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(12345 + vstep)
                # micro forward to avoid OOM
                vloss_sum = 0.0
                vcount = 0
                for i in range(0, Bv, micro_bsz):
                    x = vimgs[i:i+micro_bsz]
                    vnoise = torch.randn_like(x)
                    with autocast(dtype=torch.float16):
                        vloss = diffusion.consistency_losses(
                            model,
                            x.half(),
                            num_scales=ct_hparams['num_scales'],
                            target_model=target_model,
                            teacher_model=None,
                            teacher_diffusion=None,
                            noise=vnoise,
                        )['loss'].mean()
                    vloss_sum += float(vloss.item()) * x.shape[0]
                    vcount += x.shape[0]
                vstep_loss = vloss_sum / max(1, vcount)
                vrun += vstep_loss; vsteps += 1
                if FAST_RUN and MAX_STEPS_PER_EPOCH is not None and vstep + 1 >= max(1, MAX_STEPS_PER_EPOCH//2):
                    print("Early stop val (FAST_RUN)")
                    break
        torch.set_rng_state(_cpu); 
        if _cuda is not None: torch.cuda.set_rng_state_all(_cuda)
        avg_val = vrun / max(1, vsteps)
        if run is not None:
            swanlab.log({"val/avg": float(avg_val), "epoch": epoch})
        print(f"[val]  epoch={epoch} avg={avg_val:.4f}")

        if avg_val < best:
            best = avg_val
            best_path = f"checkpoints/ct_unet_ema_best_val{best:.6f}.pth"
            torch.save({'state_dict': target_model.state_dict(), 'epoch': epoch, 'ct_hparams': ct_hparams, 'meta': meta}, best_path)
            if run is not None:
                swanlab.log({"val/best": float(best)})
            print(f"[ckpt] saved best -> {best_path}")
        # Save last EMA every epoch
        last_path = f"checkpoints/ct_unet_ema_last_e{epoch:03d}.pth"
        torch.save({'state_dict': target_model.state_dict(), 'epoch': epoch, 'ct_hparams': ct_hparams, 'meta': meta}, last_path)

        # Sample and decode a short audio preview (first batch only)
        try:
            outdir = Path('samples') / f"e{epoch:03d}_s{total_step:06d}"
            outdir.mkdir(parents=True, exist_ok=True)
            stems = cfg['model']['params'].get('stem_names', [str(i) for i in range(cfg['model']['params'].get('num_stems', 4))])
            num_stems = len(stems)
            sample_steps = int(os.getenv('SAMPLE_STEPS', '20'))
            # use last val batch shapes as reference
            shape = list(vimgs.shape)
            with torch.no_grad():
                gen_imgs = karras_sample(
                    diffusion,
                    model,
                    shape=shape,
                    steps=sample_steps,
                    device=vimgs.device,
                    sigma_min=ct_hparams['sigma_min'],
                    sigma_max=ct_hparams['sigma_max'],
                    model_kwargs={},
                )
            # invert to latents and decode first sample
            gen_latents = imgs_to_latents(gen_imgs.float().cpu(), meta, num_stems=num_stems)
            one = gen_latents[0]  # (S,C,L)
            for si in range(num_stems):
                stem_lat = one[si].numpy()  # (C,L)
                audio = ae.decode(stem_lat, denoising_steps=1)
                # Convert to numpy (T, ch)
                if isinstance(audio, torch.Tensor):
                    at = audio.detach().float().cpu()
                    if at.ndim == 2 and at.shape[0] <= 16 and at.shape[0] < at.shape[1]:
                        at = at.transpose(0, 1)  # (ch,T) -> (T,ch)
                    wav = at.numpy()
                else:
                    wav = np.asarray(audio, dtype=np.float32)
                    if wav.ndim == 2 and wav.shape[0] <= 16 and wav.shape[0] < wav.shape[1]:
                        wav = wav.T
                wav = np.clip(wav, -1.0, 1.0).astype(np.float32)
                if wav.ndim == 1:
                    wav = wav[:, None]
                stem_name = stems[si] if si < len(stems) else f"stem{si}"
                path = outdir / f"gen_{stem_name}.wav"
                sf.write(str(path), wav, 44100, subtype='PCM_16')
                if run is not None and hasattr(swanlab, 'Audio'):
                    try:
                        swanlab.log({f"audio/{stem_name}": swanlab.Audio(str(path))})
                    except Exception:
                        pass
            print(f"[sample] saved preview to {outdir}")
        except Exception as e:
            print(f"[sample] WARN: generation failed: {e}")

    print(f"Best ckpt: {best_path}, val={best:.6f}")
    if run is not None:
        swanlab.finish()


if __name__ == "__main__":
    main()
