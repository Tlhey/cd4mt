# CD4MT – Minimal Training + Inference Notebook

A compact workspace for experimenting with CD4MT on Slakh multi‑track audio. The main entry is the notebook `visualize_cd4mt.ipynb`, which walks through data loading, CAE encoding/decoding, CT‑UNet setup, generation, and simple visualizations.

- Flow: ldm data loader, music2latent CAE, CT UNet, inspect shapes, visualize and play audio.

## Project Report
Everything in visualize_cd4mt.ipynb
- 1. env – device and imports.
- 2. config and load_data – prints config and fetches a batch; logs `(B,S,T)`.
- 3. CAE encoder test – encodes to `(B,64,L)`, stacks to `(B,4,64,L)`; then decode + MSE.
- 4. CD4MT model – CT UNet with `S*C_lat=256` in/out, load EMA ckpt.
- 5. Generation test – use Karras sampler (`test.py` pattern) or Lightning `ScoreDiffusionModel.sample(...)`.d

## Key Shapes
- Waveforms: `(B, S, T)` (e.g., `(4, 4, 524288)`); mix `(B, T)`.
- CAE latent per stem: `(B, C_lat, L)` (e.g., `(4, 64, 127)`).
- Stacked latents: `(B, S, C_lat, L)` (e.g., `(4, 4, 64, 127)`).
- 2D map for UNet: `(B, S*C_lat, H, W)` with `H = W = ceil(sqrt(L))` (for `L=127`, `H=W=12`).

## Data & Configs
- Dataset: `dataset/slakh_44100` (https://github.com/gladia-research-group/multi-source-diffusion-models/tree/main/data)
- Example configs:
  - `configs/cd4mt_small.yaml`
  - `configs/cd4mt_medium.yaml`

## Checkpoints
- CT‑UNet EMA checkpoints under `checkpoints/` are auto‑discovered, e.g.:
  - `checkpoints/ct_unet_ema_best_val*.pth`
  - `checkpoints/ct_unet_ema_last_e*.pth`

## Code Used:
CM: https://github.com/openai/consistency_models/
Music2latnet: https://github.com/SonyCSLParis/music2latent/

