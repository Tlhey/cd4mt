#!/usr/bin/env python
"""
ct.py - Core utilities for Consistency Training (CT) and Distillation

This module adapts the open-source "Consistency Models" formulation to the
repo's 1D latent audio diffusion setting (see src/score/diffusion.py).

Key pieces provided:
- get_weightings(): weighting schedules from the CM reference implementation.
- karras_pair(): utility to sample adjacent (t, t_next) from a Karras schedule.
- consistency_loss(): single-batch CT/CD loss that compares the student
  denoised output at t against the target model's denoised output at t_next,
  with x_{t_next} produced by either a Heun (teacher) or Euler (no-teacher)
  one-step update, closely following src/consistency_models/cm/karras_diffusion.py.

This file is intentionally small; ct_train.py implements the loop and EMA.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

import torch
import torch.nn.functional as F


WeightSchedule = Literal["snr", "snr+1", "karras", "truncated-snr", "uniform"]
LossNorm = Literal["l2", "l1"]


def get_weightings(schedule: WeightSchedule, snrs: torch.Tensor, sigma_data: float) -> torch.Tensor:
    """Weighting schedules from CM reference implementation.

    - snr: 1 / sigma^2
    - karras: snr + 1/sigma_data^2 (helps stabilize high-noise regime)
    """
    if schedule == "snr":
        return snrs
    if schedule == "snr+1":
        return snrs + 1.0
    if schedule == "karras":
        return snrs + 1.0 / (sigma_data ** 2)
    if schedule == "truncated-snr":
        return torch.clamp(snrs, min=1.0)
    if schedule == "uniform":
        return torch.ones_like(snrs)
    raise NotImplementedError(f"Unknown weight schedule: {schedule}")


@dataclass
class KarrasScheduleSpec:
    sigma_min: float = 0.002
    sigma_max: float = 5.0
    rho: float = 7.0
    num_scales: int = 64


def _karras_sigma_at(index: torch.Tensor, spec: KarrasScheduleSpec, device: torch.device) -> torch.Tensor:
    """Return sigma(index) for a discrete Karras schedule as in CM repo.

    index may be a vector of per-example indices in [0, num_scales-1].
    """
    r_inv = 1.0 / float(spec.rho)
    s_min = spec.sigma_min ** r_inv
    s_max = spec.sigma_max ** r_inv
    frac = index.to(torch.float32) / max(1, spec.num_scales - 1)
    sigma = (s_max + frac * (s_min - s_max)) ** spec.rho
    return sigma.to(device=device)


def karras_pair(batch_size: int, spec: KarrasScheduleSpec, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    """Sample adjacent (t, t_next) from a discrete Karras schedule.

    Returns two [B] tensors of noise levels with t >= t_next > 0.
    """
    assert spec.num_scales >= 2, "num_scales must be >= 2"
    # indices in [0, num_scales-2]
    idx = torch.randint(0, spec.num_scales - 1, (batch_size,), device=device)
    t = _karras_sigma_at(idx, spec, device)
    t_next = _karras_sigma_at(idx + 1, spec, device)
    return t, t_next


def _snr(sigma: torch.Tensor) -> torch.Tensor:
    return sigma.reciprocal() ** 2


@torch.no_grad()
def _heun_step(x: torch.Tensor,
               t: torch.Tensor,
               t_next: torch.Tensor,
               x0: torch.Tensor,
               teacher_denoise_fn) -> torch.Tensor:
    """Single Heun step using teacher denoise function as in CM repo."""
    dims = x.ndim
    if teacher_denoise_fn is None:
        denoised = x0
    else:
        denoised = teacher_denoise_fn(x, t)
    d = (x - denoised) / t.view(-1, *([1] * (dims - 1)))
    x_euler = x + (t_next - t).view(-1, *([1] * (dims - 1))) * d

    if teacher_denoise_fn is None:
        denoised_next = x0
    else:
        denoised_next = teacher_denoise_fn(x_euler, t_next)
    d_next = (x_euler - denoised_next) / t_next.view(-1, *([1] * (dims - 1)))

    x_heun = x + 0.5 * (t_next - t).view(-1, *([1] * (dims - 1))) * (d + d_next)
    return x_heun


@torch.no_grad()
def _euler_step(x: torch.Tensor,
                t: torch.Tensor,
                t_next: torch.Tensor,
                x0: torch.Tensor,
                teacher_denoise_fn=None) -> torch.Tensor:
    """Single Euler step. If no teacher present, use x0 as denoiser."""
    dims = x.ndim
    if teacher_denoise_fn is None:
        denoised = x0
    else:
        denoised = teacher_denoise_fn(x, t)
    d = (x - denoised) / t.view(-1, *([1] * (dims - 1)))
    x_next = x + (t_next - t).view(-1, *([1] * (dims - 1))) * d
    return x_next


def consistency_loss(
    *,
    student_denoise_fn,
    target_denoise_fn,
    x_start: torch.Tensor,
    spec: KarrasScheduleSpec,
    sigma_data: float,
    weight_schedule: WeightSchedule = "karras",
    loss_norm: LossNorm = "l2",
    teacher_denoise_fn=None,
    rng: Optional[torch.Generator] = None,
) -> torch.Tensor:
    """Compute the per-example CT loss and return mean over batch.

    Arguments:
      - student_denoise_fn: callable (x_noisy, sigmas|sigma) -> x_denoised for student.
      - target_denoise_fn: callable for target/EMA network at t_next.
      - teacher_denoise_fn: optional callable for Heun step. If None, Euler with x0.
    """
    device = x_start.device
    B = x_start.shape[0]

    if rng is not None:
        torch.manual_seed(int(torch.seed()))  # keep global state safe

    # Sample a noise and adjacent sigmas
    noise = torch.randn_like(x_start, device=device)
    t, t_next = karras_pair(B, spec, device)

    # Construct x_t and advance one step to x_{t_next}
    x_t = x_start + t.view(-1, *([1] * (x_start.ndim - 1))) * noise

    # distiller/student at t
    student_out = student_denoise_fn(x_t, sigmas=t)

    # Advance with teacher (Heun) if provided; otherwise Euler with x0
    x_t_next = _heun_step(x_t, t, t_next, x_start, teacher_denoise_fn) if teacher_denoise_fn is not None else _euler_step(x_t, t, t_next, x_start)

    # target prediction at t_next (EMA model)
    with torch.no_grad():
        target_out = target_denoise_fn(x_t_next, sigmas=t_next)

    # Weighting by SNR schedule
    snr = _snr(t)
    weights = get_weightings(weight_schedule, snr, sigma_data)

    # Per-example distance
    if loss_norm == "l1":
        diffs = torch.abs(student_out - target_out)
    else:  # "l2"
        diffs = (student_out - target_out) ** 2

    # mean over non-batch dims, then weight and average
    while diffs.ndim > 1:
        diffs = diffs.mean(dim=-1)
    per_example = diffs * weights
    return per_example.mean()

