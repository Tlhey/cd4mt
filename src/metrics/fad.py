import torch
import torch.nn.functional as F
from typing import List, Tuple


def _log_mel_embedding(wavs: List[torch.Tensor], sr: int, n_mels: int = 128,
                       n_fft: int = 1024, hop_length: int = 512) -> torch.Tensor:
    """Compute simple log-Mel embeddings for a list of waveforms.

    Each waveform is converted to a log-mel spectrogram and split into frames;
    we then treat each time frame as an embedding vector (n_mels), stacking
    across all files, which approximates the embedding distribution used by FAD.
    """
    import torchaudio

    mel = torchaudio.transforms.MelSpectrogram(
        sample_rate=sr, n_fft=n_fft, hop_length=hop_length, n_mels=n_mels,
        f_min=20.0, f_max=sr // 2
    )
    to_db = torchaudio.transforms.AmplitudeToDB(stype='power')

    feats = []
    for w in wavs:
        if w.ndim == 2:
            # mixdown
            w = w.mean(dim=0)
        w = w.to(torch.float32)
        S = mel(w.unsqueeze(0))            # [1, n_mels, T]
        S = to_db(S).squeeze(0)            # [n_mels, T]
        S = (S - S.mean()) / (S.std() + 1e-6)
        feats.append(S.T)                   # [T, n_mels]

    return torch.cat(feats, dim=0)          # [N_frames, n_mels]


def _sym_sqrtm(A: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Matrix square root for symmetric positive definite matrices."""
    # Ensure symmetry
    A = 0.5 * (A + A.T)
    eigvals, eigvecs = torch.linalg.eigh(A)
    eigvals = torch.clamp(eigvals, min=eps)
    return (eigvecs * eigvals.sqrt()) @ eigvecs.T


def _mean_and_cov(X: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    mu = X.mean(dim=0)
    Xc = X - mu
    # unbiased covariance
    C = (Xc.T @ Xc) / (Xc.shape[0] - 1)
    # numerical jitter for stability
    C = 0.5 * (C + C.T) + torch.eye(C.shape[0], device=C.device) * 1e-6
    return mu, C


@torch.no_grad()
def compute_fad(reference_wavs: List[torch.Tensor], generated_wavs: List[torch.Tensor], sr: int) -> float:
    """Compute Fréchet-like Audio Distance between two sets of waveforms.

    This implementation follows the FAD structure but uses log-mel frames as
    embeddings to avoid external dependencies. If you prefer the official VGGish
    embeddings, swap `_log_mel_embedding` with a VGGish/YAMNet extractor.
    """
    ref = _log_mel_embedding(reference_wavs, sr)
    gen = _log_mel_embedding(generated_wavs, sr)

    mu_r, C_r = _mean_and_cov(ref)
    mu_g, C_g = _mean_and_cov(gen)

    sqrt_C = _sym_sqrtm(_sym_sqrtm(C_r) @ C_g @ _sym_sqrtm(C_r))

    diff = mu_r - mu_g
    fad = float(diff.dot(diff) + torch.trace(C_r + C_g - 2 * sqrt_C))
    return max(fad, 0.0)

