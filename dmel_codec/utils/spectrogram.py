import torch
import torchaudio.functional as F
from torch import Tensor, nn
from librosa.filters import mel as librosa_mel_fn


class LinearSpectrogram(nn.Module):

    def __init__(
        self,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        center=False,
        num_mels=128,
        f_min=0,
        f_max=None,
        sample_rate=44100,
        mode="reflect",
    ):
        super().__init__()

        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.mode = mode
        self.f_min = f_min
        self.f_max = f_max
        self.num_mels = num_mels
        self.sample_rate = sample_rate
        self.mel_basis_cache = {}
        self.hann_window_cache = {}

    def spectral_normalize_torch(self, magnitudes: Tensor) -> Tensor:
        return self.dynamic_range_compression_torch(magnitudes)

    def dynamic_range_compression_torch(self, x: Tensor, C=1, clip_val=1e-5) -> Tensor:
        return torch.log(torch.clamp(x, min=clip_val) * C)

    def forward(self, y: Tensor) -> Tensor:
        device = y.device
        key = f"{self.n_fft}_{self.num_mels}_{self.sample_rate}_{self.hop_length}_{self.win_length}_{self.f_min}_{self.f_max}_{device}"
        if key not in self.mel_basis_cache:
            mel = librosa_mel_fn(
                sr=self.sample_rate,
                n_fft=self.n_fft,
                n_mels=self.num_mels,
                fmin=self.f_min,
                fmax=self.f_max,
            )
            self.mel_basis_cache[key] = torch.from_numpy(mel).float().to(device)
            self.hann_window_cache[key] = torch.hann_window(self.win_length).to(device)

        mel_basis = self.mel_basis_cache[key]
        hann_window = self.hann_window_cache[key]

        padding = (self.n_fft - self.hop_length) // 2

        y = torch.nn.functional.pad(
            y.unsqueeze(1) if y.ndim == 2 else y, (padding, padding), mode=self.mode
        ).squeeze(1)

        spec = torch.stft(
            y,
            self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=hann_window,
            center=self.center,
            pad_mode=self.mode,
            normalized=False,
            onesided=True,
            return_complex=True,
        )
        spec = torch.sqrt(torch.view_as_real(spec).pow(2).sum(-1) + 1e-9)

        mel_spec = torch.matmul(mel_basis, spec)
        mel_spec = self.spectral_normalize_torch(mel_spec)

        return mel_spec


class LogMelSpectrogram(nn.Module):
    def __init__(
        self,
        sample_rate=44100,
        n_fft=2048,
        win_length=2048,
        hop_length=512,
        n_mels=128,
        center=False,
        f_min=0.0,
        f_max=None,
    ):
        super().__init__()

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.center = center
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max or float(sample_rate // 2)

        self.spectrogram = LinearSpectrogram(
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            center=center,
            num_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            sample_rate=sample_rate,
            mode="reflect",
        )

    def forward(
        self, x: Tensor, return_linear: bool = False, sample_rate: int = None
    ) -> Tensor:
        if sample_rate is not None and sample_rate != self.sample_rate:
            x = F.resample(x, orig_freq=sample_rate, new_freq=self.sample_rate)

        x = self.spectrogram(x)

        return x
