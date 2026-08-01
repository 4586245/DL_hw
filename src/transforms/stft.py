import random

import torch
from torch import nn


class spectrogram(nn.Module):
    def __init__(
        self,
        n_fft=512,
        win_length=320,
        hop_length=160,
        num_frames=750,
        is_train=False,
        eps=1e-8,
    ):
        super().__init__()
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.num_frames = num_frames
        self.is_train = is_train
        self.eps = eps
        self.register_buffer("window", torch.hann_window(win_length))

    def forward(self, wav):
        spec = torch.stft(
            wav,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            return_complex=True,
        )
        spec = torch.log(spec.abs().pow(2) + self.eps)
        return self._trim_pad(spec)

    def _trim_pad(self, spec):
        # from paper: K = 750
        frames = spec.shape[-1]

        if frames > self.num_frames:
            if self.is_train:
                start = random.randint(0, frames - self.num_frames)
            else:
                start = 0
            spec = spec[:, start : start + self.num_frames]

        elif frames < self.num_frames:
            spec = torch.nn.functional.pad(spec, (0, self.num_frames - frames))

        return spec
