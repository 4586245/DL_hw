import random

import torch
import torchaudio
from torch import nn


class MelSpectrogram(nn.Module):
    """Fixed-size power mel-spectrogram."""

    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=320,
        hop_length=160,
        n_mels=64,
        num_frames=600,
        is_train=False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.is_train = is_train
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            win_length=win_length,
            hop_length=hop_length,
            n_mels=n_mels,
            window_fn=torch.blackman_window,
            power=2.0,
            center=True,
            pad_mode="reflect",
            normalized=False,
        )

    def forward(self, wav):
        spec = self.mel(wav)
        frames = spec.shape[-1]

        if frames > self.num_frames:
            start = 0
            if self.is_train:
                start = random.randint(0, frames - self.num_frames)
            spec = spec[..., start : start + self.num_frames]
        elif frames < self.num_frames:
            spec = torch.nn.functional.pad(spec, (0, self.num_frames - frames))

        return spec
