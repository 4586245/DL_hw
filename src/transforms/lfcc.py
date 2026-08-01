import random

import torch
import torchaudio
from torch import nn


class LFCC(nn.Module):
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=320,
        hop_length=160,
        n_filter=20,
        n_lfcc=20,
        num_frames=750,
        is_train=False,
    ):
        super().__init__()
        self.num_frames = num_frames
        self.is_train = is_train

        self.lfcc = torchaudio.transforms.LFCC(
            sample_rate=sample_rate,
            n_filter=n_filter,
            n_lfcc=n_lfcc,
            speckwargs={
                "n_fft": n_fft,
                "win_length": win_length,
                "hop_length": hop_length,
            },
        )
        self.deltas = torchaudio.transforms.ComputeDeltas()

    def forward(self, wav):
        feats = self.lfcc(wav)  # (n_lfcc, frames)
        d1 = self.deltas(feats)
        d2 = self.deltas(d1)
        feats = torch.cat([feats, d1, d2], dim=0)  # (60, frames)

        return self._trim_pad(feats)

    def _trim_pad(self, feats):
        frames = feats.shape[-1]

        if frames > self.num_frames:
            if self.is_train:
                start = random.randint(0, frames - self.num_frames)
            else:
                start = 0
            feats = feats[:, start : start + self.num_frames]
        elif frames < self.num_frames:
            feats = torch.nn.functional.pad(feats, (0, self.num_frames - frames))

        return feats
