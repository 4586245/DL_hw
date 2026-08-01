import torch
from torch import nn


# mfm was used in STC ... for the ASVspoof2019 paper
class MFM(nn.Module):
    def __init__(self, dim=1):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        a, b = torch.chunk(x, 2, dim=self.dim)
        return torch.max(a, b)


def linear_filterbank(n_freq, n_filter, sample_rate=16000):
    f_max = sample_rate / 2
    edges = torch.linspace(0, f_max, n_filter + 2)
    freqs = torch.linspace(0, f_max, n_freq)

    fb = torch.zeros(n_filter, n_freq)
    for i in range(n_filter):
        left, center, right = edges[i], edges[i + 1], edges[i + 2]

        rising = (freqs - left) / (center - left)
        falling = (right - freqs) / (right - center)

        fb[i] = torch.clamp(torch.minimum(rising, falling), min=0.0)

    return fb


class LCNN(nn.Module):
    def __init__(self, in_freq=257, compressed=60, dropout=0.75, n_class=2):
        super().__init__()
        if in_freq == compressed:
            self.freq_compress = nn.Identity()
        else:
            self.freq_compress = nn.Linear(in_freq, compressed)

            with torch.no_grad():
                self.freq_compress.weight.copy_(linear_filterbank(in_freq, compressed))
                self.freq_compress.bias.zero_()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=5, stride=1, padding=2),
            MFM(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 96, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(2, 2),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 96, kernel_size=1, stride=1),
            MFM(),
            nn.BatchNorm2d(48),
            nn.Conv2d(48, 128, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=1, stride=1),
            MFM(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=1, stride=1),
            MFM(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            MFM(),
            nn.MaxPool2d(2, 2),
        )

        self.head = nn.Sequential(
            nn.Linear(32 * 3 * 46, 160),
            MFM(),
            nn.Dropout(dropout),
            nn.BatchNorm1d(80),
            nn.Linear(80, n_class),
        )

    def forward(self, data_object, **batch):
        x = data_object
        x = x.transpose(1, 2)  # (batch, 750, 257)
        x = self.freq_compress(x)  # (batch, 750, 60)
        x = x.transpose(1, 2)  # (batch, 60, 750)

        x = x.unsqueeze(1)  # (batch, 1, 60, 750)
        x = self.conv(x)  # (batch, 32, 3, 46)
        x = x.flatten(1)  # (batch, 4416)
        logits = self.head(x)  # (batch, 2)
        return {"logits": logits}

    def __str__(self):
        params = sum([p.numel() for p in self.parameters()])
        train_params = sum([p.numel() for p in self.parameters() if p.requires_grad])

        res = super().__str__()
        res = res + f"\nAll parameters: {params}"
        res = res + f"\nTrainable parameters: {train_params}"
        return res
