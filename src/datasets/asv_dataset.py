import random
from pathlib import Path

import torchaudio

from src.datasets.base_dataset import BaseDataset


class ASVsDataset(BaseDataset):
    def __init__(
        self,
        audio_dir,
        protocol_path,
        *args,
        **kwargs,
    ):
        index = self._build_index(audio_dir, protocol_path)
        super().__init__(index, *args, **kwargs)

    def _build_index(self, audio_dir, protocol_path):
        audio_dir = Path(audio_dir)
        index = []
        with open(protocol_path, "r") as f:
            for line in f:
                a, utt_id, b, c, label = line.strip().split()
                index.append(
                    {
                        "path": str(audio_dir / f"{utt_id}.flac"),
                        "label": 1 if label == "bonafid" else 0,
                    }
                )
        return index

    def load_object(self, path):
        wav, a = torchaudio.load(path)
        return wav.squeeze(0)
