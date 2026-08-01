"""Validate the Kaggle training setup."""

import argparse
import os
from collections import Counter
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

from src.datasets.data_utils import get_dataloaders
from src.loss import LcnnLoss
from src.utils.io_utils import ROOT_PATH

EXPECTED_SPLITS = {
    "train": {"total": 25_380, "bonafide": 2_580, "spoof": 22_800},
    "dev": {"total": 24_844, "bonafide": 2_548, "spoof": 22_296},
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracker", choices=("wandb", "comet"), required=True)
    return parser.parse_args()


def check_tracker(tracker):
    if tracker == "wandb":
        if not os.environ.get("WANDB_API_KEY"):
            raise RuntimeError("WANDB_API_KEY is not loaded from Kaggle Secrets.")

        import wandb

        if not wandb.login(anonymous="never"):
            raise RuntimeError("W&B authentication failed.")
        print("tracker: W&B authentication passed")
        return

    api_key = os.environ.get("COMET_API_KEY")
    if not api_key:
        raise RuntimeError("COMET_API_KEY is not loaded from Kaggle Secrets.")

    from comet_ml.api import API

    workspace = API(api_key=api_key).get_default_workspace()
    if not workspace:
        raise RuntimeError("Comet authentication failed: no default workspace.")
    print(f"tracker: Comet authentication passed (workspace={workspace})")


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for the full Kaggle run.")
    check_tracker(args.tracker)

    config_dir = str(ROOT_PATH / "src" / "configs")
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        config = compose(config_name="highscore")

    device = "cuda"
    dataloaders, _ = get_dataloaders(config, device)

    for split, expected in EXPECTED_SPLITS.items():
        dataset = dataloaders[split].dataset
        counts = Counter(item["label"] for item in dataset._index)
        actual = {
            "total": len(dataset),
            "bonafide": counts[1],
            "spoof": counts[0],
        }
        if actual != expected:
            raise RuntimeError(
                f"Unexpected {split} protocol counts: {actual}; expected {expected}."
            )

        for entry in (dataset._index[0], dataset._index[-1]):
            if not Path(entry["path"]).is_file():
                raise FileNotFoundError(entry["path"])

        print(f"{split}: {actual}")

    batch = next(iter(dataloaders["train"]))
    expected_shape = (config.dataloader.batch_size, 64, 600)
    if tuple(batch["data_object"].shape) != expected_shape:
        raise RuntimeError(
            f"Unexpected feature batch shape {tuple(batch['data_object'].shape)}; "
            f"expected {expected_shape}."
        )

    model = instantiate(config.model).to(device)
    criterion = LcnnLoss().to(device)
    data = batch["data_object"].to(device, non_blocking=True)
    labels = batch["labels"].to(device, non_blocking=True)
    outputs = model(data_object=data)
    loss = criterion(logits=outputs["logits"], labels=labels)["loss"]
    loss.backward()

    if not torch.isfinite(loss):
        raise RuntimeError(f"Non-finite smoke-test loss: {loss.item()}")
    if tuple(outputs["logits"].shape) != (config.dataloader.batch_size, 2):
        raise RuntimeError(f"Unexpected logits shape: {tuple(outputs['logits'].shape)}")

    print(f"device: {torch.cuda.get_device_name(0)}")
    print(f"features: {tuple(data.shape)}")
    print(f"logits: {tuple(outputs['logits'].shape)}")
    print(f"loss: {loss.item():.6f}")
    print("PREFLIGHT PASSED: the full highscore run can be started.")


if __name__ == "__main__":
    main()
