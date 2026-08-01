"""Validate a two-column ASVspoof submission and print its EER."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

from calculate_eer import compute_eer


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("protocol_path", type=Path)
    return parser.parse_args()


def main():
    args = parse_args()
    protocol = []
    with args.protocol_path.open() as file:
        for line_number, line in enumerate(file, start=1):
            fields = line.split()
            if len(fields) != 5:
                raise ValueError(f"Malformed protocol line {line_number}")
            protocol.append((fields[1], int(fields[-1] == "bonafide")))

    scores = {}
    with args.csv_path.open(newline="") as file:
        for line_number, row in enumerate(csv.reader(file), start=1):
            if len(row) != 2:
                raise ValueError(f"CSV line {line_number} must contain two fields")
            utterance_id, raw_score = row
            if utterance_id in scores:
                raise ValueError(f"Duplicate utterance ID: {utterance_id}")
            try:
                score = float(raw_score)
            except ValueError as error:
                raise ValueError(
                    f"Non-numeric score on CSV line {line_number}: {raw_score}"
                ) from error
            if not math.isfinite(score):
                raise ValueError(f"Non-finite score for {utterance_id}")
            if not 0.0 <= score <= 1.0:
                raise ValueError(f"Score outside [0, 1] for {utterance_id}: {score}")
            scores[utterance_id] = score

    protocol_ids = [utterance_id for utterance_id, _ in protocol]
    missing = set(protocol_ids) - scores.keys()
    extra = scores.keys() - set(protocol_ids)
    if missing or extra:
        raise ValueError(
            f"ID mismatch: {len(missing)} missing and {len(extra)} unexpected IDs"
        )
    if len(scores) != len(protocol):
        raise ValueError(
            f"Row count mismatch: CSV={len(scores)}, protocol={len(protocol)}"
        )

    ordered_scores = np.asarray([scores[item] for item in protocol_ids])
    labels = np.asarray([label for _, label in protocol])
    eer, _ = compute_eer(ordered_scores[labels == 1], ordered_scores[labels == 0])
    eer_percent = 100.0 * eer

    print(f"rows: {len(scores)}")
    print("format: valid two-column CSV without a header")
    print("IDs: exact protocol match")
    print(f"EER: {eer_percent:.6f}%")


if __name__ == "__main__":
    main()
