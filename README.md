# LCNN Voice Anti-spoofing

PyTorch-template implementation of a countermeasure for the Logical Access
partition of ASVspoof 2019. The system classifies each utterance as bona fide
or spoofed and reports Equal Error Rate (EER).

## Method

The input waveform is converted to a log-power STFT with a 25 ms Hann window,
10 ms hop, 512-point FFT, and 750 time frames. During training, long recordings
are cropped randomly; evaluation uses a deterministic crop. A learnable linear
frequency projection, initialized as a 60-band linear filterbank, feeds an LCNN
whose convolutional and fully connected blocks use Max-Feature-Map activations.
The classifier is trained with two-class cross-entropy. Dropout is placed before
the final batch-normalization layer as required by the assignment.

Labels and scores follow the grading convention: `bonafide = 1`, `spoof = 0`,
and a larger submitted score means stronger support for bona fide speech.

## Installation

Python 3.10 is recommended. Install dependencies in a clean environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
```

Configure either Comet ML or Weights & Biases before a tracked run. The
one-shot Kaggle recipe uses Comet via the `COMET_API_KEY` environment variable.
For a W&B run:

```bash
wandb login
```

## Dataset

Download the ASVspoof 2019 LA data and protocols. The default paths target the
Kaggle dataset `awsaf49/asvpoof-2019-dataset`. For another location, override
the Hydra values on the command line. For example:

```bash
python3 train.py \
  datasets.train.audio_dir=/data/LA/ASVspoof2019_LA_train/flac \
  datasets.train.protocol_path=/data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.train.trn.txt \
  datasets.dev.audio_dir=/data/LA/ASVspoof2019_LA_dev/flac \
  datasets.dev.protocol_path=/data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.dev.trl.txt \
  datasets.eval.audio_dir=/data/LA/ASVspoof2019_LA_eval/flac \
  datasets.eval.protocol_path=/data/LA/ASVspoof2019_LA_cm_protocols/ASVspoof2019.LA.cm.eval.trl.txt
```

## Training

For the single production Kaggle run, follow
[`KAGGLE_RUNBOOK.md`](KAGGLE_RUNBOOK.md) exactly. It includes a fail-fast GPU,
dataset, Comet authentication, feature-shape, forward, and backward preflight.

Run the one-batch overfitting check first:

```bash
python3 train.py -cn=onebatch
```

Then train the full model:

```bash
python3 train.py -cn=baseline
```

An additional reproduced recipe (`64` mel bins, `600` frames, AdamW and cosine
decay) is available as a separate experiment. Its source notebook recorded
`4.609586%` evaluation EER:

```bash
python3 train.py -cn=highscore
python3 inference.py -cn=inference_mel
```

Use the equivalent Comet-tracked configuration when W&B is unavailable:

```bash
python3 train.py -cn=highscore_comet
```

This recipe is kept separate from the STFT baseline so both experiments remain
reproducible and can be compared in the report.

The default recipe uses Adam with learning rate `3e-4`, batch size 32, and a
StepLR decay of 0.5 every 10 epochs. The tracker logs train loss, development/evaluation
loss, EER, learning rate, and gradient norm. Checkpoints are written beneath
`saved/lcnn-baseline/`; `model_best.pth` is selected by minimum development EER.

Hydra overrides can adjust resources, for example:

```bash
python3 train.py -cn=baseline dataloader.batch_size=16 dataloader.num_workers=4
```

## Evaluation and submission

Run inference with the best checkpoint. Dataset path overrides work as above:

```bash
python3 inference.py \
  inferencer.from_pretrained=saved/lcnn-baseline/model_best.pth
```

Predictions are saved to `data/saved/asvspoof/eval_scores.csv` with no header,
one `utterance_id,score` pair per line, matching `grading.py`. Rename the file
to your university username, place it in `students_solutions`, copy the official
evaluation protocol beside `grading.py`, and verify it:

```bash
mkdir -p students_solutions
cp data/saved/asvspoof/eval_scores.csv students_solutions/your_username.csv
python3 grading.py
```

Do not commit API keys, Kaggle tokens, or other credentials.

## Project structure

- `src/model/lcnn.py` — LCNN and Max-Feature-Map layers.
- `src/transforms/stft.py` — fixed-length log-power STFT front-end.
- `src/datasets/asv_dataset.py` — protocol parsing and audio loading.
- `src/metrics/eer.py` — utterance-level EER accumulation.
- `src/configs/` — reproducible training and inference configurations.
- `train.py`, `inference.py` — template entry points.

The final course report and exported Comet plots are experiment artifacts and
must be added after the full training run.
