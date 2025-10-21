# Assignment 3 — CNN from scratch (mini-repo)

This folder contains a small implementation of a convolutional neural network (CNN) for the CIFAR-10
assignment. It includes data loading, a minimal CNN, training experiment harness and utility code used
in the course assignments.

Contents
- `cnn.py` — CNN implementation (convolution via patch-matrix MX + fully connected layers).
- `dataset.py` — CIFAR-10 dataset loader and one-hot label utilities.
- `experiment.py` — Training loop and logging harness using a cyclical learning rate.
- `utils.py`, `learning_rate.py`, `plot.py` — helpers for activations, loss, LR schedule and plotting.
- `test_cnn.py` — unit / smoke tests for the CNN code.
- `assignment_3.py` — top-level runner (small script to launch experiments).

Quick setup

1. Create and activate a Python environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run an experiment:

```bash
python assignment_3.py --arch <architecture>
```

Where `<architecture>` is one of `arch1`, `arch2`, `arch3`, `arch4`, `arch2_long`, `arch3_long`, `arch2_long_more_filters`, `arch5_baseline`, `arch5_label_smoothing`.

Run unit tests (pytest):

```bash
pytest
```

- Here we check the correctness of the CNN forward and backward passes, and the gradient computations.
