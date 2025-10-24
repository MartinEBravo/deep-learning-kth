import numpy as np
import torch
from pathlib import Path
import pickle
import pprint
import json
from dataclasses import dataclass
from tqdm import tqdm
import time
import pandas as pd
import matplotlib.pyplot as plt

pprint = pprint.PrettyPrinter(indent=4)

BASE_DIR = Path(__file__).resolve().parent
DATA_ROOT = BASE_DIR.parent / "Datasets"
REPORT_IMG_DIR = BASE_DIR.parent / "reports" / "imgs"
SUMMARY_PATH = REPORT_IMG_DIR.parent / "reports"
debug_file = DATA_ROOT / "debug_info.npz"
load_data = np.load(debug_file)

# ---- Utility Functions -----


def ReLU(s):
    return np.maximum(0, s)


def softmax(s):
    """
    Maps an array of logits to a probability distribution.
    """
    s = s - np.max(s, axis=0, keepdims=True)
    e = np.exp(s)
    return e / np.sum(e, axis=0, keepdims=True)


def compute_accuracy(p, y_true):
    """
    Compute classification accuracy given probabilities and integer labels.
    """
    predictions = np.argmax(p, axis=0)
    return np.mean(predictions == y_true)


def apply_label_smoothing(Y, eps):
    """
    Apply label smoothing to one-hot encoded targets Y of shape (K, n).
    eps in [0, 1). For eps=0 returns Y unchanged.
    """
    if eps <= 0.0:
        return Y
    K = Y.shape[0]
    return (1.0 - eps) * Y + (eps / (K - 1)) * (1.0 - Y)


def to_one_hot(y: np.ndarray):
    """
    Converts a numpy array of labels to a one-hot encoded numpy array.
    """
    return np.eye(10)[y]


# ---- Dataset  -----


class CIFAR10Dataset:
    def __init__(self):
        self._prepare_datasets()

    def _load_batch(self, cifar_dir: str):
        with open(cifar_dir, "rb") as fo:
            dict = pickle.load(fo, encoding="bytes")
        X = dict[b"data"].astype(np.float32) / 255.0
        X = X.transpose()
        d = X.shape[0]
        n = X.shape[1]
        y = np.array(dict[b"labels"])
        Y = np.zeros((10, n))
        for i in range(len(y)):
            Y[:, i] = to_one_hot(y[i])
        K = Y.shape[0]
        assert X.shape == (d, n), "Dimensions invalid"
        assert Y.shape == (K, n), "Dimensions invalid"
        assert y.shape == (n,), "Dimensions invalid"
        return X, Y, y

    def _normalize_data(self, X_train, X_validation, X_test, d: int = 3072):
        mean_X = np.mean(X_train, axis=1).reshape(d, 1)
        std_X = np.std(X_train, axis=1).reshape(d, 1)
        std_X[std_X == 0] = 1.0
        X_train = (X_train - mean_X) / std_X
        X_validation = (X_validation - mean_X) / std_X
        X_test = (X_test - mean_X) / std_X
        return X_train, X_validation, X_test

    def _prepare_datasets(self, validation_start: int = 49000):
        """
        Load CIFAR-10 batches, normalize, and reshape for convolutional training.
        """
        train_batch_paths = [
            DATA_ROOT / "cifar-10-batches-py" / f"data_batch_{i}" for i in range(1, 6)
        ]
        test_path = DATA_ROOT / "cifar-10-batches-py" / "test_batch"

        X_train_list, Y_train_list, y_train_list = [], [], []
        for batch_path in train_batch_paths:
            X_batch, Y_batch, y_batch = self._load_batch(batch_path)
            X_train_list.append(X_batch)
            Y_train_list.append(Y_batch)
            y_train_list.append(y_batch)

        X_train_full = np.concatenate(X_train_list, axis=1)
        Y_train_full = np.concatenate(Y_train_list, axis=1)
        y_train_full = np.concatenate(y_train_list, axis=0)

        X_test, Y_test, y_test = self._load_batch(test_path)

        X_val = X_train_full[:, validation_start:]
        Y_val = Y_train_full[:, validation_start:]
        y_val = y_train_full[validation_start:]

        X_train = X_train_full[:, :validation_start]
        Y_train = Y_train_full[:, :validation_start]
        y_train = y_train_full[:validation_start]

        X_train, X_val, X_test = self._normalize_data(X_train, X_val, X_test)

        self.X_train = self._reshape_for_conv(X_train)
        self.Y_train = Y_train
        self.y_train = y_train
        self.X_val = self._reshape_for_conv(X_val)
        self.Y_val = Y_val
        self.y_val = y_val
        self.X_test = self._reshape_for_conv(X_test)
        self.Y_test = Y_test
        self.y_test = y_test

    def _reshape_for_conv(self, X):
        return np.transpose(X.reshape((32, 32, 3, -1), order="F"), (1, 0, 2, 3))

    def get_training_data(self):
        return self.X_train, self.Y_train, self.y_train

    def get_validation_data(self):
        return self.X_val, self.Y_val, self.y_val

    def get_test_data(self):
        return self.X_test, self.Y_test, self.y_test


# ---- CNN  -----


class CNN:
    def __init__(
        self,
        f,
        n_filters,
        hidden_dim,
        stride=4,
        num_classes=10,
        F=None,
        bF=None,
        W1=None,
        b1=None,
        W2=None,
        b2=None,
        random_seed=42,
    ):
        np.random.seed(random_seed)
        # Store parameters
        self.f = f
        self.out_h = (32 - f) // stride + 1
        self.n_p = self.out_h * self.out_h
        self.n_filters = n_filters
        self.hidden_dim = hidden_dim
        self.stride = stride
        self.num_classes = num_classes

        # Gradients placeholders
        self.dL_dF = np.random.randn(f * f * 3, n_filters)
        self.dL_dbF = np.random.randn(n_filters, 1)
        self.dL_dW1 = np.random.randn(
            hidden_dim, ((32 - f) // stride + 1) ** 2 * n_filters
        )
        self.dL_db1 = np.random.randn(hidden_dim, 1)
        self.dL_dW2 = np.random.randn(num_classes, hidden_dim)
        self.dL_db2 = np.random.randn(num_classes, 1)

        # Initialize weights
        self.conv_dim = ((32 - self.f) // self.stride + 1) ** 2 * self.n_filters

        self.F = (
            (
                np.random.randn(self.f * self.f * 3, self.n_filters)
                * np.sqrt(2 / (self.f * self.f * 3))
            ).astype(np.float32)
            if F is None
            else F.reshape(self.f * self.f * 3, self.n_filters, order="C")
        )
        self.bF = np.zeros((self.n_filters, 1)).astype(np.float32) if bF is None else bF
        self.W1 = (
            (
                np.random.randn(self.hidden_dim, self.conv_dim)
                * np.sqrt(2 / self.conv_dim)
            ).astype(np.float32)
            if W1 is None
            else W1
        )
        self.b1 = (
            np.zeros((self.hidden_dim, 1)).astype(np.float32) if b1 is None else b1
        )
        self.W2 = (
            (
                np.random.randn(self.num_classes, self.hidden_dim)
                * np.sqrt(2 / self.hidden_dim)
            ).astype(np.float32)
            if W2 is None
            else W2
        )
        self.b2 = (
            np.zeros((self.num_classes, 1)).astype(np.float32) if b2 is None else b2
        )

    # Compute MX matrix for convolution
    def get_MX(self, X):
        # _, _, _, n = X.shape
        # n_patches = ((32 - self.f) // self.stride + 1) * (
        #     (32 - self.f) // self.stride + 1
        # )

        # # Compute MX
        # MX = np.zeros((n_patches, self.f * self.f * 3, n))
        # for i in range(n):
        #     row_l = 0
        #     for y in range(0, 32 - self.f + 1, self.stride):
        #         for x in range(0, 32 - self.f + 1, self.stride):
        #             X_patch = X[y : y + self.f, x : x + self.f, :, i]
        #             MX[row_l, :, i] = X_patch.reshape((self.f * self.f * 3), order="C")
        #             row_l += 1

        # Efficient MX computation using strides
        H, W, C, N = X.shape
        out_h = (H - self.f) // self.stride + 1
        out_w = (W - self.f) // self.stride + 1

        shape = (out_h, out_w, self.f, self.f, C, N)
        strides = (
            X.strides[0] * self.stride,
            X.strides[1] * self.stride,
            X.strides[0],
            X.strides[1],
            X.strides[2],
            X.strides[3],
        )
        patches = np.lib.stride_tricks.as_strided(X, shape=shape, strides=strides)
        MX = patches.reshape(out_h * out_w, self.f * self.f * C, N)

        return MX.astype(np.float32)

    # Efficient convolution using matrix multiplication
    def _conv_step(self, MX):
        # Compute dimensions
        _, _, n = MX.shape
        out_h = (32 - self.f) // self.stride + 1
        out_w = (32 - self.f) // self.stride + 1
        conv_outputs_mat = np.einsum("ijn, jl ->iln", MX, self.F, optimize=True)
        conv_outputs_mat += self.bF[np.newaxis, :, :]
        return conv_outputs_mat.reshape((out_h, out_w, self.n_filters, n), order="C")

    def _flat_step(self, conv_out):
        _, _, _, n = conv_out.shape
        return conv_out.reshape(-1, n)

    # Fully connected Step
    def _fc_step(self, conv_flat):
        X1 = ReLU(self.W1 @ ReLU(conv_flat) + self.b1)
        p = softmax(self.W2 @ X1 + self.b2)
        return X1, p

    # Normal Forward pass
    def forward(self, MX):
        conv_out = self._conv_step(MX)
        conv_flat = self._flat_step(conv_out)
        _, p = self._fc_step(conv_flat)
        return p

    # ---- Backward Functions -----
    def _update_grads(self, MX, Y, lam=0.0, label_smoothing=0.0):
        _, _, n = MX.shape

        # Forward
        conv_out = self._conv_step(MX)
        conv_flat = self._flat_step(conv_out)
        X1, p = self._fc_step(conv_flat)

        # Backward
        # Fully Connected Layers
        dL_ds = p - apply_label_smoothing(Y, label_smoothing)
        self.dL_dW2 = (dL_ds @ X1.T) / n
        self.dL_db2 = np.sum(dL_ds, axis=1, keepdims=True) / n
        dL_X1 = self.W2.T @ dL_ds
        dL_X1[X1 <= 0] = 0
        self.dL_dW1 = (dL_X1 @ ReLU(conv_flat).T) / n
        self.dL_db1 = np.sum(dL_X1, axis=1, keepdims=True) / n

        # Convolutional Layer
        dL_dh = self.W1.T @ dL_X1
        dL_dh[conv_flat <= 0] = 0
        GG = dL_dh.reshape(self.n_p, self.n_filters, n, order="C")
        MXt = np.transpose(MX, (1, 0, 2))
        self.dL_dF = np.einsum("ijn, jln -> il", MXt, GG, optimize=True) / n
        self.dL_dbF = (np.sum(GG, axis=(0, 2)).reshape(self.n_filters, 1)) / n

        # Regularization
        self.dL_dF += lam * self.F / n
        self.dL_dW1 += lam * self.W1 / n
        self.dL_dW2 += lam * self.W2 / n

    def backward(self, MX, Y, lam=0.0, learning_rate=0.01, label_smoothing=0.0):
        """
        Perform a backward pass and update parameters.
        """
        # compute and store grads
        self._update_grads(MX, Y, lam, label_smoothing)
        # apply parameter updates
        self.F -= learning_rate * self.dL_dF
        self.bF -= learning_rate * self.dL_dbF
        self.W1 -= learning_rate * self.dL_dW1
        self.b1 -= learning_rate * self.dL_db1
        self.W2 -= learning_rate * self.dL_dW2
        self.b2 -= learning_rate * self.dL_db2

    def compute_loss(self, p, y, lam=0.0, label_smoothing=0.0):
        n = p.shape[1]
        cross_entropy = (
            -np.sum(
                apply_label_smoothing(y, label_smoothing) * np.log(p + label_smoothing)
            )
            / n
        )
        reg_term = (
            lam
            * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.F**2))
            / (2 * n)
        )
        return cross_entropy + reg_term

    import numpy as np


# ---- Tests -----


def test_convolution():
    # Load data
    X = load_data["X"]
    Fs = load_data["Fs"]
    targets = load_data["conv_outputs"]
    X_ims = np.transpose(X.reshape((32, 32, 3, 5), order="F"), (1, 0, 2, 3))
    assert (
        X_ims.shape == (32, 32, 3, 5)
        and X.shape == (3072, 5)
        and Fs.shape == (4, 4, 3, 2)
    ), "Wrong shape"

    # Execute convolutions using the CNN helpers
    # instantiate a minimal CNN to reuse its MX/conv code
    net = CNN(f=Fs.shape[0], n_filters=Fs.shape[3], hidden_dim=1, F=Fs)
    MX = net.get_MX(X=X_ims)
    prediction = net._conv_step(MX)

    # Compare convolutions
    assert np.allclose(prediction, targets), (
        "Convolution outputs do not match expected values"
    )


def test_forward():
    # Input
    conv_flat = load_data["conv_flat"]

    # CNN parameters
    W1 = load_data["W1"]
    b1 = load_data["b1"]
    W2 = load_data["W2"]
    b2 = load_data["b2"]

    # Outputs
    X1_target = load_data["X1"]
    p_target = load_data["P"]

    # Build a dummy CNN with the correct hidden dimension
    net = CNN(
        f=4,
        n_filters=conv_flat.shape[0] // ((32 // 4) ** 2),
        hidden_dim=X1_target.shape[1],
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
    )

    # Use the CNN method that consumes conv_flat directly
    X1, p = net._fc_step(conv_flat)

    X1_expected = np.squeeze(X1_target, axis=0)
    assert np.allclose(p, p_target), "Probabilities do not match expected values"
    assert np.allclose(X1, X1_expected), "X1 does not match expected values"


def test_backward():
    X = load_data["X"]
    Fs = load_data["Fs"]
    Y = load_data["Y"]
    grad_Fs = load_data["grad_Fs_flat"]
    X_ims = np.transpose(X.reshape((32, 32, 3, 5), order="F"), (1, 0, 2, 3))

    # CNN parameters
    W1 = load_data["W1"]
    b1 = load_data["b1"]
    W2 = load_data["W2"]
    b2 = load_data["b2"]
    Fs = load_data["Fs"]
    X1_target = load_data["X1"]

    # Build a dummy CNN with the correct hidden dimension
    net = CNN(
        f=4,
        n_filters=Fs.shape[3],
        hidden_dim=X1_target.shape[1],
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        F=Fs,
    )

    MX = net.get_MX(X_ims)
    net.backward(MX=MX, Y=Y)
    assert np.allclose(net.dL_dF, grad_Fs), (
        "Filter gradients do not match expected values"
    )


def test_pytorch():
    # Input
    X = load_data["X"]
    Fs = load_data["Fs"]
    Y = load_data["Y"]

    # CNN parameters
    W1 = load_data["W1"]
    b1 = load_data["b1"]
    W2 = load_data["W2"]
    b2 = load_data["b2"]

    # Build a dummy CNN with the correct hidden dimension
    net = CNN(
        f=4,
        n_filters=Fs.shape[3],
        hidden_dim=W1.shape[0],
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        F=Fs,
    )

    dL_dF_torch, dL_dW1_torch, dL_db1_torch, dL_dW2_torch, dL_db2_torch = (
        compute_grads_with_torch(X, Y, net)
    )

    X = np.transpose(X.reshape((32, 32, 3, 5), order="F"), (1, 0, 2, 3))
    MX = net.get_MX(X)
    net.backward(MX, Y)

    assert np.allclose(net.dL_dF, dL_dF_torch), (
        "Filter gradients do not match implemented gradients"
    )
    assert np.allclose(net.dL_dW1, dL_dW1_torch), (
        "W1 gradients do not match implemented gradients"
    )
    assert np.allclose(net.dL_db1, dL_db1_torch), (
        "b1 gradients do not match implemented gradients"
    )
    assert np.allclose(net.dL_dW2, dL_dW2_torch), (
        "W2 gradients do not match implemented gradients"
    )
    assert np.allclose(net.dL_db2, dL_db2_torch), (
        "b2 gradients do not match implemented gradients"
    )


def compute_grads_with_torch(X, Y_onehot, net):
    dtype = torch.float64

    n = X.shape[1]
    f = net.f
    nf = net.n_filters
    oh = (32 - f) // net.stride + 1

    F = torch.from_numpy(net.F).to(dtype).clone().detach().requires_grad_(True)
    W1 = torch.from_numpy(net.W1).to(dtype).clone().detach().requires_grad_(True)
    b1 = torch.from_numpy(net.b1).to(dtype).clone().detach().requires_grad_(True)
    W2 = torch.from_numpy(net.W2).to(dtype).clone().detach().requires_grad_(True)
    b2 = torch.from_numpy(net.b2).to(dtype).clone().detach().requires_grad_(True)

    X_ims_np = np.transpose(X.reshape(32, 32, 3, n, order="F"), (1, 0, 2, 3))
    X_ims = torch.from_numpy(X_ims_np).to(dtype)

    MX = torch.zeros((oh * oh, f * f * 3, n), dtype=dtype)
    row = 0
    for y0 in range(0, 32 - f + 1, net.stride):
        for x0 in range(0, 32 - f + 1, net.stride):
            patch = X_ims[y0 : y0 + f, x0 : x0 + f, :, :]
            MX[row, :, :] = patch.reshape(f * f * 3, n)
            row += 1

    conv_mat = torch.einsum("ijn,jl->iln", MX, F)
    conv_out = conv_mat.reshape(oh, oh, nf, n)

    conv_flat = conv_out.reshape(-1, n)
    relu = torch.nn.ReLU()
    conv_flat = relu(conv_flat)

    hidden = relu(W1 @ conv_flat + b1)

    scores = W2 @ hidden + b2

    P = torch.softmax(scores, dim=0)

    y_t = torch.from_numpy(Y_onehot).to(dtype)
    y_int = torch.argmax(y_t, dim=0)

    loss = torch.mean(-torch.log(P[y_int, torch.arange(n)]))

    loss.backward()
    return (
        F.grad.detach().cpu().numpy(),
        W1.grad.detach().cpu().numpy(),
        b1.grad.detach().cpu().numpy(),
        W2.grad.detach().cpu().numpy(),
        b2.grad.detach().cpu().numpy(),
    )


# ---- Learning Rate Scheduler -----


class LearningRateScheduler:
    def __init__(self, n_min, n_max, step_size, n_cycles, increasing=True):
        self.n_min = n_min
        self.n_max = n_max
        self.step_size = step_size
        self.n_cycles = n_cycles
        self.increasing = increasing
        self.lr_functions = {
            True: self._cyclical_learning_rate_increasing,
            False: self._cyclical_learning_rate,
        }
        self.cycle_end_iters = {
            True: np.cumsum([2 * self.step_size * (2**i) for i in range(self.n_cycles)])
            .astype(int)
            .tolist(),
            False: np.cumsum([2 * self.step_size] * self.n_cycles).astype(int).tolist(),
        }
        self.cycle_mid_iters = {
            True: (
                np.cumsum([2 * self.step_size * (2**i) for i in range(self.n_cycles)])
                - np.array([self.step_size * (2**i) for i in range(self.n_cycles)])
            )
            .astype(int)
            .tolist(),
            False: (
                np.cumsum([2 * self.step_size] * self.n_cycles)
                - np.array([self.step_size] * self.n_cycles)
            )
            .astype(int)
            .tolist(),
        }

    def get_lr(self, iteration):
        return self.lr_functions[self.increasing](
            self.n_min, self.n_max, self.step_size, iteration
        )

    def get_total_iterations(self):
        multiplier = (1 << self.n_cycles) - 1 if self.increasing else self.n_cycles
        total_iterations = 2 * self.step_size * multiplier
        last_iter = self.cycle_end_iters[self.increasing][-1]
        assert total_iterations == last_iter, (
            "Total iterations do not match the last cycle end iteration."
        )
        return total_iterations

    def get_cycle_mid_end_iters(self):
        return np.sort(
            np.array(
                self.cycle_mid_iters[self.increasing]
                + self.cycle_end_iters[self.increasing]
            )
        ).tolist()

    def _cyclical_learning_rate(self, n_min, n_max, step_size, iteration):
        cycle = np.floor(1 + iteration / (2 * step_size))
        x = np.abs(iteration / step_size - 2 * cycle + 1)
        lr = n_min + (n_max - n_min) * np.maximum(0, (1 - x))
        return lr

    def _cyclical_learning_rate_increasing(self, n_min, n_max, step_size, iteration):
        remaining = iteration
        cycle = 0
        current_step = step_size
        cycle_length = current_step * 2

        while remaining >= cycle_length:
            remaining -= cycle_length
            cycle += 1
            current_step = step_size * (2**cycle)
            cycle_length = current_step * 2

        progress = remaining / current_step
        lr = (
            n_min + (n_max - n_min) * progress
            if progress <= 1.0
            else n_max - (n_max - n_min) * (progress - 1.0)
        )
        return lr


# ---- Experiment  -----


@dataclass
class ExperimentConfig:
    # Model parameters
    f: int = 4
    n_filters: int = 10
    hidden_dim: int = 50
    dataset: str = "CIFAR10"
    random_seed: int = 42

    # Cyclic Learning Rate parameters
    increasing: bool = False
    n_cycles: int = 3
    step: int = 800
    eta_min: float = 1e-5
    eta_max: float = 1e-1
    batch_size: int = 100
    iter_eval: int = 800

    # Regularization parameter
    lam: float = 0.003

    # Label smoothing
    label_smoothing: float = 0.0


class ExperimentLogger:
    def __init__(self, name: str):
        self.id = f"{name}_{int(time.time())}"
        self.path = f"{self.id}.jsonl"
        self._create_jsonl_log_file()

    def _create_jsonl_log_file(self):
        with open(SUMMARY_PATH / self.path, "w") as f:
            f.write("")
        pprint.pprint(f"[LOGGER]: Created log file at {self.path}")

    def log(self, **kwargs):
        with open(SUMMARY_PATH / self.path, "a") as f:
            json_line = json.dumps({**kwargs, "id": self.id})
            f.write(json_line + "\n")


class Experiment:
    def __init__(self, config: ExperimentConfig, logger: ExperimentLogger):
        self.config = config
        self.logger = logger
        self.cnn = CNN(
            f=config.f,
            n_filters=config.n_filters,
            hidden_dim=config.hidden_dim,
            random_seed=config.random_seed,
        )

    def _train_model(self, dataset: CIFAR10Dataset):
        start_time = time.time()

        # Get data
        X_train, Y_train, _ = dataset.get_training_data()
        X_test, Y_test, _ = dataset.get_test_data()
        n_train, n_test = X_train.shape[3], X_test.shape[3]

        # Shuffle data
        indices_train, indices_test = np.arange(n_train), np.arange(n_test)
        np.random.shuffle(indices_test), np.random.shuffle(indices_train)
        X_train, Y_train = (X_train[:, :, :, indices_train], Y_train[:, indices_train])
        X_test, Y_test = (X_test[:, :, :, indices_test], Y_test[:, indices_test])

        # Initialize learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            n_min=self.config.eta_min,
            n_max=self.config.eta_max,
            step_size=self.config.step,
            increasing=self.config.increasing,
            n_cycles=self.config.n_cycles,
        )
        iterations = lr_scheduler.get_total_iterations()
        cycle_iters = lr_scheduler.get_cycle_mid_end_iters()

        for it in tqdm(range(iterations)):
            # Mini-batch
            start = (it * self.config.batch_size) % n_train
            end = min(start + self.config.batch_size, n_train)
            X_train_batch, Y_train_batch = (
                X_train[:, :, :, start:end],
                Y_train[:, start:end],
            )

            # Forward + backward
            lr = lr_scheduler.get_lr(it)
            MX_train_batch = self.cnn.get_MX(X_train_batch)
            self.cnn.backward(
                MX_train_batch,
                Y_train_batch,
                lam=self.config.lam,
                learning_rate=lr,
                label_smoothing=self.config.label_smoothing,
            )

            if (it + 1) in cycle_iters or it == 0:
                current_cycle = next(
                    i + 1
                    for i, end_iter in enumerate(cycle_iters)
                    if it + 1 <= end_iter
                )
                avg_train_loss, avg_train_acc = self._compute_acc_loss(
                    X_train, Y_train, n_train // self.config.batch_size
                )
                avg_test_loss, avg_test_acc = self._compute_acc_loss(
                    X_test, Y_test, n_test // self.config.batch_size
                )
                self.logger.log(
                    config=vars(self.config),
                    cycle=current_cycle,
                    iteration=it + 1,
                    train_loss=avg_train_loss,
                    train_acc=avg_train_acc,
                    test_loss=avg_test_loss,
                    test_acc=avg_test_acc,
                    time_elapsed=time.time() - start_time,
                )

    def _compute_acc_loss(self, X, Y, n_batches):
        total_loss, total_acc = 0.0, 0.0

        for j in range(n_batches):
            start = j * self.config.batch_size
            end = start + self.config.batch_size
            X_batch = X[:, :, :, start:end]
            Y_batch = Y[:, start:end]

            MX_batch = self.cnn.get_MX(X_batch)
            p_val = self.cnn.forward(MX_batch)
            total_loss += self.cnn.compute_loss(
                p_val,
                Y_batch,
                self.config.lam,
                label_smoothing=self.config.label_smoothing,
            )
            total_acc += compute_accuracy(p_val, np.argmax(Y_batch, axis=0))

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches * 100

        return avg_loss, avg_acc

    def run(self):
        if self.config.dataset == "CIFAR10":
            dataset = CIFAR10Dataset()
            self._train_model(dataset)
        else:
            raise NotImplementedError("Dataset not implemented yet.")


# ---- Plotting Functions -----

# Global variables
select_var = {
    "iterations": 0,
    "train_loss": 1,
    "train_acc": 2,
    "test_loss": 3,
    "test_acc": 4,
    "time_elapsed": 5,
}


def select_iterations_losses_and_acc(df: pd.DataFrame):
    iterations, train_loss, train_acc, test_loss, test_acc, time_elapsed = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i in range(len(df)):
        row = df.iloc[i]
        iterations.append(row["iteration"])
        train_loss.append(row["train_loss"])
        train_acc.append(row["train_acc"])
        test_loss.append(row["test_loss"])
        test_acc.append(row["test_acc"])
        time_elapsed.append(row["time_elapsed"])
    return iterations, train_loss, train_acc, test_loss, test_acc, time_elapsed


# Exercise 3
arch1 = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch1_1761014439.jsonl", lines=True)
)
arch2 = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch2_1761014453.jsonl", lines=True)
)
arch3 = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch3_1761014494.jsonl", lines=True)
)
arch4 = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch4_1761014626.jsonl", lines=True)
)

# Train for longer
arch2_long = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch2_long_1761014970.jsonl", lines=True)
)
arch3_long = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch3_long_1761015113.jsonl", lines=True)
)
arch2_long_more_filters = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch2_long_more_filters_1761016822.jsonl", lines=True)
)

# Exercise 4
arch5_baseline = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch5_baseline_1761019769.jsonl", lines=True)
)
arch5_label_smoothing = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch5_label_smoothing_1761021952.jsonl", lines=True)
)


def plot_curves(archs, archs_names=["arch1", "arch2", "arch3", "arch4"]):
    for i in range(len(archs)):
        arch_stats = archs[i]
        plt.plot(
            arch_stats[select_var["iterations"]],
            arch_stats[select_var["test_acc"]],
            "g",
            label="Test Accuracy",
        )
        plt.plot(
            arch_stats[select_var["iterations"]],
            arch_stats[select_var["train_acc"]],
            "b",
            label="Train Accuracy",
        )
        plt.xlabel("Iterations")
        plt.ylim(0, 100)
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(f"{archs_names[i]} Accuracy")
        plt.savefig(REPORT_IMG_DIR / f"accuracy_arch_{archs_names[i]}.pdf")
        plt.clf()

        arch_stats = archs[i]
        plt.plot(
            arch_stats[select_var["iterations"]],
            arch_stats[select_var["test_loss"]],
            "g",
            label="Test Loss",
        )
        plt.plot(
            arch_stats[select_var["iterations"]],
            arch_stats[select_var["train_loss"]],
            "b",
            label="Train Loss",
        )
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.ylim(0, 3)
        plt.legend()
        plt.title(f"{archs_names[i]} Loss")
        plt.savefig(REPORT_IMG_DIR / f"loss_arch_{archs_names[i]}.pdf")
        plt.clf()


def plot_compare_times(archs):
    archs_names = ["arch1", "arch2", "arch3", "arch4"]
    counts = [
        archs[0][select_var["time_elapsed"]][-1],
        archs[1][select_var["time_elapsed"]][-1],
        archs[2][select_var["time_elapsed"]][-1],
        archs[3][select_var["time_elapsed"]][-1],
    ]
    colors = ["red", "blue", "orange", "green"]

    plt.bar(archs_names, counts, color=colors, label=archs_names)
    plt.xlabel("Architecture")
    plt.ylabel("Seconds")
    plt.title("Training time")
    plt.savefig(REPORT_IMG_DIR / "compare_times.pdf")
    plt.clf()


if __name__ == "__main__":
    import argparse

    np.random.seed(42)

    parser = argparse.ArgumentParser(description="CNN Experiment Runner")
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help="Architecture name for the experiment (e.g., 'arch1', 'arch2')",
    )

    args = parser.parse_args()
    arch_name = args.arch

    # Tests
    if arch_name == "tests":
        test_convolution()
        test_forward()
        test_backward()
        test_pytorch()
        print("All tests passed!")
        exit(0)

    # Experiment 1
    elif arch_name == "arch1":
        config = ExperimentConfig(
            f=2,
            n_filters=3,
            hidden_dim=50,
        )

    # Experiment 2
    elif arch_name == "arch2":
        config = ExperimentConfig(
            f=4,
            n_filters=10,
            hidden_dim=50,
        )

    # Experiment 3
    elif arch_name == "arch3":
        config = ExperimentConfig(
            f=8,
            n_filters=40,
            hidden_dim=50,
        )

    # Experiment 4
    elif arch_name == "arch4":
        config = ExperimentConfig(
            f=16,
            n_filters=160,
            hidden_dim=50,
        )

    # Experiment 2 Longer with increasing LR
    elif arch_name == "arch2_long":
        config = ExperimentConfig(
            f=4,
            n_filters=10,
            hidden_dim=50,
            increasing=True,
        )

    # Experiment 3 Longer with Increasing LR
    elif arch_name == "arch3_long":
        config = ExperimentConfig(
            f=8,
            n_filters=40,
            hidden_dim=50,
            increasing=True,
        )

    # Experiment 2 Longer with more n_filters
    elif arch_name == "arch2_long_more_filters":
        config = ExperimentConfig(
            f=4,
            n_filters=40,
            hidden_dim=50,
            increasing=True,
        )

    # Experiment 5: Label Smoothing Baseline
    elif arch_name == "arch5_baseline":
        config = ExperimentConfig(
            f=4,
            n_filters=40,
            hidden_dim=300,
            increasing=False,
            n_cycles=4,
            lam=0.0025,
            label_smoothing=0.0,
        )

    # Experiment 5: Label Smoothing
    elif arch_name == "arch5_label_smoothing":
        config = ExperimentConfig(
            f=4,
            n_filters=40,
            hidden_dim=300,
            increasing=False,
            n_cycles=4,
            lam=0.0025,
            label_smoothing=0.1,
        )

    # Plot all curves
    elif arch_name == "plot_all":
        plot_curves([arch1, arch2, arch3, arch4])
        plot_compare_times([arch1, arch2, arch3, arch4])
        plot_curves(
            [arch2_long, arch3_long, arch2_long_more_filters],
            archs_names=["arch2_long", "arch3_long", "arch2_long_more_filters"],
        )
        plot_curves(
            [arch5_baseline, arch5_label_smoothing],
            archs_names=["arch5_baseline", "arch5_label_smoothing"],
        )

    else:
        raise ValueError(f"Unknown architecture name: {arch_name}")

    logger = ExperimentLogger(arch_name)
    experiment = Experiment(config, logger)
    experiment.run()
