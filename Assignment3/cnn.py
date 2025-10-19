import time
import numpy as np
from utils import (
    ReLU,
    softmax,
    compute_accuracy,
    compute_loss,
    cyclical_learning_rate,
    apply_label_smoothing,
    cyclical_learning_rate_increasing,
)
import tqdm
from config import eps


class CNN:
    def __init__(self, f, n_filters, kernels, hidden_dim, stride=4, num_classes=10):
        conv_dim = ((32 - f) // stride + 1) ** 2 * n_filters
        self.F = np.random.randn(f, f, 3, n_filters) * np.sqrt(2 / (f * f * 3))
        self.W1 = np.random.randn(hidden_dim, conv_dim) * np.sqrt(2 / conv_dim)
        self.b1 = np.zeros((hidden_dim, 1))
        self.W2 = np.random.randn(num_classes, hidden_dim) * np.sqrt(2 / hidden_dim)
        self.b2 = np.zeros((num_classes, 1))

        # Store parameters
        self.f = f
        self.n_filters = n_filters
        self.stride = stride
        self.kernels = kernels

    def _get_MX(self, X):
        # Compute dimensions
        H, W, C, n = X.shape
        n_patches = ((H - self.f) // self.stride + 1) * (
            (W - self.f) // self.stride + 1
        )

        # Compute MX
        MX = np.zeros((n_patches, self.f * self.f * C, n))
        for i in range(n):
            row_l = 0
            for y in range(0, H - self.f + 1, self.stride):
                for x in range(0, W - self.f + 1, self.stride):
                    X_patch = X[y : y + self.f, x : x + self.f, :, i]
                    MX[row_l, :, i] = X_patch.reshape((self.f * self.f * C), order="C")
                    row_l += 1

        return MX

    def _conv3d_from_MX(self, MX):
        # Compute dimensions
        n_patches, f_C, n = MX.shape
        f, f, _, n_filters = self.kernels.shape
        out_h = (32 - f) // self.stride + 1
        out_w = (32 - f) // self.stride + 1

        F_all = self.kernels.reshape((f * f * 3, n_filters), order="C")
        conv_outputs_mat = np.einsum("ijn, jl ->iln", MX, F_all, optimize=True)

        return conv_outputs_mat.reshape((out_h, out_w, n_filters, n), order="C")

    def forward(self, X, return_x1=False, conv_flat=False):
        MX = self._get_MX(X)
        if isinstance(conv_flat, bool):
            conv_out = self._conv3d_from_MX(MX, self.kernels, self.stride)
            conv_flat = conv_out.reshape((-1, X.shape[3]))
        conv_flat = ReLU(conv_flat)
        X1 = ReLU(self.W1 @ conv_flat + self.b1)
        p = softmax(self.W2 @ X1 + self.b2)
        if return_x1:
            return X1, p
        return p

    def _update_grads(self, X, Y, lam):
        n = X.shape[3]
        MX = self._get_MX(X)
        conv_out = self._conv3d_from_MX(MX, self.kernels, self.stride)
        relu_mask = conv_out > 0
        conv_out = conv_out * relu_mask
        conv_flat = conv_out.reshape((-1, n))
        X1 = ReLU(self.W1 @ conv_flat + self.b1)
        P = softmax(self.W2 @ X1 + self.b2)

        G = P - Y

        grads = {}

        grads["W2"] = (G @ X1.T) / n
        grads["b2"] = np.sum(G, axis=1, keepdims=True) / n

        G_hidden = self.W2.T @ G
        G_hidden[X1 <= 0] = 0

        grads["W1"] = (G_hidden @ conv_flat.T) / n
        grads["b1"] = np.sum(G_hidden, axis=1, keepdims=True) / n

        G_conv_flat = self.W1.T @ G_hidden
        conv_shape = conv_out.shape
        G_conv_out = G_conv_flat.reshape(conv_shape, order="C")
        G_conv_out *= relu_mask

        f = self.kernels.shape[0]
        C = self.kernels.shape[2]
        n_filters = self.kernels.shape[3]

        grads["F"] = np.zeros_like(self.kernels)

        for i in range(n):
            grad_F_all_i = MX[:, :, i].T @ G_conv_out[:, :, :, i].reshape(
                (-1, n_filters)
            )
            grads["F"] += grad_F_all_i.reshape((f, f, C, n_filters), order="C")

        grads["F"] /= n

        # Regularization
        scale = lam / n
        grads["W1"] += scale * self.W1
        grads["W2"] += scale * self.W2
        grads["F"] += scale * self.kernels
        self.grads = grads

    def backward(self, learning_rate=0.01):
        self._update_grads()
        self.W1 -= learning_rate * self.grads["W1"]
        self.b1 -= learning_rate * self.grads["b1"]
        self.W2 -= learning_rate * self.grads["W2"]
        self.b2 -= learning_rate * self.grads["b2"]
        self.kernels -= learning_rate * self.grads["F"]


def train_model(
    X,
    Y,
    X_val,
    Y_val,
    net_params,
    n_cycles=3,
    batch_size=100,
    step=800,
    n_min=1e-5,
    n_max=1e-1,
    lam=0.0,
    stride=4,
    increasing=False,
    label_smoothing=0.0,
    log_interval=None,
    extra_eval_sets=None,
):
    n = X.shape[3]
    num_batches = int(np.ceil(n / batch_size))
    filter_size = net_params["F"].shape[0]

    Y_true = Y.copy()
    Y_train_used = apply_label_smoothing(Y_true, label_smoothing).astype(np.float64)
    y_true_indices = np.argmax(Y_true, axis=0)

    def build_eval_entry(
        X_eval, Y_eval_raw, label_indices, apply_flag=True, batch_size_eval=500
    ):
        Y_eval_raw = Y_eval_raw.astype(np.float64, copy=True)
        if apply_flag and label_smoothing > 0:
            Y_eval = apply_label_smoothing(Y_eval_raw, label_smoothing)
        else:
            Y_eval = Y_eval_raw
        return {
            "X": X_eval,
            "Y_eval": Y_eval,
            "label_indices": label_indices,
            "batch_size": batch_size_eval,
        }

    eval_sets = {}

    if X_val is not None and Y_val is not None:
        eval_sets["val"] = build_eval_entry(
            X_val,
            Y_val,
            np.argmax(Y_val, axis=0),
            apply_flag=label_smoothing > 0,
            batch_size_eval=500,
        )

    if extra_eval_sets:
        for name, data in extra_eval_sets.items():
            X_extra = data.get("X")
            Y_extra = data.get("Y")
            if X_extra is None or Y_extra is None:
                raise ValueError(f"Evaluation set '{name}' must provide 'X' and 'Y'.")
            labels_extra = data.get("labels")
            if labels_extra is None:
                labels_extra = np.argmax(Y_extra, axis=0)
            batch_size_eval = data.get("batch_size", 500)
            apply_flag = data.get("apply_smoothing", label_smoothing > 0)
            eval_sets[name] = build_eval_entry(
                X_extra,
                Y_extra,
                labels_extra,
                apply_flag=apply_flag,
                batch_size_eval=batch_size_eval,
            )

    indices = np.arange(n)
    np.random.shuffle(indices)
    X = X[:, :, :, indices]
    Y_true = Y_true[:, indices]
    Y_train_used = Y_train_used[:, indices]
    y_true_indices = y_true_indices[indices]

    if not increasing:
        total_steps = n_cycles * step * 2
    else:
        total_steps = 0
        for i in range(n_cycles):
            total_steps += step * (2 ** (i + 1))

    if log_interval is None:
        log_interval = max(1, step // 2)

    time_start = time.time()

    metrics = {
        "train_loss": [],
        "train_acc": [],
        "learning_rates": [],
        "iterations": [],
    }
    for eval_name in eval_sets:
        metrics[f"{eval_name}_loss"] = []
        metrics[f"{eval_name}_acc"] = []

    for iteration in tqdm.tqdm(range(total_steps)):
        i = iteration % num_batches
        start = i * batch_size
        end = min(start + batch_size, n)
        X_batch = X[:, :, :, start:end]
        MX_batch = get_MX(X_batch, filter_size, stride=stride)
        Y_batch = Y_train_used[:, start:end]
        y_batch_indices = y_true_indices[start:end]

        if increasing:
            learning_rate = cyclical_learning_rate_increasing(
                n_min, n_max, step, iteration
            )
        else:
            learning_rate = cyclical_learning_rate(n_min, n_max, step, iteration)

        p = forward_pass(MX_batch, net_params, stride=stride)

        loss = compute_loss(p, Y_batch, lam=lam, net_params=net_params)

        grads = backward_pass(MX_batch, Y_batch, net_params, lam, stride=stride)

        for key in grads:
            net_params[key] -= learning_rate * grads[key]

        should_log = ((iteration + 1) % log_interval == 0) or (
            iteration == total_steps - 1
        )
        if should_log:
            train_acc = compute_accuracy(p, y_batch_indices)

            metrics["train_loss"].append(loss)
            metrics["train_acc"].append(train_acc)
            metrics["learning_rates"].append(learning_rate)
            metrics["iterations"].append(iteration)

            for eval_name, eval_data in eval_sets.items():
                eval_loss, eval_acc = evaluate_dataset(
                    eval_data["X"],
                    eval_data["Y_eval"],
                    eval_data["label_indices"],
                    net_params,
                    lam,
                    stride=stride,
                    batch_size=eval_data.get("batch_size", 500),
                )
                metrics[f"{eval_name}_loss"].append(eval_loss)
                metrics[f"{eval_name}_acc"].append(eval_acc)

    total_time = time.time() - time_start

    return net_params, metrics, total_time


def evaluate_dataset(
    X,
    Y_eval,
    label_indices,
    net_params,
    lam,
    stride=4,
    batch_size=500,
):
    n = X.shape[3]
    total_ce = 0.0
    correct = 0
    processed = 0

    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        X_batch = X[:, :, :, start:end]
        MX_batch = get_MX(X_batch, net_params["F"].shape[0], stride=stride)
        p_batch = forward_pass(MX_batch, net_params, stride=stride)

        targets = Y_eval[:, start:end]
        total_ce += -np.sum(targets * np.log(p_batch + eps))

        preds = np.argmax(p_batch, axis=0)
        correct += np.sum(preds == label_indices[start:end])
        processed += end - start

    data_loss = total_ce / processed
    reg_term = (
        lam
        * (
            np.sum(net_params["W1"] ** 2)
            + np.sum(net_params["W2"] ** 2)
            + np.sum(net_params["F"] ** 2)
        )
        / (2 * processed)
    )
    loss = data_loss + reg_term
    accuracy = correct / processed
    return loss, accuracy


def train_architecture(
    data,
    arch_config,
    training_config,
    name=None,
    label_smoothing=0.0,
    increasing=False,
    log_interval=None,
    monitor_test=False,
):
    stride = arch_config.get("stride", 4)
    f = arch_config["f"]
    n_filters = arch_config["n_filters"]
    hidden_dim = arch_config["hidden_dim"]
    num_classes = training_config.get("num_classes", 10)
    lam = training_config["lam"]

    net_params = initialize_network(
        f, n_filters, hidden_dim, num_classes=num_classes, stride=stride
    )

    extra_eval_sets = {}
    eval_batch_size = training_config.get("eval_batch_size", 500)
    if monitor_test:
        extra_eval_sets["test"] = {
            "X": data["X_test"],
            "Y": data["Y_test"],
            "labels": data["y_test"],
            "apply_smoothing": False,
            "batch_size": eval_batch_size,
        }

    net_params, metrics, total_time = train_model(
        data["X_train"],
        data["Y_train"],
        data["X_val"],
        data["Y_val"],
        net_params,
        n_cycles=training_config["n_cycles"],
        batch_size=training_config["batch_size"],
        step=training_config["step"],
        n_min=training_config["n_min"],
        n_max=training_config["n_max"],
        lam=lam,
        stride=stride,
        increasing=increasing,
        label_smoothing=label_smoothing,
        log_interval=log_interval,
        extra_eval_sets=extra_eval_sets if extra_eval_sets else None,
    )

    test_loss, test_acc = evaluate_dataset(
        data["X_test"],
        data["Y_test"].astype(np.float64, copy=True),
        data["y_test"],
        net_params,
        lam,
        stride=stride,
        batch_size=eval_batch_size,
    )
    val_loss, val_acc = evaluate_dataset(
        data["X_val"],
        data["Y_val"].astype(np.float64, copy=True),
        data["y_val"],
        net_params,
        lam,
        stride=stride,
        batch_size=min(eval_batch_size, data["X_val"].shape[3]),
    )

    result = {
        "name": name or f"f{f}_nf{n_filters}_nh{hidden_dim}",
        "arch": {
            "f": f,
            "n_filters": n_filters,
            "hidden_dim": hidden_dim,
            "stride": stride,
        },
        "training": dict(training_config),
        "label_smoothing": label_smoothing,
        "increasing": increasing,
        "metrics": metrics,
        "train_time_sec": total_time,
        "test_accuracy": test_acc,
        "test_loss": test_loss,
        "val_accuracy": val_acc,
        "val_loss": val_loss,
    }

    return result, net_params
