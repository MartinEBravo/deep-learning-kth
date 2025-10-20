import numpy as np


def ReLU(s):
    return np.maximum(0, s)


def softmax(s):
    """
    Maps an array of logits to a probability distribution.
    """
    return np.exp(s) / np.dot(np.ones(s.shape[0]), np.exp(s))


def compute_accuracy(p, y_true):
    """
    Compute classification accuracy given probabilities and integer labels.
    """
    predictions = np.argmax(p, axis=0)
    return np.mean(predictions == y_true)


def apply_label_smoothing(Y, smoothing):
    """
    Apply label smoothing to one-hot encoded targets.
    """
    if smoothing <= 0.0:
        return Y
    K, n = Y.shape
    smooth_value = smoothing / (K - 1)
    Y_smooth = np.full((K, n), smooth_value, dtype=np.float64)
    target_indices = np.argmax(Y, axis=0)
    Y_smooth[target_indices, np.arange(n)] = 1.0 - smoothing
    return Y_smooth


def to_one_hot(y: np.ndarray):
    """
    Converts a numpy array of labels to a one-hot encoded numpy array.
    """
    return np.eye(10)[y]


def cyclical_learning_rate(n_min, n_max, step_size, iteration):
    cycle = np.floor(1 + iteration / (2 * step_size))
    x = np.abs(iteration / step_size - 2 * cycle + 1)
    lr = n_min + (n_max - n_min) * np.maximum(0, (1 - x))
    return lr


def cyclical_learning_rate_increasing(n_min, n_max, step_size, iteration):
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
    if progress <= 1.0:
        lr = n_min + (n_max - n_min) * progress
    else:
        lr = n_max - (n_max - n_min) * (progress - 1.0)
    return lr
