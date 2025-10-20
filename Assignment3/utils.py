import numpy as np


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


def apply_label_smoothing(Y, smoothing):
    """
    Apply label smoothing to one-hot encoded targets.
    """
    if smoothing <= 0.0:
        return Y
    K, n = Y.shape
    smooth_value = smoothing / (K - 1)
    Y_smooth = np.full((K, n), smooth_value, dtype=np.float32)
    target_indices = np.argmax(Y, axis=0)
    Y_smooth[target_indices, np.arange(n)] = 1.0 - smoothing
    return Y_smooth


def to_one_hot(y: np.ndarray):
    """
    Converts a numpy array of labels to a one-hot encoded numpy array.
    """
    return np.eye(10)[y]
