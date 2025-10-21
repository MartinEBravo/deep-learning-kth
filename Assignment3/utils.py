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
