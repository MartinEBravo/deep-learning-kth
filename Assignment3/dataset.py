import numpy as np
import pickle
from config import DATA_ROOT
from utils import to_one_hot


def load_batch(cifar_dir: str):
    """
    Retrieves the dataset and converts it to Tensors.

    Args:
        filename (str): File path to the Dataset
    Returns:
        X (torch.Tensor): Tensor of size (d,n) of type torch.float32
        Y (torch.Tensor): Tensor of size (K,n) of type torch.float32
        y (torch.Tensor): Tensor of size (n,1) of type torch.float32
    """
    with open(cifar_dir, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    X = dict[b"data"].astype(np.float64) / 255.0
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


def normalize_data(X_train, X_validation, X_test, d: int = 3072):
    mean_X = np.mean(X_train, axis=1).reshape(d, 1)
    std_X = np.std(X_train, axis=1).reshape(d, 1)
    std_X[std_X == 0] = 1.0
    X_train = (X_train - mean_X) / std_X
    X_validation = (X_validation - mean_X) / std_X
    X_test = (X_test - mean_X) / std_X
    return X_train, X_validation, X_test


def prepare_datasets(validation_start: int = 49000):
    """
    Load CIFAR-10 batches, normalize, and reshape for convolutional training.
    """
    train_batch_paths = [
        DATA_ROOT / "cifar-10-batches-py" / f"data_batch_{i}" for i in range(1, 6)
    ]
    test_path = DATA_ROOT / "cifar-10-batches-py" / "test_batch"

    X_train_list, Y_train_list, y_train_list = [], [], []
    for batch_path in train_batch_paths:
        X_batch, Y_batch, y_batch = load_batch(batch_path)
        X_train_list.append(X_batch)
        Y_train_list.append(Y_batch)
        y_train_list.append(y_batch)

    X_train_full = np.concatenate(X_train_list, axis=1)
    Y_train_full = np.concatenate(Y_train_list, axis=1)
    y_train_full = np.concatenate(y_train_list, axis=0)

    X_test, Y_test, y_test = load_batch(test_path)

    X_val = X_train_full[:, validation_start:]
    Y_val = Y_train_full[:, validation_start:]
    y_val = y_train_full[validation_start:]

    X_train = X_train_full[:, :validation_start]
    Y_train = Y_train_full[:, :validation_start]
    y_train = y_train_full[:validation_start]

    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

    def reshape_for_conv(X):
        return np.transpose(X.reshape((32, 32, 3, -1), order="F"), (1, 0, 2, 3))

    data = {
        "X_train": reshape_for_conv(X_train),
        "Y_train": Y_train,
        "y_train": y_train,
        "X_val": reshape_for_conv(X_val),
        "Y_val": Y_val,
        "y_val": y_val,
        "X_test": reshape_for_conv(X_test),
        "Y_test": Y_test,
        "y_test": y_test,
    }

    return data
