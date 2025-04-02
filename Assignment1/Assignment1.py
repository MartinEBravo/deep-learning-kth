import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import Tensor

import pickle
from typing import Tuple


def display_cifar_10_examples(
    cifar_dir: str = "./Datasets/cifar-10-batches-py/data_batch_1", ni: int = 5
) -> None:
    # Load a batch of training data
    with open(cifar_dir, "rb") as fo:
        dict = pickle.load(fo, encoding="bytes")
    # Extract the image data and cast to float from the dict dictionary
    X = dict[b"data"].astype(np.float64) / 255.0
    X = X.transpose()
    nn = X.shape[1]
    # Reshape each image from a column vector to a 3d array
    X_im = X.reshape((32, 32, 3, nn), order="F")
    X_im = np.transpose(X_im, (1, 0, 2, 3))
    # Display the first 5 images
    fig, axs = plt.subplots(1, 5, figsize=(10, 5))
    for i in range(ni):
        axs[i].imshow(X_im[:, :, :, i])
        axs[i].axis("off")
    plt.show()


def load_batch(cifar_dir: str) -> Tuple[Tensor, Tensor, Tensor]:
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
    d = 1
    n = 1
    K = 1
    X = torch.Tensor()
    Y = torch.Tensor()
    y = torch.Tensor()

    assert X.shape == (d, n), "Dimensions invalid"
    assert Y.shape == (K, n), "Dimensions invalid"
    assert y.shape == (n,), "Dimensions invalid"
    return X, Y, y


if __name__ == "__main__":
    # Visualizing dataset
    train_dir = "./Datasets/cifar-10-batches-py/data_batch_1"
    visualization_dir = "./Datasets/cifar-10-batches-py/data_batch_2"
    test_dir = "./Datasets/cifar-10-batches-py/test_batch"
    display_cifar_10_examples()
