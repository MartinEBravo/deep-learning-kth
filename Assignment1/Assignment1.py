import numpy as np
import matplotlib.pyplot as plt

import pickle


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


def to_one_hot(idx: int, length: int = 10):
    array = np.zeros(length)
    array[idx] = 1
    return array


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
    X_train = (X_train - mean_X) / std_X
    X_validation = (X_validation - mean_X) / std_X
    X_test = (X_test - mean_X) / std_X
    return X_train, X_validation, X_test


def init_weights(K: int = 10, d: int = 3072):
    rng = np.random.default_rng()
    BitGen = type(rng.bit_generator)
    seed = 42
    rng.bit_generator.state = BitGen(seed).state
    init_net = {}
    init_net["W"] = 0.01 * rng.standard_normal(size=(K, d))
    init_net["b"] = np.zeros((K, 1))
    return init_net


def softmax(z, K: int = 10):
    return np.exp(z) / (np.ones(K).T * np.exp(z))


def apply_network(X, net):
    W = net["W"]
    b = net["b"]
    return softmax(W * X + b)


if __name__ == "__main__":
    # Visualizing dataset
    train_dir = "./Datasets/cifar-10-batches-py/data_batch_1"
    validation_dir = "./Datasets/cifar-10-batches-py/data_batch_2"
    test_dir = "./Datasets/cifar-10-batches-py/test_batch"

    # display_cifar_10_examples()
    X_train, Y_train, y_train = load_batch(train_dir)
    X_validation, Y_validation, y_validation = load_batch(validation_dir)
    X_test, Y_test, y_test = load_batch(test_dir)

    # Normalize X
    X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)

    # Initialize network
    net = init_weights()

    #
