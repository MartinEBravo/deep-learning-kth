import copy
import os

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch


# Global variable
eps = 1e-9


# Create reports directory
if not os.path.exists("reports"):
    os.makedirs("reports")


def compute_grads_with_torch(X, y, network_params, lam):
    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)

    # will be computing the gradient w.r.t. these parameters
    W = torch.tensor(network_params["W"], requires_grad=True, dtype=torch.float64)
    b = torch.tensor(network_params["b"], requires_grad=True, dtype=torch.float64)

    N = X.shape[1]

    scores = torch.matmul(W, Xt) + b.unsqueeze(1)
    ## give an informative name to this torch class
    apply_softmax = torch.nn.Softmax(dim=0)

    # apply softmax to each column of scores
    P = apply_softmax(scores)

    ## compute the loss
    reg_term = lam * torch.sum(W**2)
    loss = torch.mean(-torch.log(P[y, np.arange(N)])) + reg_term

    # compute the backward pass relative to the loss and the named parameters
    loss.backward()

    # extract the computed gradients and make them numpy arrays
    grads = {}
    grads["W"] = (
        W.grad.numpy() if W.grad is not None else np.zeros_like(W.detach().numpy())
    )
    grads["b"] = (
        b.grad.numpy() if b.grad is not None else np.zeros_like(b.detach().numpy())
    )

    return grads


def show_cifar_10_examples(
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
    plt.savefig("reports/assignment1_cifar_examples.png")


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
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=0)


def apply_network(X, net):
    W = net["W"]
    b = net["b"]
    return softmax(W @ X + b)


def compute_loss(P, Y, net, lam=0.5):
    D_len = Y.shape[1]
    W = net["W"]
    loss = -np.sum(Y * np.log(P + eps)) / D_len
    reg = lam * np.sum(W * W)
    return loss + reg


def compute_accuracy(P, y):
    y_pred = np.argmax(P, axis=0)
    return np.mean(y_pred == y) * 100


def backward_pass(X, Y, P, net, lam):
    n = X.shape[1]
    W = net["W"]
    G = P - Y
    grads = {}
    dL_dW = (G @ X.T) / n
    dL_db = (G @ np.ones((n, 1))) / n
    grads["W"] = dL_dW + 2 * lam * W
    grads["b"] = dL_db
    return grads


def testing_grad(X_train, Y_train, y_train):
    d_small = 10
    n_small = 3
    lam = 0
    rng = np.random.default_rng()
    small_net = {}
    small_net["W"] = 0.01 * rng.standard_normal(size=(10, d_small))
    small_net["b"] = np.zeros((10, 1))
    X_small = X_train[0:d_small, 0:n_small]
    Y_small = Y_train[:, 0:n_small]
    P = apply_network(X_small, small_net)
    my_grads = backward_pass(X_small, Y_small, P, small_net, lam)
    torch_grads = compute_grads_with_torch(X_small, y_train[0:n_small], small_net, lam)
    dist_W = np.abs(my_grads["W"] - torch_grads["W"]) / np.maximum(
        eps, np.abs(my_grads["W"]) + np.abs(torch_grads["W"])
    )
    dist_b = np.abs(my_grads["b"] - torch_grads["b"]) / np.maximum(
        eps, np.abs(my_grads["b"]) + np.abs(torch_grads["b"])
    )
    mean_dist_W = np.mean(dist_W)
    mean_dist_b = np.mean(dist_b)

    print(f"Gradient check - W error: {mean_dist_W:.2e}, b error: {mean_dist_b:.2e}")

    assert mean_dist_W < 1, f"W mismatch too large: {mean_dist_W}"
    assert mean_dist_b < 1, f"b mismatch too large: {mean_dist_b}"


def mini_batch_GD(X_train, Y_train, y_train, gd_params, net, lam):
    n_batch = gd_params["n_batch"]
    eta = gd_params["eta"]
    n_epochs = gd_params["n_epochs"]

    train_losses = []

    n = X_train.shape[1]

    for epoch in range(n_epochs):
        loss = 0
        acc = 0
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            inds = range(j_start, j_end)
            X_batch = X_train[:, inds]
            Y_batch = Y_train[:, inds]
            y_batch = y_train[inds]

            P = apply_network(X_batch, net)
            loss += compute_loss(P, Y_batch, net, lam)
            acc = compute_accuracy(P, y_batch)

            # Backward
            grads = backward_pass(X_batch, Y_batch, P, net, lam)

            # Update
            net["W"] -= eta * grads["W"]
            net["b"] -= eta * grads["b"]

            train_losses.append(loss)
        print(f"Epoch {epoch + 1}, Loss: {loss:.4f}, Accuracy: {acc:.2f}%")
    return net


def show_learned_matrices(trained_net):
    Ws = trained_net["W"].transpose().reshape((32, 32, 3, 10), order="F")
    W_im = np.transpose(Ws, (1, 0, 2, 3))
    fig, axs = plt.subplots(1, 10, figsize=(20, 5))
    for i in range(10):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        axs[i].imshow(w_im_norm)
        axs[i].axis("off")
    plt.savefig("reports/assignment1_learned_matrices.png")
    plt.show()


if __name__ == "__main__":
    # Visualizing dataset
    train_dir = "./Datasets/cifar-10-batches-py/data_batch_1"
    validation_dir = "./Datasets/cifar-10-batches-py/data_batch_2"
    test_dir = "./Datasets/cifar-10-batches-py/test_batch"

    show_cifar_10_examples()

    X_train, Y_train, y_train = load_batch(train_dir)
    X_validation, Y_validation, y_validation = load_batch(validation_dir)
    X_test, Y_test, y_test = load_batch(test_dir)

    # Normalize X
    X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)

    # Initialize network
    net = init_weights()

    # Testing the network
    p = apply_network(X_train[:, 0:100], net)
    testing_grad(X_train, Y_train, y_train)

    # Training
    gd_params = {"n_batch": 100, "eta": 0.001, "n_epochs": 40}
    lam = 0
    trained_net = copy.deepcopy(net)
    trained_net = mini_batch_GD(X_train, Y_train, y_train, gd_params, trained_net, lam)

    # Show learned matrices
    show_learned_matrices(trained_net)
