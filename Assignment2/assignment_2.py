import os

import numpy as np
import pickle
import torch


# Global variable
eps = 1e-9


# Create reports directory
if not os.path.exists("reports"):
    os.makedirs("reports")

# Create imgs directory
if not os.path.exists("reports/imgs"):
    os.makedirs("reports/imgs")


def compute_grads_with_torch(X, y, network_params, lam):
    # torch requires arrays to be torch tensors
    Xt = torch.from_numpy(X)
    yt = torch.from_numpy(y)

    # will be computing the gradient w.r.t. these parameters
    W1 = torch.tensor(network_params["W"][0], requires_grad=True, dtype=torch.float64)
    b1 = torch.tensor(network_params["b"][0], requires_grad=True, dtype=torch.float64)
    W2 = torch.tensor(network_params["W"][1], requires_grad=True, dtype=torch.float64)
    b2 = torch.tensor(network_params["b"][1], requires_grad=True, dtype=torch.float64)

    n = X.shape[1]

    s1 = torch.matmul(W1, Xt) + b1
    h = torch.relu(s1)
    s2 = torch.matmul(W2, h) + b2

    P = torch.nn.Softmax(dim=0)(s2)

    ## compute the loss
    reg_term = lam * torch.sum(W1**2) + torch.sum(W2**2)
    loss = -torch.sum(yt * torch.log(P + eps)) / n + reg_term

    # compute the backward pass relative to the loss and the named parameters
    loss.backward()

    # extract the computed gradients and make them numpy arrays
    grads = {"W": [0, 0], "b": [0, 0]}
    grads["W"][0] = (
        W1.grad.numpy() if W1.grad is not None else np.zeros_like(W1.detach().numpy())
    )
    grads["b"][0] = (
        b1.grad.numpy() if b1.grad is not None else np.zeros_like(b1.detach().numpy())
    )
    grads["W"][1] = (
        W2.grad.numpy() if W2.grad is not None else np.zeros_like(W2.detach().numpy())
    )
    grads["b"][1] = (
        b2.grad.numpy() if b2.grad is not None else np.zeros_like(b2.detach().numpy())
    )

    return grads


def to_one_hot(y: np.ndarray):
    """
    Converts a numpy array of labels to a one-hot encoded numpy array.
    """
    return np.eye(10)[y]


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


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def softmax(z):
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def relu(z):
    return np.maximum(0, z)


def apply_network(X, net):
    W1 = net["W"][0]
    b1 = net["b"][0]
    W2 = net["W"][1]
    b2 = net["b"][1]

    s1 = W1 @ X + b1
    h = relu(s1)
    s2 = W2 @ h + b2
    P = softmax(s2)
    return P


def compute_loss(p, y, lam, net):
    n = p.shape[1]
    W1 = net["W"][0]
    W2 = net["W"][1]

    cross_entropy = -np.sum(y * np.log(p + eps)) / n
    reg_term = np.sum(W1**2) + np.sum(W2**2)
    J = cross_entropy + lam * reg_term
    return J


def init_params(d: int, m: int):
    rng = np.random.default_rng()
    net_params = {}
    net_params["W"] = []
    net_params["b"] = []
    net_params["W"].append(1 / np.sqrt(d) * rng.standard_normal(size=(m, d)))
    net_params["b"].append(np.zeros((m, 1)))
    net_params["W"].append(1 / np.sqrt(m) * rng.standard_normal(size=(10, m)))
    net_params["b"].append(np.zeros((10, 1)))
    return net_params


def backward_pass(X, Y, P, net, lam):
    W1 = net["W"][0]
    b1 = net["b"][0]
    W2 = net["W"][1]
    b2 = net["b"][1]

    s1 = W1 @ X + b1
    h = relu(s1)
    s2 = W2 @ h + b2
    P = softmax(s2)
    G = P - Y

    n = X.shape[1]
    grads = {}
    grads["W"] = [None, None]
    grads["b"] = [None, None]

    grads["W"][1] = (G @ h.T) / n + 2 * lam * W2
    grads["b"][1] = np.sum(G, axis=1, keepdims=True) / n

    G_hidden = W2.T @ G
    G_hidden[s1 <= 0] = 0

    grads["W"][0] = (G_hidden @ X.T) / n + 2 * lam * W1
    grads["b"][0] = np.sum(G_hidden, axis=1, keepdims=True) / n

    return grads


def compute_distances(my_grads, torch_grads):
    return np.abs(my_grads - torch_grads) / np.maximum(
        eps, np.abs(my_grads) + np.abs(torch_grads)
    )


def testing_grad(X_train, Y_train, y_train):
    d_small = 5
    n_small = 3
    m = 6
    lam = 0

    small_net = init_params(d_small, m)

    X_small = X_train[0:d_small, 0:n_small]
    Y_small = Y_train[:, 0:n_small]

    P = apply_network(X_small, small_net)

    my_grads = backward_pass(X_small, Y_small, P, small_net, lam)
    torch_grads = compute_grads_with_torch(X_small, y_train[0:n_small], small_net, lam)

    dist_W1 = compute_distances(my_grads["W"][0], torch_grads["W"][0])
    dist_b1 = compute_distances(my_grads["b"][0], torch_grads["b"][0])
    dist_W2 = compute_distances(my_grads["W"][1], torch_grads["W"][1])
    dist_b2 = compute_distances(my_grads["b"][1], torch_grads["b"][1])

    print("My grads b2:", my_grads["b"][1])
    print("Torch grads b2:", torch_grads["b"][1])

    assert np.mean(dist_W1) < 1, f"W1 mismatch too large: {np.mean(dist_W1)}"
    assert np.mean(dist_W2) < 1, f"W2 mismatch too large: {np.mean(dist_W2)}"
    assert np.mean(dist_b1) < 1, f"b1 mismatch too large: {np.mean(dist_b1)}"
    assert np.mean(dist_b2) < 1, f"b2 mismatch too large: {np.mean(dist_b2)}"


if __name__ == "__main__":
    X_train, Y_train, y_train = load_batch(
        "./Datasets/cifar-10-batches-py/data_batch_1"
    )
    testing_grad(X_train, Y_train, y_train)
