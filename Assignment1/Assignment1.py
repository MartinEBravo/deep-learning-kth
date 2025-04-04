import copy
import os
import tqdm
import shutil

import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch


# Global variable
eps = 1e-9


# Create reports directory
if not os.path.exists("reports"):
    os.makedirs("reports")

# Clear the reports directory
for filename in os.listdir("reports"):
    file_path = os.path.join("reports", filename)
    try:
        if os.path.isfile(file_path) or os.path.islink(file_path):
            os.unlink(file_path)
        elif os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(f"Failed to delete {file_path}. Reason: {e}")


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
    plt.close()


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

    assert mean_dist_W < 1, f"W mismatch too large: {mean_dist_W}"
    assert mean_dist_b < 1, f"b mismatch too large: {mean_dist_b}"


def mini_batch_GD(
    X_train, Y_train, y_train, X_val, Y_val, y_val, gd_params, net, verbose=False
):
    train_losses = []
    val_losses = []
    n_batch = gd_params["n_batch"]
    eta = gd_params["eta"]
    n_epochs = gd_params["n_epochs"]
    lam = gd_params["lam"]

    n = X_train.shape[1]
    for epoch in tqdm.tqdm(range(n_epochs), desc="Training Progress"):
        for j in range(n // n_batch):
            j_start = j * n_batch
            j_end = (j + 1) * n_batch
            inds = range(j_start, j_end)
            X_batch = X_train[:, inds]
            Y_batch = Y_train[:, inds]

            P = apply_network(X_batch, net)
            # Backward
            grads = backward_pass(X_batch, Y_batch, P, net, lam)

            # Update
            net["W"] -= eta * grads["W"]
            net["b"] -= eta * grads["b"]

        # Compute loss and accuracy for the entire training set
        P = apply_network(X_train, net)
        train_loss = compute_loss(P, Y_train, net, lam)
        train_acc = compute_accuracy(P, y_train)
        train_losses.append(train_loss)

        # Validation
        P_val = apply_network(X_val, net)
        val_loss = compute_loss(P_val, Y_val, net, lam)
        val_acc = compute_accuracy(P_val, y_val)
        val_losses.append(val_loss)

        # Print progress
        if verbose:
            print(
                f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%"
            )

    return net, train_losses, val_losses


def show_learned_matrices(trained_net, name="experiment"):
    Ws = trained_net["W"].transpose().reshape((32, 32, 3, 10), order="F")
    W_im = np.transpose(Ws, (1, 0, 2, 3))
    fig, axs = plt.subplots(1, 10, figsize=(20, 5))
    for i in range(10):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        axs[i].imshow(w_im_norm)
        axs[i].axis("off")
    plt.savefig(f"reports/assignment1_learned_matrices_{name}.png")
    plt.close()


def show_loss_evolution(train_loss, val_loss, gd_params, name="experiment"):
    plt.plot(train_loss, label="Training Loss", color="blue")
    plt.plot(val_loss, label="Validation Loss", color="orange", linestyle="--")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title(
        f"Loss Evolution (eta={gd_params['eta']}, n_epochs={gd_params['n_epochs']}, n_batch={gd_params['n_batch']}, lambda={gd_params['lam']})"
    )
    plt.legend()
    plt.grid(True)
    plt.savefig(f"reports/assignment1_loss_evolution_{name}.png")
    plt.close()


def run_experiment(
    X_train,
    Y_train,
    y_train,
    X_val,
    Y_val,
    y_val,
    X_test,
    Y_test,
    y_test,
    gd_params,
    net,
    name="experiment",
):
    print(f"Running experiment {name} with parameters: {gd_params}")

    trained_net = copy.deepcopy(net)
    trained_net, train_losses, val_losses = mini_batch_GD(
        X_train, Y_train, y_train, X_val, Y_val, y_val, gd_params, trained_net
    )

    # Test the network
    P_test = apply_network(X_test, trained_net)
    test_loss = compute_loss(P_test, Y_test, trained_net)
    test_acc = compute_accuracy(P_test, y_test)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    show_learned_matrices(trained_net, name=name)
    show_loss_evolution(train_losses, val_losses, gd_params, name=name)
    return trained_net, train_losses, val_losses


if __name__ == "__main__":
    # Visualizing dataset
    train_dir = "./Datasets/cifar-10-batches-py/data_batch_1"
    val_dir = "./Datasets/cifar-10-batches-py/data_batch_2"
    test_dir = "./Datasets/cifar-10-batches-py/test_batch"

    show_cifar_10_examples()

    X_train, Y_train, y_train = load_batch(train_dir)
    X_val, Y_val, y_val = load_batch(val_dir)
    X_test, Y_test, y_test = load_batch(test_dir)

    # Normalize X
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

    # Initialize network
    net = init_weights()

    # Testing the network
    p = apply_network(X_train[:, 0:100], net)
    testing_grad(X_train, Y_train, y_train)

    experiments = [
        {"eta": 0.001, "n_epochs": 40, "n_batch": 100, "lam": 0},
        {"eta": 0.01, "n_epochs": 40, "n_batch": 100, "lam": 0},
        {"eta": 0.001, "n_epochs": 40, "n_batch": 10, "lam": 0},
        {"eta": 0.001, "n_epochs": 40, "n_batch": 100, "lam": 1},
    ]

    # Run experiments
    for i, gd_params in enumerate(experiments):
        experiment_name = f"experiment_{i + 1}"
        trained_net, train_losses, val_losses = run_experiment(
            X_train,
            Y_train,
            y_train,
            X_val,
            Y_val,
            y_val,
            X_test,
            Y_test,
            y_test,
            gd_params,
            net,
            name=experiment_name,
        )
    print("All experiments completed.")
    print("Check reports folder for results.")
