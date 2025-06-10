import os

import matplotlib.pyplot as plt
import numpy as np
import pickle
import torch
import tqdm
import cv2

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
    z = z - np.max(z, axis=0)
    return np.exp(z) / np.sum(np.exp(z), axis=0)


def relu(z):
    return np.maximum(0, z)


def apply_network(X, net, dropout_rate=0.0, training=True):
    W1 = net["W"][0]
    b1 = net["b"][0]
    W2 = net["W"][1]
    b2 = net["b"][1]

    s1 = W1 @ X + b1
    h = relu(s1)
    if training and dropout_rate > 0:
        dropout_mask = (np.random.rand(*h.shape) > dropout_rate).astype(np.float64)
        h = h * dropout_mask
        h /= 1 - dropout_rate

    s2 = W2 @ h + b2
    P = softmax(s2)
    return P


def compute_cost(p, y, lam, net):
    n = p.shape[1]
    W1 = net["W"][0]
    W2 = net["W"][1]

    cross_entropy = -np.sum(y * np.log(p + eps)) / n
    reg_term = np.sum(W1**2) + np.sum(W2**2)  # L2 Regularization
    J = cross_entropy + lam * reg_term
    return J


def compute_accuracy(p, y):
    return np.sum(np.argmax(p, axis=0) == y) / len(y)


def compute_loss(p, y):
    n = p.shape[1]
    cross_entropy = -np.sum(y * np.log(p + eps)) / n
    return cross_entropy


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

    assert np.mean(dist_W1) < 1, f"W1 mismatch too large: {np.mean(dist_W1)}"
    assert np.mean(dist_W2) < 1, f"W2 mismatch too large: {np.mean(dist_W2)}"
    assert np.mean(dist_b1) < 1, f"b1 mismatch too large: {np.mean(dist_b1)}"
    assert np.mean(dist_b2) < 1, f"b2 mismatch too large: {np.mean(dist_b2)}"

    print("The distance between the gradients computed by the two methods is small.")


def cyclical_learning_rate(n_min, n_max, step_size, epoch):
    cycle = np.floor(1 + epoch / (2 * step_size))
    x = np.abs(epoch / step_size - 2 * cycle + 1)
    lr = n_min + (n_max - n_min) * np.maximum(0, (1 - x))
    return lr


def train_network(
    X_train,
    Y_train,
    y_train,
    X_validation,
    Y_validation,
    y_validation,
    net,
    lam,
    n_min,
    n_max,
    step_size,
    cycles,
    batch_size,
    dropout_rate=0.0,
    use_adam=False,
):
    n_train = X_train.shape[1]

    print(
        f"Training setup: {n_train} training samples, {X_train.shape[0]} features, {cycles} cycles, {step_size} step size, {batch_size} batch size, {dropout_rate} dropout rate, {use_adam} use Adam optimizer"
    )

    n_epochs = int(np.ceil(cycles * 2 * step_size / (X_train.shape[1] // batch_size)))

    train_losses = []
    validation_losses = []
    train_accuracies = []
    validation_accuracies = []
    train_costs = []
    validation_costs = []
    m, v, t = {}, {}, 0

    if use_adam:
        m = {
            "W": [np.zeros_like(net["W"][0]), np.zeros_like(net["W"][1])],
            "b": [np.zeros_like(net["b"][0]), np.zeros_like(net["b"][1])],
        }
        v = {
            "W": [np.zeros_like(net["W"][0]), np.zeros_like(net["W"][1])],
            "b": [np.zeros_like(net["b"][0]), np.zeros_like(net["b"][1])],
        }
        t = 0
    iters = 0
    for _ in tqdm.tqdm(range(n_epochs), desc="Epochs"):
        # Shuffle training data
        perm = np.random.permutation(n_train)
        X_shuffled, Y_shuffled = X_train[:, perm], Y_train[:, perm]

        # Mini-batch training
        for i in range(0, n_train, batch_size):
            lr = cyclical_learning_rate(n_min, n_max, step_size, iters)
            iters += 1
            batch_X = X_shuffled[:, i : i + batch_size]
            batch_Y = Y_shuffled[:, i : i + batch_size]

            P = apply_network(batch_X, net, dropout_rate, training=True)
            grads = backward_pass(batch_X, batch_Y, P, net, lam)

            if use_adam:
                if "t" not in locals():
                    t = 0
                t += 1
                net, m, v = adam_update(net, grads, m, v, t, lr)
            else:
                net["W"][0] -= lr * grads["W"][0]
                net["b"][0] -= lr * grads["b"][0]
                net["W"][1] -= lr * grads["W"][1]
                net["b"][1] -= lr * grads["b"][1]

        # Compute metrics on full datasets
        P_train = apply_network(X_train, net)
        P_val = apply_network(X_validation, net)

        train_loss = compute_loss(P_train, Y_train)
        validation_loss = compute_loss(P_val, Y_validation)

        train_costs.append(compute_cost(P_train, Y_train, lam, net))
        validation_costs.append(compute_cost(P_val, Y_validation, lam, net))

        train_accuracies.append(compute_accuracy(P_train, y_train))
        validation_accuracies.append(compute_accuracy(P_val, y_validation))

        train_losses.append(train_loss)
        validation_losses.append(validation_loss)

    return (
        train_losses,
        validation_losses,
        train_accuracies,
        validation_accuracies,
        train_costs,
        validation_costs,
    )


def test_network(X_test, Y_test, y_test, net, lam):
    P = apply_network(X_test, net)
    accuracy = compute_accuracy(P, y_test)
    loss = compute_loss(P, Y_test)
    cost = compute_cost(P, Y_test, lam, net)
    return accuracy, loss, cost


def flip_images(X):
    """
    Horizontally flips CIFAR-10 images represented as (3072, N) matrix.
    Each image is a 32x32x3 flattened into 3072.
    """
    # Create flip indices once
    aa = np.arange(32).reshape((32, 1))
    bb = np.arange(31, -1, -1).reshape((1, 32))
    ind_flip = (32 * aa + bb).flatten()
    inds_flip = np.concatenate([ind_flip, 1024 + ind_flip, 2048 + ind_flip])
    return X[inds_flip, :]


def data_augmentation(X):
    X_aug = X.copy()
    N = X.shape[1]

    for i in range(N):
        img = X_aug[:, i].reshape(3, 32, 32).transpose(1, 2, 0)  # (32, 32, 3)

        if np.random.rand() > 0.5:
            img = np.fliplr(img)

        tx = np.random.randint(-3, 4)
        ty = np.random.randint(-3, 4)
        M_trans = np.array([[1, 0, tx], [0, 1, ty]], dtype=np.float32)
        for c in range(3):
            img[:, :, c] = cv2.warpAffine(
                img[:, :, c], M_trans, (32, 32), borderMode=cv2.BORDER_REFLECT
            )

        angle = np.random.uniform(-15, 15)
        M_rot = cv2.getRotationMatrix2D((16, 16), angle, 1)
        for c in range(3):
            img[:, :, c] = cv2.warpAffine(
                img[:, :, c], M_rot, (32, 32), borderMode=cv2.BORDER_REFLECT
            )

        X_aug[:, i] = img.transpose(2, 0, 1).reshape(-1)

    return X_aug


def adam_update(
    params, grads, m, v, t, learning_rate, beta1=0.9, beta2=0.999, eps=1e-8
):
    for l_idx in range(2):
        m["W"][l_idx] = beta1 * m["W"][l_idx] + (1 - beta1) * grads["W"][l_idx]
        m["b"][l_idx] = beta1 * m["b"][l_idx] + (1 - beta1) * grads["b"][l_idx]

        v["W"][l_idx] = beta2 * v["W"][l_idx] + (1 - beta2) * (grads["W"][l_idx] ** 2)
        v["b"][l_idx] = beta2 * v["b"][l_idx] + (1 - beta2) * (grads["b"][l_idx] ** 2)

        m_hat_W = m["W"][l_idx] / (1 - beta1**t)
        m_hat_b = m["b"][l_idx] / (1 - beta1**t)
        v_hat_W = v["W"][l_idx] / (1 - beta2**t)
        v_hat_b = v["b"][l_idx] / (1 - beta2**t)

        params["W"][l_idx] -= learning_rate * m_hat_W / (np.sqrt(v_hat_W) + eps)
        params["b"][l_idx] -= learning_rate * m_hat_b / (np.sqrt(v_hat_b) + eps)

    return params, m, v


def plot_costs(
    train_costs,
    validation_costs,
    name,
):
    plt.figure(figsize=(12, 5))
    plt.plot(train_costs, label="Training Cost")
    plt.plot(validation_costs, label="Validation Cost")
    plt.legend()
    plt.savefig(f"reports/imgs/assignment_2_cost_results_{name}.png")


def plot_accuracies(
    train_accuracies,
    validation_accuracies,
    name,
):
    plt.figure(figsize=(12, 5))
    plt.plot(train_accuracies, label="Training Accuracy")
    plt.plot(validation_accuracies, label="Validation Accuracy")
    plt.legend()
    plt.savefig(f"reports/imgs/assignment_2_accuracy_results_{name}.png")


def plot_losses(
    train_losses,
    validation_losses,
    name,
):
    plt.figure(figsize=(12, 5))
    plt.plot(train_losses, label="Training Loss")
    plt.plot(validation_losses, label="Validation Loss")
    plt.legend()
    plt.savefig(f"reports/imgs/assignment_2_loss_results_{name}.png")


def check_gradients_setup():
    X_train, Y_train, y_train = load_batch(
        "./Datasets/cifar-10-batches-py/data_batch_1"
    )
    testing_grad(X_train, Y_train, y_train)


def train_network_setup():
    print("--------- Exercise 3 ---------")
    print("Training with 1 cycle")
    X_train, Y_train, y_train = load_batch(
        "./Datasets/cifar-10-batches-py/data_batch_1"
    )
    X_validation, Y_validation, y_validation = load_batch(
        "./Datasets/cifar-10-batches-py/data_batch_2"
    )
    X_test, Y_test, y_test = load_batch("./Datasets/cifar-10-batches-py/test_batch")

    X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)

    d = X_train.shape[0]
    m = 64
    lam = 0.01
    n_min = 1e-5
    n_max = 1e-1
    batch_size = 100
    step_size = 500
    name = "e3"

    cycles = 1

    net = init_params(d, m)

    (
        train_losses,
        validation_losses,
        train_accuracies,
        validation_accuracies,
        train_costs,
        validation_costs,
    ) = train_network(
        X_train,
        Y_train,
        y_train,
        X_validation,
        Y_validation,
        y_validation,
        net,
        lam,
        n_min,
        n_max,
        step_size,
        cycles,
        batch_size,
    )

    plot_costs(
        train_costs,
        validation_costs,
        name,
    )

    plot_accuracies(
        train_accuracies,
        validation_accuracies,
        name,
    )

    plot_losses(
        train_losses,
        validation_losses,
        name,
    )

    accuracy, loss, cost = test_network(X_test, Y_test, y_test, net, lam)

    print(f"Network performs with accuracy: {accuracy}, loss: {loss}, and cost: {cost}")
    print("--------------------------------")


def train_network_setup_2():
    print("--------- Exercise 4 ---------")
    print("Training with 3 cycles")
    X_train, Y_train, y_train = load_batch(
        "./Datasets/cifar-10-batches-py/data_batch_1"
    )
    X_validation, Y_validation, y_validation = load_batch(
        "./Datasets/cifar-10-batches-py/data_batch_2"
    )
    X_test, Y_test, y_test = load_batch("./Datasets/cifar-10-batches-py/test_batch")

    X_train, X_validation, X_test = normalize_data(X_train, X_validation, X_test)

    d = X_train.shape[0]
    m = 64
    lam = 0.01
    n_min = 1e-5
    n_max = 1e-1
    step_size = 800
    batch_size = 100
    cycles = 3
    name = "e4"

    net = init_params(d, m)

    (
        train_losses,
        validation_losses,
        train_accuracies,
        validation_accuracies,
        train_costs,
        validation_costs,
    ) = train_network(
        X_train,
        Y_train,
        y_train,
        X_validation,
        Y_validation,
        y_validation,
        net,
        lam,
        n_min,
        n_max,
        step_size,
        cycles,
        batch_size,
    )

    plot_costs(
        train_costs,
        validation_costs,
        name,
    )

    plot_accuracies(
        train_accuracies,
        validation_accuracies,
        name,
    )

    plot_losses(
        train_losses,
        validation_losses,
        name,
    )

    accuracy, loss, cost = test_network(X_test, Y_test, y_test, net, lam)

    print(f"Network performs with accuracy: {accuracy}, loss: {loss}, and cost: {cost}")
    print("--------------------------------")


def train_network_setup_3():
    print("--------- Exercise 4 ---------")
    print("Finding the optimal lambda")
    ####### 1. Training with all batches #######
    # Load the dataset
    batch_1 = "./Datasets/cifar-10-batches-py/data_batch_1"
    batch_2 = "./Datasets/cifar-10-batches-py/data_batch_2"
    batch_3 = "./Datasets/cifar-10-batches-py/data_batch_3"
    batch_4 = "./Datasets/cifar-10-batches-py/data_batch_4"
    batch_5 = "./Datasets/cifar-10-batches-py/data_batch_5"
    test_dir = "./Datasets/cifar-10-batches-py/test_batch"

    # Load all batches
    X_train_1, Y_train_1, y_train_1 = load_batch(batch_1)
    X_train_2, Y_train_2, y_train_2 = load_batch(batch_2)
    X_train_3, Y_train_3, y_train_3 = load_batch(batch_3)
    X_train_4, Y_train_4, y_train_4 = load_batch(batch_4)
    X_train_5, Y_train_5, y_train_5 = load_batch(batch_5)

    # Concatenate all batches
    X_train = np.concatenate(
        (X_train_1, X_train_2, X_train_3, X_train_4, X_train_5), axis=1
    )
    Y_train = np.concatenate(
        (Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5), axis=1
    )
    y_train = np.concatenate(
        (y_train_1, y_train_2, y_train_3, y_train_4, y_train_5), axis=0
    )

    # Validation set is the last 5000 samples of the training set
    X_val = X_train[:, -5000:]
    Y_val = Y_train[:, -5000:]
    y_val = y_train[-5000:]
    X_train = X_train[:, :-5000]
    Y_train = Y_train[:, :-5000]
    y_train = y_train[:-5000]

    # Load the test set
    X_test, Y_test, y_test = load_batch(test_dir)
    # Normalize X
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

    # Initialize network
    d = X_train.shape[0]
    n = X_train.shape[1]
    m = 64
    n_min = 1e-5
    n_max = 1e-1
    cycles = 1
    batch_size = 100
    step_size = 2 * np.floor(n / batch_size)

    lambda_values = []
    train_accuracies_values = []
    validation_accuracies_values = []
    test_accuracies_values = []

    l_min = -5
    l_max = -1
    rng = np.random.default_rng()

    for i in range(10):
        log_lambda = l_min + (l_max - l_min) * rng.random()
        lam = 10**log_lambda
        lambda_values.append(lam)

    print(f"Lambda values: {lambda_values}")

    for i in range(10):
        print(f"Training with lambda: {lambda_values[i]}")
        lam = lambda_values[i]
        net = init_params(d, m)
        (
            _,
            _,
            train_accuracies,
            validation_accuracies,
            _,
            _,
        ) = train_network(
            X_train,
            Y_train,
            y_train,
            X_val,
            Y_val,
            y_val,
            net,
            lam,
            n_min,
            n_max,
            step_size,
            cycles,
            batch_size,
        )
        accuracy, _, _ = test_network(X_test, Y_test, y_test, net, lam)
        lambda_values.append(lam)
        train_accuracies_values.append(train_accuracies[-1])
        validation_accuracies_values.append(validation_accuracies[-1])
        test_accuracies_values.append(accuracy)
    # Order based on test accuracy, from highest to lowest
    sorted_indices = np.argsort(test_accuracies_values)[::-1]
    lambda_values = np.array(lambda_values)[sorted_indices]
    train_accuracies_values = np.array(train_accuracies_values)[sorted_indices]
    validation_accuracies_values = np.array(validation_accuracies_values)[
        sorted_indices
    ]
    test_accuracies_values = np.array(test_accuracies_values)[sorted_indices]

    for i in range(10):
        print(
            f"Lambda: {lambda_values[i]}, Train accuracy: {train_accuracies_values[i]}, Validation accuracy: {validation_accuracies_values[i]}, Test accuracy: {test_accuracies_values[i]}"
        )

    print("--------------------------------")

    print(f"Optimal lambda: {lambda_values[0]}")
    n_min = 1e-5
    n_max = 1e-1
    step_size = 800
    cycles = 3
    batch_size = 100
    optimal_lambda = lambda_values[0]
    name = "e4_optimal_lambda"

    ####### 2. Training with optimal lambda #######
    net = init_params(d, m)
    (
        train_losses,
        validation_losses,
        train_accuracies,
        validation_accuracies,
        train_costs,
        validation_costs,
    ) = train_network(
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        net,
        optimal_lambda,
        n_min,
        n_max,
        step_size,
        cycles,
        batch_size,
    )

    accuracy, loss, cost = test_network(X_test, Y_test, y_test, net, optimal_lambda)

    plot_costs(
        train_costs,
        validation_costs,
        name,
    )

    plot_accuracies(
        train_accuracies,
        validation_accuracies,
        name,
    )

    plot_losses(
        train_losses,
        validation_losses,
        name,
    )

    print(f"Network performs with accuracy: {accuracy}, loss: {loss}, and cost: {cost}")


def train_network_setup_4():
    print("--------- Exercise 5 ---------")
    print("Improving results, with more hidden units, dropout, and data augmentation")

    # Load the dataset
    batch_1 = "./Datasets/cifar-10-batches-py/data_batch_1"
    batch_2 = "./Datasets/cifar-10-batches-py/data_batch_2"
    batch_3 = "./Datasets/cifar-10-batches-py/data_batch_3"
    batch_4 = "./Datasets/cifar-10-batches-py/data_batch_4"
    batch_5 = "./Datasets/cifar-10-batches-py/data_batch_5"
    test_dir = "./Datasets/cifar-10-batches-py/test_batch"

    # Load all batches
    X_train_1, Y_train_1, y_train_1 = load_batch(batch_1)
    X_train_2, Y_train_2, y_train_2 = load_batch(batch_2)
    X_train_3, Y_train_3, y_train_3 = load_batch(batch_3)
    X_train_4, Y_train_4, y_train_4 = load_batch(batch_4)
    X_train_5, Y_train_5, y_train_5 = load_batch(batch_5)

    # Concatenate all batches
    X_train = np.concatenate(
        (X_train_1, X_train_2, X_train_3, X_train_4, X_train_5), axis=1
    )
    Y_train = np.concatenate(
        (Y_train_1, Y_train_2, Y_train_3, Y_train_4, Y_train_5), axis=1
    )
    y_train = np.concatenate(
        (y_train_1, y_train_2, y_train_3, y_train_4, y_train_5), axis=0
    )

    # Validation set is the last 5000 samples of the training set
    X_val = X_train[:, -5000:]
    Y_val = Y_train[:, -5000:]
    y_val = y_train[-5000:]
    X_train = X_train[:, :-5000]
    Y_train = Y_train[:, :-5000]
    y_train = y_train[:-5000]

    # Load the test set
    X_test, Y_test, y_test = load_batch(test_dir)

    ####### 1. Data Augmentation #######

    X_train_aug = data_augmentation(X_train)
    Y_train_aug = Y_train.copy()
    y_train_aug = y_train.copy()

    X_train = np.concatenate((X_train, X_train_aug), axis=1)
    Y_train = np.concatenate((Y_train, Y_train_aug), axis=1)
    y_train = np.concatenate((y_train, y_train_aug), axis=0)

    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

    ####### 2. Dropout #######
    dropout_rate = 0.2

    ####### 3. More hidden units #######
    m = 512

    # Training
    d = X_train.shape[0]
    n_min = 1e-5
    n_max = 1e-1
    lam = 0.01
    batch_size = 100
    step_size = X_train.shape[1] // batch_size
    cycles = 5
    name = "data_augmentation_dropout_more_hidden_units"

    net = init_params(d, m)

    (
        train_losses,
        validation_losses,
        train_accuracies,
        validation_accuracies,
        train_costs,
        validation_costs,
    ) = train_network(
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        net,
        lam,
        n_min,
        n_max,
        step_size,
        cycles,
        batch_size,
        dropout_rate,
    )

    plot_costs(train_costs, validation_costs, name)
    plot_accuracies(train_accuracies, validation_accuracies, name)
    plot_losses(train_losses, validation_losses, name)

    accuracy, loss, cost = test_network(X_test, Y_test, y_test, net, lam)

    print(f"Accuracy: {accuracy}, Loss: {loss}, Cost: {cost}")


def train_network_setup_5():
    print("--------- Exercise 6 ---------")

    X_train, Y_train, y_train = load_batch(
        "./Datasets/cifar-10-batches-py/data_batch_1"
    )
    X_val, Y_val, y_val = load_batch("./Datasets/cifar-10-batches-py/data_batch_2")
    X_test, Y_test, y_test = load_batch("./Datasets/cifar-10-batches-py/test_batch")

    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

    print("Improving results with Adam Optimizer")

    # Training
    m = 64
    d = X_train.shape[0]
    n_min = 1e-5
    n_max = 1e-3
    step_size = 500
    cycles = 1
    batch_size = 100
    lam = 0.01
    dropout_rate = 0.0
    name = "Adam Optimizer"
    use_adam = True

    net = init_params(d, m)

    (
        train_losses,
        validation_losses,
        train_accuracies,
        validation_accuracies,
        train_costs,
        validation_costs,
    ) = train_network(
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        net,
        lam,
        n_min,
        n_max,
        step_size,
        cycles,
        batch_size,
        dropout_rate,
        use_adam,
    )

    plot_costs(train_costs, validation_costs, name)
    plot_accuracies(train_accuracies, validation_accuracies, name)
    plot_losses(train_losses, validation_losses, name)

    accuracy, loss, cost = test_network(X_test, Y_test, y_test, net, lam)

    print(f"Accuracy: {accuracy}, Loss: {loss}, Cost: {cost}")


if __name__ == "__main__":
    experiments = [
        {
            "name": "Check gradients setup",
            "function": check_gradients_setup,
        },
        {
            "name": "Train network from figure 3",
            "function": train_network_setup,
        },
        {
            "name": "Train network from figure 4",
            "function": train_network_setup_2,
        },
        {
            "name": "Find optimal lambda",
            "function": train_network_setup_3,
        },
        {
            "name": "Improve results with more hidden units, dropout, and data augmentation",
            "function": train_network_setup_4,
        },
        {
            "name": "Improve results with Adam optimizer",
            "function": train_network_setup_5,
        },
    ]

    decision = input("Do you want to run all experiments? (y/n): ")
    if decision == "y":
        for experiment in experiments:
            print(f"Running {experiment['name']}")
            experiment["function"]()
            print("--------------------------------")
    else:
        for experiment in experiments:
            decision = input(f"Do you want to run {experiment['name']}? (y/n): ")
            if decision == "y":
                print(f"Running {experiment['name']}")
                experiment["function"]()
                print("--------------------------------")
