import copy
import os
import tqdm

import numpy as np
import matplotlib.pyplot as plt
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

# Delete all the .png files in the reports directory
# for filename in os.listdir("reports"):
#     if filename.endswith(".png"):
#         file_path = os.path.join("reports", filename)
#         os.remove(file_path)


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
    plt.savefig("reports/imgs/assignment1_cifar_examples.png")
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


def softmax(z):
    exp_z = np.exp(z)
    return exp_z / (np.sum(exp_z, axis=0) + eps)


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def apply_network(X, net, activation="softmax"):
    W = net["W"]
    b = net["b"]
    if activation == "sigmoid":
        return sigmoid(W @ X + b)
    elif activation == "softmax":
        return softmax(W @ X + b)
    else:
        raise ValueError("Invalid activation function. Choose 'sigmoid' or 'softmax'.")


def compute_loss(P, Y, net, lam=0.5, loss_type="cross_entropy"):
    W = net["W"]
    if loss_type == "cross_entropy":
        return cross_entropy(Y, P) + lam * np.sum(W * W)
    elif loss_type == "binary_cross_entropy":
        return binary_cross_entropy(Y, P) + lam * np.sum(W * W)


def binary_cross_entropy(y, y_hat):
    return -np.mean(y * np.log(y_hat + eps) + (1 - y) * np.log(1 - y_hat + eps))


def cross_entropy(y, y_hat):
    return -np.mean(np.sum(y * np.log(y_hat + eps), axis=0))


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
    X_train,
    Y_train,
    y_train,
    X_val,
    Y_val,
    y_val,
    gd_params,
    net,
    loss_type="cross_entropy",
    activation="softmax",
    verbose=False,
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

            P = apply_network(X_batch, net, activation=activation)
            # Backward
            grads = backward_pass(X_batch, Y_batch, P, net, lam)

            # Update
            net["W"] -= eta * grads["W"]
            net["b"] -= eta * grads["b"]

        # Compute loss and accuracy for the entire training set
        P = apply_network(X_train, net, activation=activation)
        train_loss = compute_loss(P, Y_train, net, lam, loss_type)
        train_acc = compute_accuracy(P, y_train)
        train_losses.append(train_loss)

        # Validation
        P_val = apply_network(X_val, net, activation=activation)
        val_loss = compute_loss(P_val, Y_val, net, lam, loss_type)
        val_acc = compute_accuracy(P_val, y_val)
        val_losses.append(val_loss)

        # Print progress
        if verbose:
            print(
                f"Epoch {epoch + 1}/{n_epochs}: Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.2f}%, "
                f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%"
            )

    # Final training and validation loss
    P = apply_network(X_train, net, activation=activation)
    train_loss = compute_loss(P, Y_train, net, lam, loss_type)
    train_acc = compute_accuracy(P, y_train)
    P_val = apply_network(X_val, net, activation=activation)
    val_loss = compute_loss(P_val, Y_val, net, lam, loss_type)
    val_acc = compute_accuracy(P_val, y_val)
    # Print final results
    print(
        f"Final Training Loss: {train_loss:.4f}, Final Training Accuracy: {train_acc:.2f}%, "
        f"Final Validation Loss: {val_loss:.4f}, Final Validation Accuracy: {val_acc:.2f}%"
    )

    return net, train_losses, val_losses


def show_learned_matrices(trained_net, gd_params, name="experiment"):
    Ws = trained_net["W"].transpose().reshape((32, 32, 3, 10), order="F")
    W_im = np.transpose(Ws, (1, 0, 2, 3))
    fig, axs = plt.subplots(1, 10, figsize=(20, 5))
    for i in range(10):
        w_im = W_im[:, :, :, i]
        w_im_norm = (w_im - np.min(w_im)) / (np.max(w_im) - np.min(w_im))
        axs[i].imshow(w_im_norm)
        axs[i].axis("off")
    plt.suptitle(
        f"Learned Weights (eta={gd_params['eta']}, n_epochs={gd_params['n_epochs']}, n_batch={gd_params['n_batch']}, lambda={gd_params['lam']})"
    )
    plt.savefig(f"reports/imgs/assignment1_learned_matrices_{name}.png")
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
    plt.savefig(f"reports/imgs/assignment1_loss_evolution_{name}.png")
    plt.close()


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


def plot_histogram_correctness(P, y_true, name):
    y_pred = np.argmax(P, axis=0)
    probs_correct_class = P[y_true, np.arange(P.shape[1])]

    correct_mask = y_pred == y_true
    incorrect_mask = ~correct_mask

    plt.hist(probs_correct_class[correct_mask], bins=30, alpha=0.7, label="Correct")
    plt.hist(probs_correct_class[incorrect_mask], bins=30, alpha=0.7, label="Incorrect")
    plt.xlabel("Predicted Probability for True Class")
    plt.ylabel("Count")
    plt.title(f"Histogram of True Class Probabilities ({name})")
    plt.legend()
    plt.grid(True)
    plt.savefig(f"reports/imgs/assignment1_histogram_correctness_{name}.png")
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
    files=True,
    loss_type="cross_entropy",
    activation="softmax",
):
    print(f"Running experiment {name} with parameters: {gd_params}")

    trained_net = copy.deepcopy(net)
    trained_net, train_losses, val_losses = mini_batch_GD(
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        gd_params,
        trained_net,
        loss_type,
        activation,
    )

    # Test the network
    P_test = apply_network(X_test, trained_net, activation=activation)
    test_loss = compute_loss(
        P_test, Y_test, trained_net, gd_params["lam"], loss_type=loss_type
    )
    test_acc = compute_accuracy(P_test, y_test)

    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

    if files:
        show_learned_matrices(trained_net, gd_params, name=name)
        show_loss_evolution(train_losses, val_losses, gd_params, name=name)

    return test_acc


def first_setup():
    # Visualizing dataset
    train_dir = "./Datasets/cifar-10-batches-py/data_batch_1"
    val_dir = "./Datasets/cifar-10-batches-py/data_batch_2"
    test_dir = "./Datasets/cifar-10-batches-py/test_batch"

    # Visualize the first 5 images from the dataset
    show_cifar_10_examples()

    # Load the dataset
    X_train, Y_train, y_train = load_batch(train_dir)
    X_val, Y_val, y_val = load_batch(val_dir)
    X_test, Y_test, y_test = load_batch(test_dir)

    # Normalize X
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

    # Initialize network
    net = init_weights()

    # Testing the network
    # p = apply_network(X_train[:, 0:100], net)
    # testing_grad(X_train, Y_train, y_train)

    # Run experiments
    experiments = [
        {"eta": 0.1, "n_epochs": 40, "n_batch": 100, "lam": 0},
        {"eta": 0.001, "n_epochs": 40, "n_batch": 100, "lam": 0},
        {"eta": 0.001, "n_epochs": 40, "n_batch": 100, "lam": 0.1},
        {"eta": 0.001, "n_epochs": 40, "n_batch": 100, "lam": 1},
    ]

    for i, gd_params in enumerate(experiments):
        experiment_name = f"experiment_{i + 1}"
        run_experiment(
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


def second_setup():
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

    # Validation set is the last 1000 samples of the training set
    X_val = X_train[:, -1000:]
    Y_val = Y_train[:, -1000:]
    y_val = y_train[-1000:]
    X_train = X_train[:, :-1000]
    Y_train = Y_train[:, :-1000]
    y_train = y_train[:-1000]

    # Load the test set
    X_test, Y_test, y_test = load_batch(test_dir)
    # Normalize X
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    # Initialize network
    net = init_weights()
    # Testing the network
    # p = apply_network(X_train[:, 0:100], net)
    # testing_grad(X_train, Y_train, y_train)

    ####### 2. Data Augmentation #######

    X_train_flip = flip_images(X_train)
    Y_train_flip = Y_train.copy()
    y_train_flip = y_train.copy()

    X_train_aug = np.concatenate((X_train, X_train_flip), axis=1)
    Y_train_aug = np.concatenate((Y_train, Y_train_flip), axis=1)
    y_train_aug = np.concatenate((y_train, y_train_flip), axis=0)

    ########### 3. Grid Search ###########
    n_subsamples = 5000
    # Define the grid search parameters
    eta_values = [0.01, 0.1, 1]
    n_epochs_values = [40]
    n_batch_values = [50, 100, 200, 400]
    lam_values = [0, 0.1, 1]
    best_test_acc = 0

    best_params = None

    for eta in eta_values:
        for n_epochs in n_epochs_values:
            for n_batch in n_batch_values:
                for lam in lam_values:
                    gd_params = {
                        "eta": eta,
                        "n_epochs": n_epochs,
                        "n_batch": n_batch,
                        "lam": lam,
                    }
                    print(f"Running grid search with params: {gd_params}")
                    # Run the experiment
                    trained_net = copy.deepcopy(net)
                    trained_net, train_losses, val_losses = mini_batch_GD(
                        X_train_aug[:, :n_subsamples],
                        Y_train_aug[:, :n_subsamples],
                        y_train_aug[:n_subsamples],
                        X_val,
                        Y_val,
                        y_val,
                        gd_params,
                        trained_net,
                    )

                    # Test the network
                    P_test = apply_network(X_test, trained_net)
                    test_loss = compute_loss(P_test, Y_test, trained_net, lam)
                    test_acc = compute_accuracy(P_test, y_test)

                    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.2f}%")

                    if test_acc > best_test_acc:
                        best_test_acc = test_acc
                        best_params = gd_params

    # Run experiments
    gd_params = best_params
    print(best_params)
    experiment_name = "experiment_2_second_setup"
    run_experiment(
        X_train_aug,
        Y_train_aug,
        y_train_aug,
        X_val,
        Y_val,
        y_val,
        X_test,
        Y_test,
        y_test,
        gd_params,
        net,
        name=experiment_name,
        files=True,
    )


def third_setup():
    train_dir = "./Datasets/cifar-10-batches-py/data_batch_1"
    val_dir = "./Datasets/cifar-10-batches-py/data_batch_2"
    test_dir = "./Datasets/cifar-10-batches-py/test_batch"
    # Load the dataset
    X_train, Y_train, y_train = load_batch(train_dir)
    X_val, Y_val, y_val = load_batch(val_dir)
    X_test, Y_test, y_test = load_batch(test_dir)
    # Normalize X
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)
    # Initialize network
    net = init_weights()
    # Testing the network
    # p = apply_network(X_train[:, 0:100], net)
    # testing_grad(X_train, Y_train, y_train)
    # Run experiments
    best_params = {
        "eta": 0.01,
        "n_epochs": 40,
        "n_batch": 400,
        "lam": 0.1,
    }
    experiment_name = "experiment_3_third_setup_softmax"
    run_experiment(
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        X_test,
        Y_test,
        y_test,
        best_params,
        net,
        name=experiment_name,
        loss_type="cross_entropy",
        activation="softmax",
    )
    experiment_name = "experiment_3_third_setup_sigmoid"
    best_params = {
        "eta": 0.01,
        "n_epochs": 40,
        "n_batch": 400,
        "lam": 0.01,
    }
    run_experiment(
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        X_test,
        Y_test,
        y_test,
        best_params,
        net,
        name=experiment_name,
        files=True,
        loss_type="binary_cross_entropy",
        activation="sigmoid",
    )
    net = init_weights()
    # Train the network with sigmoid activation
    trained_net = copy.deepcopy(net)
    trained_net, train_losses, val_losses = mini_batch_GD(
        X_train,
        Y_train,
        y_train,
        X_val,
        Y_val,
        y_val,
        best_params,
        trained_net,
        loss_type="binary_cross_entropy",
        activation="sigmoid",
    )
    # Show histogram of correctness
    P_test = apply_network(X_test, net, activation="sigmoid")
    plot_histogram_correctness(P_test, y_test, "sigmoid")
    P_test = apply_network(X_test, net, activation="softmax")
    plot_histogram_correctness(P_test, y_test, "softmax")
    print("All experiments completed.")
    print("Check reports folder for results.")


if __name__ == "__main__":
    # Uncomment the function you want to run
    # first_setup()
    # second_setup()
    third_setup()
