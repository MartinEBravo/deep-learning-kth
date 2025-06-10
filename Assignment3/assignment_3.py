import pickle
import tqdm
import numpy as np
import matplotlib.pyplot as plt
import time

np.random.seed(42)
eps = 1e-8 


def get_MX(X_ims, f, stride=4):
    # Compute dimensions
    H, W, C, n = X_ims.shape
    n_patches = ((H - f) // stride + 1) * ((W - f) // stride + 1)

    # Compute MX
    MX = np.zeros((n_patches, f * f * C, n))
    for i in range(n):
        row_l = 0
        for y in range(0, H - f + 1, stride):
            for x in range(0, W - f + 1, stride):
                X_patch = X_ims[y : y + f, x : x + f, :, i]
                MX[row_l, :, i] = X_patch.reshape((f * f * C), order="C")
                row_l += 1

    return MX


def conv_3d_from_MX(MX, kernels, stride=4):
    # Compute dimensions
    n_patches, f_C, n = MX.shape
    f, f, _, n_filters = kernels.shape
    out_h = (32 - f) // stride + 1
    out_w = (32 - f) // stride + 1

    F_all = kernels.reshape((f * f * 3, n_filters), order="C")
    conv_outputs_mat = np.einsum("ijn, jl ->iln", MX, F_all, optimize=True)

    return conv_outputs_mat.reshape((out_h, out_w, n_filters, n), order="C")


def ReLU(s):
    return np.maximum(0, s)


def softmax(s):
    exp_s = np.exp(s - np.max(s, axis=0, keepdims=True))
    return exp_s / np.sum(exp_s, axis=0, keepdims=True)


def forward_pass(MX, net_params, stride=4, return_x1=False, conv_flat=False):
    W1 = net_params["W1"]
    b1 = net_params["b1"]
    W2 = net_params["W2"]
    b2 = net_params["b2"]
    F = net_params["F"]

    if isinstance(conv_flat, bool):
        conv_out = conv_3d_from_MX(MX, F, stride)
        conv_flat = conv_out.reshape((-1, MX.shape[2]))
    conv_flat = ReLU(conv_flat)
    X1 = ReLU(W1 @ conv_flat + b1)
    p = softmax(W2 @ X1 + b2)
    if return_x1:
        return X1, p
    return p


def backward_pass(MX, Y, net_params, lam, stride=4):
    W1 = net_params["W1"]  # (hidden_dim, conv_dim)
    b1 = net_params["b1"]  # (hidden_dim, 1)
    W2 = net_params["W2"]  # (num_classes, hidden_dim)
    b2 = net_params["b2"]  # (num_classes, 1)
    F = net_params["F"]  # (f, f, C, n_filters)

    n = MX.shape[2]

    conv_out = conv_3d_from_MX(MX, F, stride)
    relu_mask = conv_out > 0
    conv_out  = conv_out * relu_mask
    conv_flat = conv_out.reshape((-1, n))
    X1 = ReLU(W1 @ conv_flat + b1)  # (hidden_dim, n)
    P = softmax(W2 @ X1 + b2)  # (num_classes, n)

    G = P - Y

    grads = {}

    grads["W2"] = (G @ X1.T) / n
    grads["b2"] = np.sum(G, axis=1, keepdims=True) / n

    G_hidden = W2.T @ G
    G_hidden[X1 <= 0] = 0

    grads["W1"] = (G_hidden @ conv_flat.T) / n
    grads["b1"] = np.sum(G_hidden, axis=1, keepdims=True) / n

    G_conv_flat = W1.T @ G_hidden
    conv_shape = conv_out.shape
    G_conv_out = G_conv_flat.reshape(conv_shape, order="C") 
    G_conv_out *= relu_mask

    f = F.shape[0]
    C = F.shape[2]
    n_filters = F.shape[3]

    grads["F"] = np.zeros_like(F)

    for i in range(n):
        grad_F_all_i = MX[:, :, i].T @ G_conv_out[:, :, :, i].reshape((-1, n_filters))
        grads["F"] += grad_F_all_i.reshape((f, f, C, n_filters), order="C")

    grads["F"] /= n

    # Regularization
    scale = lam / n
    grads["W1"] += scale * W1
    grads["W2"] += scale * W2
    grads["F"]  += scale * F

    return grads


def test_convolution():
    # Load data
    debug_file = "Datasets/debug_info.npz"
    load_data = np.load(debug_file)
    X = load_data["X"]
    Fs = load_data["Fs"]
    targets = load_data["conv_outputs"]
    assert X.shape == (3072, 5), "Wrong shape"
    assert Fs.shape == (4, 4, 3, 2), "Wrong shape"
    X_ims = np.transpose(X.reshape((32, 32, 3, 5), order="F"), (1, 0, 2, 3))
    assert X_ims.shape == (32, 32, 3, 5), "Wrong shape"

    # Execute convolutions
    MX = get_MX(X_ims, Fs.shape[0], stride=4)
    prediction = conv_3d_from_MX(MX, Fs, stride=4)

    # Compare convolutions
    assert np.allclose(prediction, targets)


def test_forward():
    # Load data
    debug_file = "Datasets/debug_info.npz"
    load_data = np.load(debug_file)
    net_params = {
        "W1": load_data["W1"],
        "b1": load_data["b1"],
        "W2": load_data["W2"],
        "b2": load_data["b2"],
        "F": 0,
    }
    conv_flat = load_data["conv_flat"]
    X1_target = load_data["X1"]
    p_target = load_data["P"]
    X1, p = forward_pass(conv_flat, net_params, return_x1=True, conv_flat=conv_flat)
    assert np.allclose(X1, X1_target)
    assert np.allclose(p, p_target)


def to_one_hot(y: np.ndarray):
    """
    Converts a numpy array of labels to a one-hot encoded numpy array.
    """
    return np.eye(10)[y]


def compute_loss(p, y, lam=0.0, net_params={"W1": 0, "W2": 0, "F": 0}):
    W1 = net_params["W1"]
    W2 = net_params["W2"]
    F = net_params["F"]
    n = p.shape[1]
    cross_entropy = -np.sum(y * np.log(p + eps)) / n
    reg_term = (
        lam
        * (np.sum(W1**2) + np.sum(W2**2) + np.sum(F**2))
        / (2 * n)
    )
    return cross_entropy + reg_term


def cyclical_learning_rate(n_min, n_max, step_size, iteration):
    cycle = np.floor(1 + iteration / (2 * step_size))
    x = np.abs(iteration / step_size - 2 * cycle + 1)
    lr = n_min + (n_max - n_min) * np.maximum(0, (1 - x))
    return lr


def cyclical_learning_rate_increasing(n_min, n_max, step_size, iteration):
    cycle = np.floor(1 + iteration / (2 * step_size))
    x = np.abs(iteration / step_size - 2 * cycle + 1)
    lr = n_min + (n_max - n_min) * np.maximum(0, (1 - x))
    return lr


def train_model(
    X, Y, 
    X_val, Y_val,
    net_params, n_cycles=3, batch_size=100, step=800, n_min=1e-5, n_max=1e-1, lam=0.0,
    stride=4, increasing=False
):
    n = X.shape[3]
    num_batches = n // batch_size
    MX = get_MX(X, net_params["F"].shape[0], stride=stride)
    MX_val = get_MX(X_val, net_params["F"].shape[0], stride=stride)

    train_losses = []
    val_losses = []
    
    # Shuffle data
    indices = np.arange(n)
    np.random.shuffle(indices)
    MX = MX[:, :, indices]
    Y = Y[:, indices]

    # Steps
    if not increasing:
        total_steps = n_cycles * step * 2
    else:
        total_steps = 0
        for i in range(n_cycles):
            total_steps += step * (2**(i+1))

    time_start = time.time()

    for iteration in tqdm.tqdm(range(total_steps)):

        i = iteration % num_batches
        start = i * batch_size
        end = start + batch_size
        MX_batch = MX[:, :, start:end]
        Y_batch = Y[:, start:end]

        # Compute learning rate based on current iteration
        if increasing:
            learning_rate = cyclical_learning_rate_increasing(n_min, n_max, step, iteration)
        else:
            learning_rate = cyclical_learning_rate(n_min, n_max, step, iteration)

        # Forward pass
        p = forward_pass(MX_batch, net_params, stride=stride)

        # Compute loss
        loss = compute_loss(p, Y_batch, lam=lam, net_params=net_params)
        train_losses.append(loss)

        # Backward pass
        grads = backward_pass(MX_batch, Y_batch, net_params, lam, stride=stride)

        # Update parameters
        for key in grads:
            net_params[key] -= learning_rate * grads[key]

        # p_val = forward_pass(MX_val, net_params, stride=stride)
        # val_loss = compute_loss(p_val, Y_val, lam=lam, net_params=net_params)
        # val_losses.append(val_loss)

    total_time = time.time() - time_start

    return net_params, train_losses, val_losses, total_time


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


def exercise_3():
    # Load data
    batch_files = [f"./Datasets/cifar-10-batches-py/data_batch_{i}" for i in range(1, 6)]
    test_file = "./Datasets/cifar-10-batches-py/test_batch"

    X_train_list, Y_train_list, y_train_list = [], [], []

    for file in batch_files:
        X, Y, y = load_batch(file)
        X_train_list.append(X)
        Y_train_list.append(Y)
        y_train_list.append(y)

    X_train = np.concatenate(X_train_list, axis=1)
    Y_train = np.concatenate(Y_train_list, axis=1)
    y_train = np.concatenate(y_train_list, axis=0)

    X_test, Y_test, y_test = load_batch(test_file)

    # Split validation
    train_idx = 49000
    X_val = X_train[:, train_idx:]
    Y_val = Y_train[:, train_idx:]
    y_val = y_train[train_idx:]

    X_train = X_train[:, :train_idx]
    Y_train = Y_train[:, :train_idx]
    y_train = y_train[:train_idx]

    # Normalize
    X_train, X_val, X_test = normalize_data(X_train, X_val, X_test)

    # Reshape for conv
    X_train = np.transpose(X_train.reshape((32, 32, 3, -1), order="F"), (1, 0, 2, 3))
    X_val = np.transpose(X_val.reshape((32, 32, 3, -1), order="F"), (1, 0, 2, 3))
    X_test = np.transpose(X_test.reshape((32, 32, 3, -1), order="F"), (1, 0, 2, 3))

    # Init network
    hyperparameters = {
        "f": [2, 4, 8, 16],
        "nf": [3, 10, 40, 160],
        "nh": [50, 50, 50, 50]
    }
    num_classes = 10
    stride = 4

    plt.figure(figsize=(12, 5))

    for i in range(4):

        print(f"Network {i}")

        f = hyperparameters["f"][i] # Filter size
        n_filters = hyperparameters["nf"][i]
        hidden_dim = hyperparameters["nh"][i]
        print(f"Hyperparameters: Filter Size = {f}, Filters = {n_filters}, Hidden Dim = {hidden_dim}")

        conv_dim = ((32 - f) // stride + 1) ** 2 * n_filters

        net_params = {
            "F" : np.random.randn(f,f,3,n_filters) * np.sqrt(2 / (f*f*3)),
            "W1": np.random.randn(hidden_dim, conv_dim) * np.sqrt(2 / conv_dim),
            "b1": np.zeros((hidden_dim,1)),
            "W2": np.random.randn(num_classes, hidden_dim) * np.sqrt(2 / hidden_dim),
            "b2": np.zeros((num_classes,1)),
        }

        # Train
        net_params, train_losses, validation_losses, total_time = train_model(
            X_train,
            Y_train,
            X_val,
            Y_val,
            net_params,
            n_cycles=3,
            batch_size=100,
            step=800,
            n_min=1e-5,
            n_max=1e-1,
            lam=.003,
            stride=stride,
        )

        # Test
        MX_test = get_MX(X_test, net_params["F"].shape[0], stride=stride)
        p_test = forward_pass(MX_test, net_params, stride=stride)
        y_pred = np.argmax(p_test, axis=0)
        accuracy = np.mean(y_pred == y_test)
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Total time: {total_time}")

        plt.plot(train_losses, label=f"Training Loss f={f}, nf={n_filters}, nh={hidden_dim}")
        plt.plot(validation_losses, label=f"Validation Loss Experiment f={f}, nf={n_filters}, nh={hidden_dim}")

    plt.legend()
    plt.savefig(f"reports/imgs/assignment_3_loss_results.png")
    
if __name__ == "__main__":
    # Testing Convolution
    # test_convolution()

    # Testing Forward Pass
    # test_forward()

    # Running Exercise 3
    exercise_3()
