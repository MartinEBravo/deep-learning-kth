import numpy as np
from config import DATA_ROOT
from cnn import CNN
from torch_grads import compute_grads_with_torch


debug_file = DATA_ROOT / "debug_info.npz"
load_data = np.load(debug_file)


def test_convolution():
    # Load data
    X = load_data["X"]
    Fs = load_data["Fs"]
    targets = load_data["conv_outputs"]
    X_ims = np.transpose(X.reshape((32, 32, 3, 5), order="F"), (1, 0, 2, 3))
    assert (
        X_ims.shape == (32, 32, 3, 5)
        and X.shape == (3072, 5)
        and Fs.shape == (4, 4, 3, 2)
    ), "Wrong shape"

    # Execute convolutions using the CNN helpers
    # instantiate a minimal CNN to reuse its MX/conv code
    net = CNN(f=Fs.shape[0], n_filters=Fs.shape[3], hidden_dim=1, F=Fs)
    MX = net.get_MX(X=X_ims)
    prediction = net._conv_step(MX)

    # Compare convolutions
    assert np.allclose(prediction, targets), (
        "Convolution outputs do not match expected values"
    )


def test_forward():
    # Input
    conv_flat = load_data["conv_flat"]

    # CNN parameters
    W1 = load_data["W1"]
    b1 = load_data["b1"]
    W2 = load_data["W2"]
    b2 = load_data["b2"]

    # Outputs
    X1_target = load_data["X1"]
    p_target = load_data["P"]

    # Build a dummy CNN with the correct hidden dimension
    net = CNN(
        f=4,
        n_filters=conv_flat.shape[0] // ((32 // 4) ** 2),
        hidden_dim=X1_target.shape[1],
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
    )

    # Use the CNN method that consumes conv_flat directly
    X1, p = net._fc_step(conv_flat)

    X1_expected = np.squeeze(X1_target, axis=0)
    assert np.allclose(p, p_target), "Probabilities do not match expected values"
    assert np.allclose(X1, X1_expected), "X1 does not match expected values"


def test_backward():
    X = load_data["X"]
    Fs = load_data["Fs"]
    Y = load_data["Y"]
    grad_Fs = load_data["grad_Fs_flat"]
    X_ims = np.transpose(X.reshape((32, 32, 3, 5), order="F"), (1, 0, 2, 3))

    # CNN parameters
    W1 = load_data["W1"]
    b1 = load_data["b1"]
    W2 = load_data["W2"]
    b2 = load_data["b2"]
    Fs = load_data["Fs"]
    X1_target = load_data["X1"]

    # Build a dummy CNN with the correct hidden dimension
    net = CNN(
        f=4,
        n_filters=Fs.shape[3],
        hidden_dim=X1_target.shape[1],
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        F=Fs,
    )

    MX = net.get_MX(X_ims)
    net.backward(MX=MX, Y=Y)
    assert np.allclose(net.dL_dF, grad_Fs), (
        "Filter gradients do not match expected values"
    )


def test_pytorch():
    # Input
    X = load_data["X"]
    Fs = load_data["Fs"]
    Y = load_data["Y"]

    # CNN parameters
    W1 = load_data["W1"]
    b1 = load_data["b1"]
    W2 = load_data["W2"]
    b2 = load_data["b2"]

    # Build a dummy CNN with the correct hidden dimension
    net = CNN(
        f=4,
        n_filters=Fs.shape[3],
        hidden_dim=W1.shape[0],
        W1=W1,
        b1=b1,
        W2=W2,
        b2=b2,
        F=Fs,
    )

    dL_dF_torch, dL_dW1_torch, dL_db1_torch, dL_dW2_torch, dL_db2_torch = (
        compute_grads_with_torch(X, Y, net)
    )
    
    X = np.transpose(X.reshape((32, 32, 3, 5), order="F"), (1, 0, 2, 3))
    MX = net.get_MX(X)
    net.backward(MX, Y)

    assert np.allclose(net.dL_dF, dL_dF_torch), (
        "Filter gradients do not match implemented gradients"
    )
    assert np.allclose(net.dL_dW1, dL_dW1_torch), (
        "W1 gradients do not match implemented gradients"
    )
    assert np.allclose(net.dL_db1, dL_db1_torch), (
        "b1 gradients do not match implemented gradients"
    )
    assert np.allclose(net.dL_dW2, dL_dW2_torch), (
        "W2 gradients do not match implemented gradients"
    )
    assert np.allclose(net.dL_db2, dL_db2_torch), (
        "b2 gradients do not match implemented gradients"
    )
