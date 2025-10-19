import numpy as np
from config import DATA_ROOT
from cnn import get_MX, conv_3d_from_MX, forward_pass


def test_convolution():
    # Load data
    debug_file = DATA_ROOT / "debug_info.npz"
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
    debug_file = DATA_ROOT / "debug_info.npz"
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
