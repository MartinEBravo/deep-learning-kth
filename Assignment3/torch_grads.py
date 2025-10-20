import torch
import numpy as np


def compute_grads_with_torch(X, Y_onehot, net):
    dtype = torch.float64

    n = X.shape[1]
    f = net.f
    nf = net.n_filters
    oh = (32 - f) // net.stride + 1

    F = torch.from_numpy(net.F).to(dtype).clone().detach().requires_grad_(True)
    W1 = torch.from_numpy(net.W1).to(dtype).clone().detach().requires_grad_(True)
    b1 = torch.from_numpy(net.b1).to(dtype).clone().detach().requires_grad_(True)
    W2 = torch.from_numpy(net.W2).to(dtype).clone().detach().requires_grad_(True)
    b2 = torch.from_numpy(net.b2).to(dtype).clone().detach().requires_grad_(True)

    X_ims_np = np.transpose(X.reshape(32, 32, 3, n, order="F"), (1, 0, 2, 3))
    X_ims = torch.from_numpy(X_ims_np).to(dtype)

    MX = torch.zeros((oh * oh, f * f * 3, n), dtype=dtype)
    row = 0
    for y0 in range(0, 32 - f + 1, net.stride):
        for x0 in range(0, 32 - f + 1, net.stride):
            patch = X_ims[y0 : y0 + f, x0 : x0 + f, :, :]
            MX[row, :, :] = patch.reshape(f * f * 3, n)
            row += 1

    conv_mat = torch.einsum("ijn,jl->iln", MX, F)
    conv_out = conv_mat.reshape(oh, oh, nf, n)

    conv_flat = conv_out.reshape(-1, n)
    relu = torch.nn.ReLU()
    conv_flat = relu(conv_flat)

    hidden = relu(W1 @ conv_flat + b1)

    scores = W2 @ hidden + b2

    P = torch.softmax(scores, dim=0)

    y_t = torch.from_numpy(Y_onehot).to(dtype)
    y_int = torch.argmax(y_t, dim=0)

    loss = torch.mean(-torch.log(P[y_int, torch.arange(n)]))

    loss.backward()
    return (
        F.grad.detach().cpu().numpy(),
        W1.grad.detach().cpu().numpy(),
        b1.grad.detach().cpu().numpy(),
        W2.grad.detach().cpu().numpy(),
        b2.grad.detach().cpu().numpy(),
    )
