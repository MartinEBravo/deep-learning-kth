import numpy as np
from utils import (
    ReLU,
    softmax,
)
from config import eps


class CNN:
    def __init__(
        self,
        f,
        n_filters,
        hidden_dim,
        stride=4,
        num_classes=10,
        F=None,
        bF=None,
        W1=None,
        b1=None,
        W2=None,
        b2=None,
    ):
        # Store parameters
        self.f = f
        self.out_h = (32 - f) // stride + 1
        self.n_p = self.out_h * self.out_h
        self.n_filters = n_filters
        self.hidden_dim = hidden_dim
        self.stride = stride
        self.num_classes = num_classes

        # Gradients placeholders
        self.dL_dF = np.random.randn(f * f * 3, n_filters)
        self.dL_dbF = np.random.randn(n_filters, 1)
        self.dL_dW1 = np.random.randn(
            hidden_dim, ((32 - f) // stride + 1) ** 2 * n_filters
        )
        self.dL_db1 = np.random.randn(hidden_dim, 1)
        self.dL_dW2 = np.random.randn(num_classes, hidden_dim)
        self.dL_db2 = np.random.randn(num_classes, 1)

        # Initialize weights
        self.conv_dim = ((32 - self.f) // self.stride + 1) ** 2 * self.n_filters

        self.F = (
            np.random.randn(self.f * self.f * 3, self.n_filters)
            * np.sqrt(2 / (self.f * self.f * 3))
            if F is None
            else F.reshape(self.f * self.f * 3, self.n_filters, order="C")
        )
        self.bF = np.zeros((self.n_filters, 1)) if bF is None else bF
        self.W1 = (
            np.random.randn(self.hidden_dim, self.conv_dim) * np.sqrt(2 / self.conv_dim)
            if W1 is None
            else W1
        )
        self.b1 = np.zeros((self.hidden_dim, 1)) if b1 is None else b1
        self.W2 = (
            np.random.randn(self.num_classes, self.hidden_dim)
            * np.sqrt(2 / self.hidden_dim)
            if W2 is None
            else W2
        )
        self.b2 = np.zeros((self.num_classes, 1)) if b2 is None else b2

    # Compute MX matrix for convolution
    def _get_MX(self, X):
        _, _, _, n = X.shape
        n_patches = ((32 - self.f) // self.stride + 1) * (
            (32 - self.f) // self.stride + 1
        )

        # Compute MX
        MX = np.zeros((n_patches, self.f * self.f * 3, n))
        for i in range(n):
            row_l = 0
            for y in range(0, 32 - self.f + 1, self.stride):
                for x in range(0, 32 - self.f + 1, self.stride):
                    X_patch = X[y : y + self.f, x : x + self.f, :, i]
                    MX[row_l, :, i] = X_patch.reshape((self.f * self.f * 3), order="C")
                    row_l += 1

        return MX

    # Efficient convolution using matrix multiplication
    def _conv_step(self, MX):
        # Compute dimensions
        _, _, n = MX.shape
        out_h = (32 - self.f) // self.stride + 1
        out_w = (32 - self.f) // self.stride + 1
        conv_outputs_mat = np.einsum("ijn, jl ->iln", MX, self.F, optimize=True)
        conv_outputs_mat += self.bF[np.newaxis, :, :]
        return conv_outputs_mat.reshape((out_h, out_w, self.n_filters, n), order="C")

    def _flat_step(self, conv_out):
        _, _, _, n = conv_out.shape
        return conv_out.reshape(-1, n)

    # Fully connected Step
    def _fc_step(self, conv_flat):
        X1 = ReLU(self.W1 @ ReLU(conv_flat) + self.b1)
        p = softmax(self.W2 @ X1 + self.b2)
        return X1, p

    # Normal Forward pass
    def forward(self, X):
        MX = self._get_MX(X)
        conv_out = self._conv_step(MX)
        conv_flat = self._flat_step(conv_out)
        _, p = self._fc_step(conv_flat)
        return p

    # ---- Backward Functions -----
    def _update_grads(self, X, Y, lam=0):
        _, _, _, n = X.shape

        # Forward
        MX = self._get_MX(X)
        conv_out = self._conv_step(MX)
        conv_flat = self._flat_step(conv_out)
        X1, p = self._fc_step(conv_flat)

        # Backward
        # Fully Connected Layers
        dL_ds = p - Y
        self.dL_dW2 = (dL_ds @ X1.T) / n
        self.dL_db2 = np.sum(dL_ds, axis=1, keepdims=True) / n
        dL_X1 = self.W2.T @ dL_ds
        dL_X1[X1 <= 0] = 0
        self.dL_dW1 = (dL_X1 @ ReLU(conv_flat).T) / n
        self.dL_db1 = np.sum(dL_X1, axis=1, keepdims=True) / n

        # Convolutional Layer
        dL_dh = self.W1.T @ dL_X1
        dL_dh[conv_flat <= 0] = 0
        GG = dL_dh.reshape(self.n_p, self.n_filters, n, order="C")
        MXt = np.transpose(MX, (1, 0, 2))
        self.dL_dF = np.einsum("ijn, jln -> il", MXt, GG, optimize=True) / n
        self.dL_dbF = (np.sum(GG, axis=(0, 2)).reshape(self.n_filters, 1)) / n

        # Regularization
        self.dL_dF += 2 * lam * self.F
        self.dL_dW1 += 2 * lam * self.W1
        self.dL_dW2 += 2 * lam * self.W2

    def backward(self, X, Y, lam=0.0, learning_rate=0.01):
        """
        Perform a backward pass and update parameters.
        """
        # compute and store grads
        self._update_grads(X, Y, lam)
        # apply parameter updates
        self.F -= learning_rate * self.dL_dF
        self.bF -= learning_rate * self.dL_dbF
        self.W1 -= learning_rate * self.dL_dW1
        self.b1 -= learning_rate * self.dL_db1
        self.W2 -= learning_rate * self.dL_dW2
        self.b2 -= learning_rate * self.dL_db2

    def compute_loss(self, p, y, lam=0.0):
        n = p.shape[1]
        cross_entropy = -np.sum(y * np.log(p + eps)) / n
        reg_term = (
            lam
            * (np.sum(self.W1**2) + np.sum(self.W2**2) + np.sum(self.F**2))
            / (2 * n)
        )
        return cross_entropy + reg_term
