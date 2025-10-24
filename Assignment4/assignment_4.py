from pathlib import Path
import numpy as np
import torch
import tqdm
import matplotlib.pyplot as plt


# Load the book data
ROOT_DIR = Path(__file__).resolve().parent
book_path = ROOT_DIR / "goblet_book.txt"
fid = open(book_path, "r", encoding="utf-8")
book_data = fid.read()
fid.close()

# Create vocabulary and mappings
vocab = sorted(list(set(book_data)))
vocab_size = len(vocab)
char_to_index = {ch: i for i, ch in enumerate(vocab)}
index_to_char = {i: ch for i, ch in enumerate(vocab)}
for i in range(vocab_size):
    char = index_to_char[i]
    assert char_to_index[char] == i


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=0, keepdims=True) + 1e-12)


class RNN:
    def __init__(self, seq_length=25, hidden_size=100, vocab_size=80, random_seed=0):
        # Model parameters
        self.seq_length = seq_length
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        np.random.seed(random_seed)

        # Adam Hyperparameters
        self.eta = 1e-3
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8
        self.t = 0

        # Initialize weights
        self.U = (1 / np.sqrt(2 * self.hidden_size)) * np.random.randn(
            self.vocab_size, self.hidden_size
        )
        self.W = (1 / np.sqrt(2 * self.hidden_size)) * np.random.randn(
            self.hidden_size, self.hidden_size
        )
        self.V = (1 / np.sqrt(2 * self.hidden_size)) * np.random.randn(
            self.vocab_size, self.hidden_size
        )

        # Initialize biases
        self.b = np.zeros((self.hidden_size, 1))
        self.c = np.zeros((self.vocab_size, 1))

        # Initialize Adam parameters
        self.mU, self.vU = np.zeros_like(self.U), np.zeros_like(self.U)
        self.mW, self.vW = np.zeros_like(self.W), np.zeros_like(self.W)
        self.mV, self.vV = np.zeros_like(self.V), np.zeros_like(self.V)
        self.mb, self.vb = np.zeros_like(self.b), np.zeros_like(self.b)
        self.mc, self.vc = np.zeros_like(self.c), np.zeros_like(self.c)

    def step(self, x_t, h_prev):
        a_t = (
            self.W @ h_prev  # (m,m) @ (m,1) -> (m,1)
            + self.U.T @ x_t  # (m,k) @ (k,1) -> (m,1)
            + self.b  # (m,1)
        )
        h_t = np.tanh(a_t)
        o_t = self.V @ h_t + self.c
        p_t = softmax(o_t)
        return h_t, p_t

    def forward(self, x, h_prev):
        self.x = x
        xs, hs, ps = {}, {}, {}
        hs[-1] = h_prev
        T = x.shape[1]
        for t in range(T):
            xs[t] = x[:, [t]]
            h_t, ps[t] = self.step(xs[t], hs[t - 1])
            hs[t] = h_t
        self.xs = xs
        self.hs = hs
        self.ps = ps
        self.h_last = hs[T - 1]
        return xs, hs, ps

    def synthesize(self, x0, h0, length):
        x = x0
        h = h0
        generated = []
        for _ in range(length):
            h, p = self.step(x, h)
            cp = np.cumsum(p, axis=0)
            a = np.random.rand()  # instead of rng.uniform()
            idx = np.argmax(cp - a > 0)
            generated.append(idx)
            x = np.zeros((self.vocab_size, 1))
            x[idx, 0] = 1
        return generated

    def compute_loss(self, y, ps):
        loss = 0
        T = y.shape[1]
        for t in range(T):
            loss += -np.log(y[:, [t]].T @ ps[t] + 1e-12)
        return loss.item() / T

    def compute_smooth_loss(self, loss, prev_smooth_loss=-1):
        return (
            (0.999 * prev_smooth_loss + 0.001 * loss) if prev_smooth_loss >= 0 else loss
        )

    def backward(self, x, y, hs, ps):
        dU, dW, dV, db, dc = (
            np.zeros_like(self.U),
            np.zeros_like(self.W),
            np.zeros_like(self.V),
            np.zeros_like(self.b),
            np.zeros_like(self.c),
        )
        dh_next = np.zeros((self.hidden_size, 1))

        T = x.shape[1]
        for t in reversed(range(T)):
            dy = ps[t] - y[:, [t]]
            dV += dy @ hs[t].T
            dc += dy
            dh = self.V.T @ dy + dh_next
            da = (1 - hs[t] ** 2) * dh
            db += da
            dW += da @ hs[t - 1].T
            dU += x[:, [t]] @ da.T
            dh_next = self.W.T @ da

        for dparam in [dU, dW, dV, db, dc]:
            np.clip(dparam, -5, 5, out=dparam)

        return dU, dW, dV, db, dc

    def update_param(self, grad, m_prev, v_prev, parameter):
        update, m_new, v_new = self.get_adam_update(grad, m_prev, v_prev)
        parameter -= update
        return m_new, v_new

    def get_adam_update(self, grad, m_prev, v_prev):
        m_new = self.beta1 * m_prev + (1 - self.beta1) * grad
        v_new = self.beta2 * v_prev + (1 - self.beta2) * (grad**2)
        m_hat = m_new / (1 - self.beta1**self.t)
        v_hat = v_new / (1 - self.beta2**self.t)
        update = self.eta * m_hat / np.sqrt(v_hat + self.eps)
        return update, m_new, v_new

    def update_parameters(self):
        dU, dW, dV, db, dc = self.backward(self.x, self.y, self.hs, self.ps)
        self.t += 1
        self.mU, self.vU = self.update_param(dU, self.mU, self.vU, self.U)
        self.mW, self.vW = self.update_param(dW, self.mW, self.vW, self.W)
        self.mV, self.vV = self.update_param(dV, self.mV, self.vV, self.V)
        self.mb, self.vb = self.update_param(db, self.mb, self.vb, self.b)
        self.mc, self.vc = self.update_param(dc, self.mc, self.vc, self.c)


class Dataset:
    def __init__(self, data, char_to_index, seq_length=25):
        self.data = data
        self.char_to_index = char_to_index
        self.seq_length = seq_length
        self.pointer = 0
        self.data_size = len(data)

    def get_batch(self):
        x_batch = np.zeros((len(self.char_to_index), self.seq_length))
        y_batch = np.zeros((len(self.char_to_index), self.seq_length))
        for t in range(self.seq_length):
            if self.pointer + t + 1 >= self.data_size:
                self.pointer = 0
            x_char = self.data[self.pointer + t]
            y_char = self.data[self.pointer + t + 1]
            x_batch[self.char_to_index[x_char], t] = 1
            y_batch[self.char_to_index[y_char], t] = 1
        self.pointer += self.seq_length
        return x_batch, y_batch


def train_rnn(
    update_steps=100000, print_every=10000, synth_length=200, final_synth_length=1000
):
    rnn = RNN(seq_length=25, hidden_size=100, vocab_size=vocab_size)
    dataset = Dataset(book_data, char_to_index, seq_length=rnn.seq_length)
    h_prev = np.zeros((rnn.hidden_size, 1))
    losses = []
    smooth_losses = []
    smooth_loss = -1
    for iter in tqdm.tqdm(range(update_steps)):
        # Get the next input and target
        x, y = dataset.get_batch()
        rnn.forward(x, h_prev)
        loss = rnn.compute_loss(y, rnn.ps)
        smooth_loss = rnn.compute_smooth_loss(loss, smooth_loss)
        losses.append(loss)
        smooth_losses.append(smooth_loss)
        if iter % print_every == 0:
            print(f"Iteration {iter}, Loss: {loss}, Smooth Loss: {smooth_loss}")
            # Synthesize text
            x0 = x[:, [0]]
            generated_indices = rnn.synthesize(x0, h_prev, synth_length)
            generated_text = "".join(index_to_char[idx] for idx in generated_indices)
            print(f"Synthesized Text:\n{generated_text}\n")
        rnn.y = y
        rnn.update_parameters()
        h_prev = rnn.h_last

    # Final synthesis
    x0 = x[:, [0]]
    generated_indices = rnn.synthesize(x0, h_prev, final_synth_length)
    generated_text = "".join(index_to_char[idx] for idx in generated_indices)
    print(f"Final Synthesized Text:\n{generated_text}\n")

    # Plot losses
    plot_loss(losses, smooth_losses)


def plot_loss(losses, smooth_losses):
    plt.figure(figsize=(10, 5))
    plt.plot(losses, label="Loss")
    plt.plot(smooth_losses, label="Smooth Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.legend()
    plt.savefig("training_loss.pdf")
    plt.show()



def compute_grads_with_torch(rnn_obj, inputs, target_indices, h0):
    tau = inputs.shape[1]
    Xt = torch.from_numpy(inputs).double()
    hprev = torch.from_numpy(h0).double()
    params = {
        "U": torch.tensor(rnn_obj.U.T, dtype=torch.float64, requires_grad=True),
        "W": torch.tensor(rnn_obj.W, dtype=torch.float64, requires_grad=True),
        "V": torch.tensor(rnn_obj.V, dtype=torch.float64, requires_grad=True),
        "b": torch.tensor(rnn_obj.b, dtype=torch.float64, requires_grad=True),
        "c": torch.tensor(rnn_obj.c, dtype=torch.float64, requires_grad=True),
    }
    tanh = torch.nn.Tanh()
    softmax = torch.nn.Softmax(dim=0)
    hidden_states = torch.empty(
        (rnn_obj.hidden_size, tau), dtype=torch.float64
    )

    for t in range(tau):
        hprev = tanh(
            params["W"] @ hprev
            + params["U"] @ Xt[:, t : t + 1]
            + params["b"]
        )
        hidden_states[:, [t]] = hprev

    outputs = params["V"] @ hidden_states + params["c"]
    probs = softmax(outputs)
    time_index = torch.arange(tau)
    target_index = torch.from_numpy(target_indices).long()
    selected = probs[target_index, time_index]
    loss = -torch.log(selected).sum()
    loss.backward()

    grads = {
        "U": params["U"].grad.detach().numpy().T,
        "W": params["W"].grad.detach().numpy(),
        "V": params["V"].grad.detach().numpy(),
        "b": params["b"].grad.detach().numpy(),
        "c": params["c"].grad.detach().numpy(),
    }
    return grads


def test_gradients_pytorch():
    np.random.seed(0)
    torch.manual_seed(0)

    seq_length = 5
    hidden_size = 10
    vocab_size = 15

    rnn = RNN(
        seq_length=seq_length,
        hidden_size=hidden_size,
        vocab_size=vocab_size,
        random_seed=0,
    )

    x = np.zeros((vocab_size, seq_length), dtype=np.float64)
    y = np.zeros_like(x)
    x_indices = np.random.randint(0, vocab_size, size=seq_length)
    y_indices = np.random.randint(0, vocab_size, size=seq_length)
    for t in range(seq_length):
        x[x_indices[t], t] = 1.0
        y[y_indices[t], t] = 1.0

    h_prev = np.zeros((hidden_size, 1), dtype=np.float64)

    rnn.forward(x, h_prev)
    dU, dW, dV, db, dc = rnn.backward(x, y, rnn.hs, rnn.ps)
    rnn_grads = {"U": dU, "W": dW, "V": dV, "b": db, "c": dc}

    torch_grads = compute_grads_with_torch(rnn, x, y_indices, h_prev)

    assert np.allclose(
        rnn_grads["U"], torch_grads["U"], atol=1e-6
    ), "Gradient U does not match!"
    assert np.allclose(
        rnn_grads["W"], torch_grads["W"], atol=1e-6
    ), "Gradient W does not match!"
    assert np.allclose(
        rnn_grads["V"], torch_grads["V"], atol=1e-6
    ), "Gradient V does not match!"
    assert np.allclose(
        rnn_grads["b"], torch_grads["b"], atol=1e-6
    ), "Gradient b does not match!"
    assert np.allclose(
        rnn_grads["c"], torch_grads["c"], atol=1e-6
    ), "Gradient c does not match!"



if __name__ == "__main__":
    test_gradients_pytorch()
    print("All gradient checks passed!")
    print("Starting training...")
    train_rnn()
