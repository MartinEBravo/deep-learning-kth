from pathlib import Path
import numpy as np


# Load the book data
ROOT_DIR = Path(__file__).resolve().parent
book_path = ROOT_DIR / "goblet_book.txt"
fid = open(book_path, "r", encoding="utf-8")
book_data = fid.read()
fid.close()

# Create vocabulary and mappings
vocab = sorted(list(set(book_data)))
print(f"Vocabulary: {vocab}")
vocab_size = len(vocab)
print(f"Vocabulary Size: {vocab_size}")
char_to_index = {ch: i for i, ch in enumerate(vocab)}
index_to_char = {i: ch for i, ch in enumerate(vocab)}
char_to_onehot = {ch: np.eye(vocab_size, dtype=np.float32)[i].reshape(-1, 1) for i, ch in enumerate(vocab)}
onehot_to_char = {i: ch for i, ch in enumerate(vocab)}


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / (e_x.sum(axis=0, keepdims=True) + 1e-12)


class RNN:
    def __init__(self, seq_length=25, m=100, eta=0.001, random_seed=0):
        # Model parameters
        self.seq_length = seq_length
        self.m = m
        self.eta = eta
        self.random_seed = random_seed
        self.K = vocab_size  # Input size

        # Set random seed
        np.random.seed(self.random_seed)

        # Initialize weights
        self.U = (1/np.sqrt(2*self.m)) * np.random.randn(self.K, self.m)
        self.W = (1/np.sqrt(2*self.m)) * np.random.randn(self.m, self.m)
        self.V = (1/np.sqrt(2*self.m)) * np.random.randn(self.K, self.m)

        # Initialize biases
        self.b = np.zeros((self.m, 1))
        self.c = np.zeros((self.K, 1))


    def forward(self, x, h_prev):

        xs, hs, os, ps, a_s = {}, {}, {}, {}, {}
        hs[-1] = h_prev
        T = x.shape[1] 
        for t in range(T):
            xs[t] = x[:, [t]]
            a_s[t] = self.W @ hs[t-1] + self.U.T @ xs[t] + self.b
            hs[t] = np.tanh(a_s[t])
            os[t] = self.V @ hs[t] + self.c
            ps[t] = softmax(os[t])

        return xs, hs, os, ps, a_s
    
    def synthesize(self, h_prev, seed_char):
        x = char_to_onehot[seed_char]
        generated_text = seed_char
        h = h_prev
        print("Generating text:", end=" ")
        print(seed_char, end="")
        for t in range(self.seq_length):
            a = self.W @ h + self.U.T @ x + self.b
            h = np.tanh(a)
            o = self.V @ h + self.c
            p = softmax(o)
            cp = np.cumsum(p, axis=0)
            a = np.random.rand()
            idx = np.argmax(cp - a > 0)
            char = index_to_char[idx]
            generated_text += char
            x = char_to_onehot[char]
            print(char, end="")
        print()
        return generated_text
    

    def backward(self, x, y, hs, ps):
        dU = np.zeros_like(self.U)
        dW = np.zeros_like(self.W)
        dV = np.zeros_like(self.V)
        db = np.zeros_like(self.b)
        dc = np.zeros_like(self.c)
        dh_next = np.zeros((self.m, 1))

        T = x.shape[1]
        for t in reversed(range(T)):
            dy = ps[t] - y[:, [t]]
            dV += dy @ hs[t].T
            dc += dy
            dh = self.V.T @ dy + dh_next
            da = (1 - hs[t] ** 2) * dh
            db += da
            dW += da @ hs[t-1].T
            dU += x[:, [t]] @ da.T
            dh_next = self.W.T @ da

        for dparam in [dU, dW, dV, db, dc]:
            np.clip(dparam, -5, 5, out=dparam)

        return dU, dW, dV, db, dc
    
    def update_parameters(self):
        dU, dW, dV, db, dc = self.backward(self.x, self.y, self.hs, self.ps)
        self.U -= self.eta * dU
        self.W -= self.eta * dW
        self.V -= self.eta * dV
        self.b -= self.eta * db
        self.c -= self.eta * dc


def generate_dataset(seq_length):
    # Output an array of n_samples x seq_length x vocab_size
    X = []
    Y = []
    n_samples = (len(book_data) - 1) // seq_length
    for i in range(n_samples):
        x_seq = []
        y_seq = []
        for j in range(seq_length):
            char = book_data[i * seq_length + j]
            next_char = book_data[i * seq_length + j + 1]
            x_seq.append(char_to_onehot[char])
            y_seq.append(char_to_onehot[next_char])
        X.append(np.hstack(x_seq))
        Y.append(np.hstack(y_seq))
    return np.array(X), np.array(Y)


def train_rnn(rnn, X, Y, n_epochs=10):
    n_samples = X.shape[0]
    for epoch in range(n_epochs):
        epoch_loss = 0
        for i in range(n_samples):
            x = X[i]  # (K, T)
            y = Y[i]  # (K, T)
            rnn.x = x
            rnn.y = y
            xs, hs, os, ps, a_s = rnn.forward(x, h_prev=np.zeros((rnn.m, 1)))
            rnn.hs = hs
            rnn.ps = ps
            T = x.shape[1]
            loss = 0.0
            for t in range(T):
                target_ix = int(np.argmax(y[:, [t]]))
                prob = float(ps[t][target_ix, 0])
                loss += -np.log(prob + 1e-12)
            epoch_loss += loss
            rnn.update_parameters()
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {epoch_loss / n_samples}")

if __name__ == "__main__":
    seq_length = 25
    X, Y = generate_dataset(seq_length)
    print(X.shape, Y.shape)
    print(f"Generated dataset with {X.shape[0]} sequences of length {seq_length}.")
    rnn = RNN(seq_length=seq_length)
    train_rnn(rnn, X, Y, n_epochs=10000)
    rnn.synthesize(h_prev=np.zeros((rnn.m, 1)), seed_char='T')