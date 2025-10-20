from pathlib import Path
import pprint
from config import SUMMARY_PATH
import json
from cnn import CNN
from dataset import CIFAR10Dataset
from utils import cyclical_learning_rate, compute_accuracy
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import pprint
import time

pprint = pprint.PrettyPrinter(indent=4)


@dataclass
class ExperimentConfig:
    # Model parameters
    f: int = 4
    n_filters: int = 10
    hidden_dim: int = 50
    dataset: str = "CIFAR10"

    # Cyclic Learning Rate parameters
    n_cycles = 3
    step: int = 800
    eta_min: float = 1e-5
    eta_max: float = 1e-1
    batch_size: int = 100

    # Regularization parameter
    lam: float = 0.003


class ExperimentLogger:
    def __init__(self, name: str):
        self.id = f"{name}_{int(time.time())}"
        self.path = f"{self.id}.jsonl"
        self._create_jsonl_log_file()

    def _create_jsonl_log_file(self):
        with open(SUMMARY_PATH / self.path, "w") as f:
            f.write("")
        pprint.pprint(f"[LOGGER]: Created log file at {self.path}")

    def log(self, **kwargs):
        with open(SUMMARY_PATH / self.path, "a") as f:
            json_line = json.dumps({**kwargs, "id": self.id})
            f.write(json_line + "\n")


class Experiment:
    def __init__(self, config: ExperimentConfig, logger: ExperimentLogger):
        self.config = config
        self.logger = logger
        self.cnn = CNN(
            f=config.f,
            n_filters=config.n_filters,
            hidden_dim=config.hidden_dim,
        )

    def _train_model(self, dataset: CIFAR10Dataset):
        start_time = time.time()

        # Get data
        X_train, Y_train, _ = dataset.get_training_data()
        X_test, Y_test, _ = dataset.get_test_data()

        # Amount of samples
        n_train, n_test = X_train.shape[3], X_test.shape[3]

        # Shuffle data
        indices_train, indices_test = np.arange(n_train), np.arange(n_test)
        np.random.shuffle(indices_test), np.random.shuffle(indices_train)
        X_train, Y_train = (X_train[:, :, :, indices_train], Y_train[:, indices_train])
        X_test, Y_test = (X_test[:, :, :, indices_test], Y_test[:, indices_test])

        iterations = self.config.n_cycles * (2 * self.config.step)

        for it in tqdm(range(iterations)):
            # Mini-batch
            start = (it * self.config.batch_size) % n_train
            end = min(start + self.config.batch_size, n_train)
            X_train_batch, Y_train_batch = (
                X_train[:, :, :, start:end],
                Y_train[:, start:end],
            )

            # Forward + backward
            lr = cyclical_learning_rate(
                self.config.eta_min,
                self.config.eta_max,
                self.config.step,
                it,
            )
            self.cnn.backward(X_train_batch, Y_train_batch, self.config.lam, lr)

            if (it + 1) % 800 == 0 or it == 0:
                avg_train_loss, avg_train_acc = self._compute_acc_loss(
                    X_train, Y_train, n_train // self.config.batch_size
                )
                avg_test_loss, avg_test_acc = self._compute_acc_loss(
                    X_test, Y_test, n_test // self.config.batch_size
                )
                self.logger.log(
                    config=vars(self.config),
                    iteration=it + 1,
                    train_loss=avg_train_loss,
                    train_acc=avg_train_acc,
                    test_loss=avg_test_loss,
                    test_acc=avg_test_acc,
                    time_elapsed=time.time() - start_time  # seconds elapsed since last log
                )

    def _compute_acc_loss(self, X, Y, n_batches):
        total_loss, total_acc = 0.0, 0.0

        for j in range(n_batches):
            s = j * self.config.batch_size
            e = s + self.config.batch_size
            X_batch = X[:, :, :, s:e]
            Y_batch = Y[:, s:e]

            p_val = self.cnn.forward(X_batch)
            total_loss += self.cnn.compute_loss(p_val, Y_batch, self.config.lam)
            total_acc += compute_accuracy(p_val, np.argmax(Y_batch, axis=0))

        avg_loss = total_loss / n_batches
        avg_acc = total_acc / n_batches * 100

        return avg_loss, avg_acc
    
    def run(self):
        if self.config.dataset == "CIFAR10":
            dataset = CIFAR10Dataset()
            self._train_model(dataset)
        else:
            raise NotImplementedError("Dataset not implemented yet.")
