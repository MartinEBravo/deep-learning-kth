import pprint
from config import SUMMARY_PATH
import json
from cnn import CNN
from dataset import CIFAR10Dataset
from utils import compute_accuracy
from learning_rate import LearningRateScheduler
from dataclasses import dataclass
import numpy as np
from tqdm import tqdm
import time

pprint = pprint.PrettyPrinter(indent=4)


@dataclass
class ExperimentConfig:
    # Model parameters
    f: int = 4
    n_filters: int = 10
    hidden_dim: int = 50
    dataset: str = "CIFAR10"
    random_seed: int = 42

    # Cyclic Learning Rate parameters
    increasing: bool = False
    n_cycles: int = 3
    step: int = 800
    eta_min: float = 1e-5
    eta_max: float = 1e-1
    batch_size: int = 100
    iter_eval: int = 800

    # Regularization parameter
    lam: float = 0.003

    # Label smoothing
    label_smoothing: float = 0.0


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
            random_seed=config.random_seed,
        )

    def _train_model(self, dataset: CIFAR10Dataset):
        start_time = time.time()

        # Get data
        X_train, Y_train, _ = dataset.get_training_data()
        X_test, Y_test, _ = dataset.get_test_data()
        n_train, n_test = X_train.shape[3], X_test.shape[3]

        # Shuffle data
        indices_train, indices_test = np.arange(n_train), np.arange(n_test)
        np.random.shuffle(indices_test), np.random.shuffle(indices_train)
        X_train, Y_train = (X_train[:, :, :, indices_train], Y_train[:, indices_train])
        X_test, Y_test = (X_test[:, :, :, indices_test], Y_test[:, indices_test])

        # Initialize learning rate scheduler
        lr_scheduler = LearningRateScheduler(
            n_min=self.config.eta_min,
            n_max=self.config.eta_max,
            step_size=self.config.step,
            increasing=self.config.increasing,
            n_cycles=self.config.n_cycles,
        )
        iterations = lr_scheduler.get_total_iterations()
        cycle_iters = lr_scheduler.get_cycle_mid_end_iters()

        for it in tqdm(range(iterations)):
            # Mini-batch
            start = (it * self.config.batch_size) % n_train
            end = min(start + self.config.batch_size, n_train)
            X_train_batch, Y_train_batch = (
                X_train[:, :, :, start:end],
                Y_train[:, start:end],
            )

            # Forward + backward
            lr = lr_scheduler.get_lr(it)
            MX_train_batch = self.cnn.get_MX(X_train_batch)
            self.cnn.backward(
                MX_train_batch,
                Y_train_batch,
                lam=self.config.lam,
                learning_rate=lr,
                label_smoothing=self.config.label_smoothing,
            )

            if (it + 1) in cycle_iters or it == 0:
                current_cycle = next(
                    i + 1
                    for i, end_iter in enumerate(cycle_iters)
                    if it + 1 <= end_iter
                )
                avg_train_loss, avg_train_acc = self._compute_acc_loss(
                    X_train, Y_train, n_train // self.config.batch_size
                )
                avg_test_loss, avg_test_acc = self._compute_acc_loss(
                    X_test, Y_test, n_test // self.config.batch_size
                )
                self.logger.log(
                    config=vars(self.config),
                    cycle=current_cycle,
                    iteration=it + 1,
                    train_loss=avg_train_loss,
                    train_acc=avg_train_acc,
                    test_loss=avg_test_loss,
                    test_acc=avg_test_acc,
                    time_elapsed=time.time() - start_time,
                )

    def _compute_acc_loss(self, X, Y, n_batches):
        total_loss, total_acc = 0.0, 0.0

        for j in range(n_batches):
            start = j * self.config.batch_size
            end = start + self.config.batch_size
            X_batch = X[:, :, :, start:end]
            Y_batch = Y[:, start:end]

            MX_batch = self.cnn.get_MX(X_batch)
            p_val = self.cnn.forward(MX_batch)
            total_loss += self.cnn.compute_loss(
                p_val,
                Y_batch,
                self.config.lam,
                label_smoothing=self.config.label_smoothing,
            )
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
