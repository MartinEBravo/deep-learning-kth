import numpy as np


class LearningRateScheduler:
    def __init__(self, n_min, n_max, step_size, increasing=True):
        self.n_min = n_min
        self.n_max = n_max
        self.step_size = step_size
        self.increasing = increasing

    def get_lr(self, iteration):
        if self.increasing:
            return self._cyclical_learning_rate_increasing(
                self.n_min, self.n_max, self.step_size, iteration
            )
        else:
            return self._cyclical_learning_rate(
                self.n_min, self.n_max, self.step_size, iteration
            )

    def _cyclical_learning_rate(self, n_min, n_max, step_size, iteration):
        cycle = np.floor(1 + iteration / (2 * step_size))
        x = np.abs(iteration / step_size - 2 * cycle + 1)
        lr = n_min + (n_max - n_min) * np.maximum(0, (1 - x))
        return lr

    def _cyclical_learning_rate_increasing(self, n_min, n_max, step_size, iteration):
        remaining = iteration
        cycle = 0
        current_step = step_size
        cycle_length = current_step * 2

        while remaining >= cycle_length:
            remaining -= cycle_length
            cycle += 1
            current_step = step_size * (2**cycle)
            cycle_length = current_step * 2

        progress = remaining / current_step
        if progress <= 1.0:
            lr = n_min + (n_max - n_min) * progress
        else:
            lr = n_max - (n_max - n_min) * (progress - 1.0)
        return lr
