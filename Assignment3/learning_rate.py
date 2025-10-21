import numpy as np


class LearningRateScheduler:
    def __init__(self, n_min, n_max, step_size, n_cycles, increasing=True):
        self.n_min = n_min
        self.n_max = n_max
        self.step_size = step_size
        self.n_cycles = n_cycles
        self.increasing = increasing
        self.lr_functions = {
            True: self._cyclical_learning_rate_increasing,
            False: self._cyclical_learning_rate,
        }
        self.cycle_end_iters = {
            True: np.cumsum([2 * self.step_size * (2**i) for i in range(self.n_cycles)])
            .astype(int)
            .tolist(),
            False: np.cumsum([2 * self.step_size] * self.n_cycles).astype(int).tolist(),
        }
        self.cycle_mid_iters = {
            True: (
                np.cumsum([2 * self.step_size * (2**i) for i in range(self.n_cycles)])
                - np.array([self.step_size * (2**i) for i in range(self.n_cycles)])
            )
            .astype(int)
            .tolist(),
            False: (
                np.cumsum([2 * self.step_size] * self.n_cycles)
                - np.array([self.step_size] * self.n_cycles)
            )
            .astype(int)
            .tolist(),
        }

    def get_lr(self, iteration):
        return self.lr_functions[self.increasing](
            self.n_min, self.n_max, self.step_size, iteration
        )

    def get_total_iterations(self):
        multiplier = (1 << self.n_cycles) - 1 if self.increasing else self.n_cycles
        total_iterations = 2 * self.step_size * multiplier
        last_iter = self.cycle_end_iters[self.increasing][-1]
        assert total_iterations == last_iter, (
            "Total iterations do not match the last cycle end iteration."
        )
        return total_iterations

    def get_cycle_mid_end_iters(self):
        return np.sort(
            np.array(
                self.cycle_mid_iters[self.increasing]
                + self.cycle_end_iters[self.increasing]
            )
        ).tolist()

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
        lr = (
            n_min + (n_max - n_min) * progress
            if progress <= 1.0
            else n_max - (n_max - n_min) * (progress - 1.0)
        )
        return lr
