import numpy as np
from experiment import ExperimentConfig, Experiment, ExperimentLogger
from test_cnn import test_convolution, test_forward, test_backward, test_pytorch

np.random.seed(42)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="CNN Experiment Runner")
    parser.add_argument(
        "--arch",
        type=str,
        required=True,
        help="Architecture name for the experiment (e.g., 'arch1', 'arch2')",
    )

    args = parser.parse_args()
    arch_name = args.arch

    # Tests
    if arch_name == "tests":
        test_convolution()
        test_forward()
        test_backward()
        test_pytorch()
        print("All tests passed!")
        exit(0)

    # Experiment 1
    elif arch_name == "arch1":
        config = ExperimentConfig(
            f=2,
            n_filters=3,
            hidden_dim=50,
        )

    # Experiment 2
    elif arch_name == "arch2":
        config = ExperimentConfig(
            f=4,
            n_filters=10,
            hidden_dim=50,
        )

    # Experiment 3
    elif arch_name == "arch3":
        config = ExperimentConfig(
            f=8,
            n_filters=40,
            hidden_dim=50,
        )

    # Experiment 4
    elif arch_name == "arch4":
        config = ExperimentConfig(
            f=16,
            n_filters=160,
            hidden_dim=50,
        )

    # Experiment 2 Longer with increasing LR
    elif arch_name == "arch2_long":
        config = ExperimentConfig(
            f=4,
            n_filters=10,
            hidden_dim=50,
            increasing=True,
        )

    # Experiment 3 Longer with Increasing LR
    elif arch_name == "arch3_long":
        config = ExperimentConfig(
            f=8,
            n_filters=40,
            hidden_dim=50,
            increasing=True,
        )

    # Experiment 2 Longer with more n_filters
    elif arch_name == "arch2_long_more_filters":
        config = ExperimentConfig(
            f=4,
            n_filters=40,
            hidden_dim=50,
            increasing=True,
        )

    # Experiment 5: Label Smoothing Baseline
    elif arch_name == "arch5_baseline":
        config = ExperimentConfig(
            f=4,
            n_filters=40,
            hidden_dim=300,
            increasing=False,
            n_cycles=4,
            lam=0.0025,
            label_smoothing=0.0,
        )

    # Experiment 5: Label Smoothing
    elif arch_name == "arch5_label_smoothing":
        config = ExperimentConfig(
            f=4,
            n_filters=40,
            hidden_dim=300,
            increasing=False,
            n_cycles=4,
            lam=0.0025,
            label_smoothing=0.1,
        )

    else:
        raise ValueError(f"Unknown architecture name: {arch_name}")

    logger = ExperimentLogger(arch_name)
    experiment = Experiment(config, logger)
    experiment.run()
