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

    # Experiment 1
    elif arch_name == "arch1":
        config = ExperimentConfig(
            f=2,
            n_filters=3,
            hidden_dim=50,
        )
        logger = ExperimentLogger("arch1")
        experiment = Experiment(config, logger)
        experiment.run()

    # Experiment 2
    elif arch_name == "arch2":
        config = ExperimentConfig(
            f=4,
            n_filters=10,
            hidden_dim=50,
        )
        logger = ExperimentLogger("arch2")
        experiment = Experiment(config, logger)
        experiment.run()

    # Experiment 3
    elif arch_name == "arch3":
        config = ExperimentConfig(
            f=8,
            n_filters=40,
            hidden_dim=50,
        )
        logger = ExperimentLogger("arch3")
        experiment = Experiment(config, logger)
        experiment.run()

    # Experiment 4
    elif arch_name == "arch4":
        config = ExperimentConfig(
            f=16,
            n_filters=160,
            hidden_dim=50,
        )
        logger = ExperimentLogger("arch4")
        experiment = Experiment(config, logger)
        experiment.run()

    else:
        raise ValueError(f"Unknown architecture name: {arch_name}")
