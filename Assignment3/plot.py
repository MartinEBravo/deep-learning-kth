from cProfile import label
from config import REPORT_IMG_DIR, SUMMARY_PATH
import pandas as pd
import matplotlib.pyplot as plt

# Global variables
select_var = {
    "iterations": 0,
    "train_loss": 1,
    "train_acc": 2,
    "test_loss": 3,
    "test_acc": 4,
    "time_elapsed": 5,
}


def select_iterations_losses_and_acc(df: pd.DataFrame):
    iterations, train_loss, train_acc, test_loss, test_acc, time_elapsed = (
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i in range(len(df)):
        row = df.iloc[i]
        iterations.append(row["iteration"])
        train_loss.append(row["train_loss"])
        train_acc.append(row["train_acc"])
        test_loss.append(row["test_loss"])
        test_acc.append(row["test_acc"])
        time_elapsed.append(row["time_elapsed"])
    return iterations, train_loss, train_acc, test_loss, test_acc, time_elapsed


# Exercise 3
arch1 = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch1_1761014439.jsonl", lines=True)
)
arch2 = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch2_1761014453.jsonl", lines=True)
)
arch3 = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch3_1761014494.jsonl", lines=True)
)
arch4 = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch4_1761014626.jsonl", lines=True)
)

# Train for longer
arch2_long = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch2_long_1761014970.jsonl", lines=True)
)
arch3_long = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch3_long_1761015113.jsonl", lines=True)
)
arch2_long_more_filters = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch2_long_more_filters_1761016822.jsonl", lines=True)
)

# Exercise 4
arch5_baseline = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch5_baseline_1761019769.jsonl", lines=True)
)
arch5_label_smoothing = select_iterations_losses_and_acc(
    pd.read_json(SUMMARY_PATH / "arch5_label_smoothing_1761021952.jsonl", lines=True)
)


def plot_curves(archs, archs_names = ["arch1" ,"arch2", "arch3", "arch4"]):
    for i in range(len(archs)):
        arch_stats = archs[i]
        plt.plot(
            arch_stats[select_var["iterations"]],
            arch_stats[select_var["test_acc"]],
            "g",
            label="Test Accuracy",
        )
        plt.plot(
            arch_stats[select_var["iterations"]],
            arch_stats[select_var["train_acc"]],
            "b",
            label="Train Accuracy",
        )
        plt.xlabel("Iterations")
        plt.ylim(0, 100)
        plt.ylabel("Accuracy")
        plt.legend()
        plt.title(f"{archs_names[i]} Accuracy")
        plt.savefig(REPORT_IMG_DIR / f"accuracy_arch_{archs_names[i]}.pdf")
        plt.clf()

        arch_stats = archs[i]
        plt.plot(
            arch_stats[select_var["iterations"]],
            arch_stats[select_var["test_loss"]],
            "g",
            label="Test Loss",
        )
        plt.plot(
            arch_stats[select_var["iterations"]],
            arch_stats[select_var["train_loss"]],
            "b",
            label="Train Loss",
        )
        plt.xlabel("Iterations")
        plt.ylabel("Loss")
        plt.ylim(0, 3)
        plt.legend()
        plt.title(f"{archs_names[i]} Loss")
        plt.savefig(REPORT_IMG_DIR / f"loss_arch_{archs_names[i]}.pdf")
        plt.clf()


def plot_compare_times(archs):
    archs_names = ["arch1" ,"arch2", "arch3", "arch4"]
    counts = [
        archs[0][select_var["time_elapsed"]][-1],
        archs[1][select_var["time_elapsed"]][-1],
        archs[2][select_var["time_elapsed"]][-1],
        archs[3][select_var["time_elapsed"]][-1],
    ]
    colors = ["red", "blue", "orange", "green"]
    
    plt.bar(archs_names, counts, color=colors, label=archs_names)
    plt.xlabel("Architecture")
    plt.ylabel("Seconds")
    plt.title("Training time")
    plt.savefig(REPORT_IMG_DIR / f"compare_times.pdf")
    plt.clf()


if __name__ == "__main__":
    plot_curves([arch1, arch2, arch3, arch4])
    plot_compare_times([arch1, arch2, arch3, arch4])
    plot_curves([arch2_long, arch3_long, arch2_long_more_filters], archs_names=["arch2_long", "arch3_long", "arch2_long_more_filters"])
    plot_curves([arch5_baseline, arch5_label_smoothing], archs_names=["arch5_baseline", "arch5_label_smoothing"])
