import matplotlib.pyplot as plt
from files import ensure_dir
from config import REPORT_IMG_DIR


def plot_bar_chart(
    results, value_fn, ylabel, filename, title=None, value_format="{:.2f}"
):
    ensure_dir(REPORT_IMG_DIR)
    labels = [res["name"] for res in results]
    values = [value_fn(res) for res in results]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(labels, values, color="#4C72B0")
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20)
    if title:
        ax.set_title(title)

    for bar, value in zip(bars, values):
        ax.annotate(
            value_format.format(value),
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 5),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / filename, dpi=200)
    plt.close(fig)


def plot_loss_curves(results, sets, filename, title=None, ylabel="Loss"):
    ensure_dir(REPORT_IMG_DIR)
    fig, ax = plt.subplots(figsize=(10, 6))
    for res in results:
        iterations = res["metrics"]["iterations"]
        if not iterations:
            continue
        for set_name in sets:
            if set_name == "train":
                metric_key = "train_loss"
            else:
                metric_key = f"{set_name}_loss"
            if metric_key not in res["metrics"]:
                continue
            linestyle = "-" if set_name == "train" else "--"
            label = f"{res['name']} {set_name.capitalize()}"
            ax.plot(
                iterations, res["metrics"][metric_key], linestyle=linestyle, label=label
            )

    ax.set_xlabel("Iteration")
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.3)
    fig.tight_layout()
    fig.savefig(REPORT_IMG_DIR / filename, dpi=200)
    plt.close(fig)
