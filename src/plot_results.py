import os
import re

import matplotlib.pyplot as plt
import numpy as np


RESULTS_DIR = "results"

METRIC_FILES = {
    "Rule-based": "rule_based_metrics.txt",
    "ML snapshot": "ml_detector_metrics.txt",
    "ML temporal": "ml_temporal_metrics.txt",
}


def read_text(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def extract_score(text, name):
    match = re.search(rf"{name}:\s+([0-9.]+)", text)
    if not match:
        raise ValueError(f"Could not find {name}")
    return float(match.group(1))


def extract_confusion_matrix(text):
    pattern = r"\[\[(\d+)\s+(\d+)\]\s*\n\s*\[\s*(\d+)\s+(\d+)\]\]"
    match = re.search(pattern, text)
    if not match:
        raise ValueError("Could not parse confusion matrix")
    values = [int(x) for x in match.groups()]
    return np.array([[values[0], values[1]], [values[2], values[3]]])


def extract_replay_rate(text):
    match = re.search(r"replay_like_values:\s+([0-9.]+)", text)
    if not match:
        return None
    return float(match.group(1))


def load_metrics():
    data = {}

    for name, filename in METRIC_FILES.items():
        path = os.path.join(RESULTS_DIR, filename)
        text = read_text(path)

        data[name] = {
            "accuracy": extract_score(text, "Accuracy"),
            "precision": extract_score(text, "Precision"),
            "recall": extract_score(text, "Recall"),
            "f1": extract_score(text, "F1-score"),
            "cm": extract_confusion_matrix(text),
            "replay_rate": extract_replay_rate(text),
        }

    return data


def plot_metric_comparison(data):
    names = list(data.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]

    x = np.arange(len(names))
    width = 0.18

    fig, ax = plt.subplots(figsize=(10, 5))

    for i, metric in enumerate(metrics):
        values = [data[name][metric] for name in names]
        ax.bar(x + (i - 1.5) * width, values, width, label=metric)

    ax.set_title("Detector metric comparison")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x)
    ax.set_xticklabels(names)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "detector_metric_comparison.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Saved: {output_path}")


def plot_confusion_matrices(data):
    for name, values in data.items():
        cm = values["cm"]

        fig, ax = plt.subplots(figsize=(5, 4))
        image = ax.imshow(cm)

        ax.set_title(f"{name} confusion matrix")
        ax.set_xlabel("Predicted label")
        ax.set_ylabel("True label")
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(["normal", "attack"])
        ax.set_yticklabels(["normal", "attack"])

        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center")

        fig.colorbar(image, ax=ax)
        fig.tight_layout()

        safe_name = name.lower().replace(" ", "_").replace("-", "")
        output_path = os.path.join(RESULTS_DIR, f"{safe_name}_confusion_matrix.png")
        fig.savefig(output_path, dpi=160)
        plt.close(fig)

        print(f"Saved: {output_path}")


def plot_replay_detection(data):
    names = []
    rates = []

    for name, values in data.items():
        if values["replay_rate"] is not None:
            names.append(name)
            rates.append(values["replay_rate"])

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(names, rates)

    ax.set_title("Replay-like attack detection rate")
    ax.set_ylabel("Detection rate")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    output_path = os.path.join(RESULTS_DIR, "replay_detection_comparison.png")
    fig.savefig(output_path, dpi=160)
    plt.close(fig)

    print(f"Saved: {output_path}")


def main():
    data = load_metrics()

    plot_metric_comparison(data)
    plot_confusion_matrices(data)
    plot_replay_detection(data)


if __name__ == "__main__":
    main()
