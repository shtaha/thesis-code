import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from lib.constants import Constants as Const
from lib.visualizer import pprint


def plot_metrics(history, y_train, y_val, save_dir=None):
    epochs = history["epochs"]

    fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
    ax.plot(epochs, history["loss"], label="Training", lw=0.5)
    ax.plot(epochs, history["val_loss"], label="Validation", lw=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()

    if save_dir:
        fig.savefig(os.path.join(save_dir, "training-loss"))

    metrics = ["accuracy", "precision", "recall", "fp", "fn", "mcc"]
    fig, _ = plt.subplots(3, 2, figsize=(16, 12))
    for i, metric in enumerate(metrics):
        ax = plt.subplot(3, 2, i + 1)

        train_metric = history[metric]
        val_metric = history["val_" + metric]

        if metric == "fn":
            metric = "fnr"
            train_n_pos = np.sum(np.equal(y_train, 1))
            val_n_pos = np.sum(np.equal(y_val, 1))

            train_metric = np.divide(train_metric, train_n_pos + 1e-9)
            val_metric = np.divide(val_metric, val_n_pos + 1e-9)
        elif metric == "fp":
            metric = "fpr"
            train_n_neg = np.sum(np.equal(y_train, 0))
            val_n_neg = np.sum(np.equal(y_val, 0))

            train_metric = np.divide(train_metric, train_n_neg + 1e-9)
            val_metric = np.divide(val_metric, val_n_neg + 1e-9)

        name = metric.replace("_", " ").upper()
        ax.plot(epochs, train_metric, label="Training", lw=0.5)
        ax.plot(epochs, val_metric, label="Validation", lw=0.5)
        ax.set_xlabel("Epoch")

        ax.set_ylabel(name)

        if metric == "mcc":
            # ax.set_ylim([-1.0, 1.0])
            pass
        else:
            ax.set_ylim([0.0, 1.0])

        ax.legend()

    if save_dir:
        fig.savefig(os.path.join(save_dir, "training-metrics"))


def plot_roc(triplets, save_dir=None):
    fig, ax = plt.subplots(figsize=(16, 5))
    for label, Y, Y_pred in triplets:
        fp, tp, _ = roc_curve(Y, Y_pred)
        ax.plot(fp, tp, label=label, lw=2)

    ax.set_xlabel("False positives")
    ax.set_ylabel("True positives")
    ax.grid(True)
    ax.legend(loc="lower right")

    if save_dir:
        fig.savefig(os.path.join(save_dir, "roc"))


def plot_cm(labels, predictions, name, p=0.5, save_dir=None):
    cm = confusion_matrix(labels, predictions > p)

    fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    fig.suptitle("Confusion matrix: Threshold at p = {:.2f}".format(p))
    ax.set_title(name)
    ax.set_ylabel("Actual label")
    ax.set_xlabel("Predicted label")

    if save_dir:
        fig.savefig(os.path.join(save_dir, f"{name.lower()}-cm"))


def describe_results(results, y, name=None):
    pprint("\n    - Dataset", name)

    for metric, value in results.items():
        if metric in ["tp", "fn", "tn", "fp"]:
            if metric in ["tp", "fn"]:
                c = 1
            else:
                c = 0

            n = np.sum(np.equal(y, c))
            rate = 100.0 * value / n

            ratio_str = "{}/{}".format(int(value), int(n))
            pprint(
                f"        - {metric.upper()}:",
                "{:<15}{:>8.2f} %".format(ratio_str, rate),
            )
        elif metric == "mcc":
            mcc_tf = float(value)
            pprint(f"        - {metric.capitalize()}:", "{:.4f}".format(mcc_tf))
        else:
            pprint(f"        - {metric.capitalize()}:", "{:.4f}".format(value))


def print_graph_dims(graph_dims):
    pprint("Graph", "Dimension")
    for field, value in graph_dims.items():
        pprint(field, value)
