import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from sklearn.metrics import confusion_matrix, roc_curve

from lib.constants import Constants as Const
from lib.visualizer import pprint


class TrainingHistory(defaultdict):
    def __init__(self):
        super(TrainingHistory, self).__init__(list)

    def update_history(self, history, epoch, prefix=None):
        if epoch not in self["epochs"]:
            self["epochs"].append(epoch)

        if history:
            for metric_name, metric_value in history.items():
                metric_value = self.to_numpy(metric_value)

                if prefix:
                    self[prefix + metric_name].append(metric_value)
                else:
                    self[metric_name].append(metric_value)

    @staticmethod
    def to_numpy(value):
        if isinstance(value, tf.Tensor):
            value = value.numpy()

        return value

    @property
    def epoch(self):
        return np.unique(self["epochs"]).tolist()

    @property
    def history(self):
        return self


def print_dataset(x, y, name):
    if isinstance(x, np.ndarray):
        pprint(
            f"    - {name}:", "X, Y", "{:>20}, {}".format(str(x.shape), str(y.shape))
        )
    elif isinstance(x, list) and isinstance(x[0], dict):
        pprint(f"    - {name}:", "X, Y", "{:>20}, {}".format(len(x), str(y.shape)))
        for field in x[0]:
            if np.equal(x[0][field].shape, x[1][field].shape).all():
                pprint(f"        - X: {field}", x[0][field].shape)
            else:
                raise ValueError("Dimension mismatch.")
    else:
        raise ValueError("Unknown data structure.")

    pprint("        - Positive labels:", "{:.2f} %".format(100 * y.mean()))
    pprint("        - Negative labels:", "{:.2f} %\n".format(100 * (1 - y).mean()))


def plot_feature_dist(data, n_cols=4, n_rows=None, save_dir=None):
    if not n_rows:
        n_rows = np.ceil(data.shape[-1] / float(n_cols)).astype(int)

    fig = plt.figure(figsize=(16, n_rows * 3))
    ax_0 = plt.subplot(n_rows, n_cols, 1)
    for j in range(data.shape[-1]):
        if j == n_rows * n_cols:
            break

        x = data[:, j]

        if j == 0:
            ax = ax_0
        else:
            ax = plt.subplot(n_rows, n_cols, j + 1, sharex=ax_0)

        ax.set_title(f"Feature {j}" + "\t[{:.2f}, {:.2f}]".format(np.min(x), np.max(x)))
        sns.histplot(data=x, ax=ax, binwidth=0.10)

    fig.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, "X-dist"))


def print_class_weights(class_weight):
    pprint("Class", "Weight")
    for c in class_weight:
        pprint(f"    - {c}", "{:.5f}".format(class_weight[c]))


def compute_weight_bias(y):
    n_negative, n_positive = np.bincount(y.astype(int))
    n = n_negative + n_positive

    class_weight = {0: n / n_negative / 2.0, 1: n / n_positive / 2.0}
    initial_bias = np.log([n_positive / n_negative])

    return class_weight, initial_bias


def plot_metric_group(training, metric_names, y_train, y_val, save_dir=None):
    fig, axes = plt.subplots(ncols=len(metric_names), figsize=(12, 4))
    for i, metric in enumerate(metric_names):
        ax = axes[i]

        epochs = training.epoch
        train_metric = training.history[metric]
        val_metric = training.history["val_" + metric]

        if metric == "fn":
            metric = "fnr"
            train_n_pos = np.sum(np.equal(y_train, 1))
            val_n_pos = np.sum(np.equal(y_val, 1))

            train_metric = np.divide(train_metric, train_n_pos)
            val_metric = np.divide(val_metric, val_n_pos)
        elif metric == "fp":
            metric = "fpr"
            train_n_neg = np.sum(np.equal(y_train, 0))
            val_n_neg = np.sum(np.equal(y_val, 0))

            train_metric = np.divide(train_metric, train_n_neg)
            val_metric = np.divide(val_metric, val_n_neg)

        name = metric.replace("_", " ").upper()
        ax.plot(epochs, train_metric, label="Training", lw=0.5)
        ax.plot(epochs, val_metric, label="Validation", lw=0.5)
        ax.set_xlabel("Epoch")

        # ax.set_ylabel(name)
        ax.set_title(name)

        if metric == "mcc":
            # ax.set_ylim([-1.0, 1.0])
            pass
        else:
            ax.set_ylim([0.0, 1.0])

        ax.legend()

    if save_dir:
        fig.savefig(os.path.join(save_dir, "training-" + "-".join(metric_names)))


def plot_metrics(training, y_train, y_val, save_dir=None):
    fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
    ax.plot(training.epoch, training.history["loss"], label="Training", lw=0.5)
    ax.plot(training.epoch, training.history["val_loss"], label="Validation", lw=0.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_yscale("log")
    ax.legend()

    if save_dir:
        fig.savefig(os.path.join(save_dir, "training-loss"))

    plot_metric_group(
        training, ["accuracy", "mcc", "auc"], y_train, y_val, save_dir=save_dir
    )
    plot_metric_group(
        training, ["precision", "recall"], y_train, y_val, save_dir=save_dir
    )
    plot_metric_group(training, ["fp", "fn"], y_train, y_val, save_dir=save_dir)


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


def describe_results(metrics, results, y, name=None):
    pprint("\n    - Dataset", name)

    for metric, value in zip(metrics, results):
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
