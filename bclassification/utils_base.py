import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from lib.constants import Constants as Const
from lib.visualizer import pprint


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

    metrics = ["accuracy", "precision", "recall", "fp", "fn", "mcc"]
    fig, _ = plt.subplots(3, 2, figsize=(16, 12))
    for i, metric in enumerate(metrics):
        ax = plt.subplot(3, 2, i + 1)

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

        ax.set_ylabel(name)

        if metric == "mcc":
            # ax.set_ylim([-1.0, 1.0])
            pass
        else:
            ax.set_ylim([0.0, 1.0])

        ax.legend()

    if save_dir:
        fig.savefig(os.path.join(save_dir, "training-metrics"))
