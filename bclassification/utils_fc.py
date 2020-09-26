import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve

from lib.constants import Constants as Const
from lib.visualizer import pprint


def obs_to_vects(obs, tc):
    lines_or_to_sub_bus = tc.lines_or_to_sub_bus(obs)
    lines_ex_to_sub_bus = tc.lines_ex_to_sub_bus(obs)
    gens_to_sub_bus = tc.gens_to_sub_bus(obs)
    loads_to_sub_bus = tc.loads_to_sub_bus(obs)

    prod_1 = np.multiply(tc.sub_bus_mask(gens_to_sub_bus, 1, np.float), obs.prod_p)
    prod_2 = np.multiply(tc.sub_bus_mask(gens_to_sub_bus, 2, np.float), obs.prod_p)
    prod_vect = np.concatenate((prod_1, prod_2))

    load_1 = np.multiply(tc.sub_bus_mask(loads_to_sub_bus, 1, np.float), obs.load_p)
    load_2 = np.multiply(tc.sub_bus_mask(loads_to_sub_bus, 2, np.float), obs.load_p)
    load_vect = np.concatenate((load_1, load_2))

    p_ors = []
    for sub_bus_or in [1, 2]:
        for sub_bus_ex in [1, 2]:
            mask = np.multiply(
                tc.sub_bus_mask(lines_or_to_sub_bus, sub_bus_or, np.float),
                tc.sub_bus_mask(lines_ex_to_sub_bus, sub_bus_ex, np.float),
            )
            p_or = np.multiply(obs.p_or, mask)
            p_ors.append(p_or)
    p_ors.append(obs.rho)

    line_vect = np.concatenate(p_ors)

    return prod_vect, load_vect, line_vect


def obs_to_vects_with_tc(tc):
    return lambda obs: obs_to_vects(obs, tc)


def obs_to_vect(obs, tc):
    return np.concatenate(obs_to_vects(obs, tc))


def obs_to_vect_with_tc(tc):
    return lambda obs: obs_to_vect(obs, tc)


def obs_vects_to_vect(obs_vects):
    return np.concatenate(obs_vects)


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


def print_dataset(x, y, name):
    pprint(f"    - {name}:", "X, Y", "{:>20}, {}".format(str(x.shape), str(y.shape)))
    pprint("        - Positive labels:", "{:.2f} %".format(100 * y.mean()))
    pprint("        - Negative labels:", "{:.2f} %".format(100 * (1 - y).mean()))
