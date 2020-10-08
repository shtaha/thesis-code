import itertools
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf
from google.colab import drive
from sklearn.metrics import confusion_matrix, roc_curve
from sklearn.model_selection import train_test_split

drive.mount('/content/drive')


def print_class_weights(class_weight):
    print("Class", "Weight")
    for c in class_weight:
        print(f"    - {c}", "{:.5f}".format(class_weight[c]))


def compute_weight_bias(y):
    n_negative, n_positive = np.bincount(y.astype(int))
    n = n_negative + n_positive

    class_weight = {0: n / n_negative / 2.0, 1: n / n_positive / 2.0}
    initial_bias = np.log([n_positive / n_negative])

    return class_weight, initial_bias


def plot_metrics(training, y_train, y_val, save_dir=None):
    fig, ax = plt.subplots()
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

    fig, ax = plt.subplots()
    sns.heatmap(cm, annot=True, fmt="d", ax=ax)
    fig.suptitle("Confusion matrix: Threshold at p = {:.2f}".format(p))
    ax.set_title(name)
    ax.set_ylabel("Actual label")
    ax.set_xlabel("Predicted label")

    if save_dir:
        fig.savefig(os.path.join(save_dir, f"{name.lower()}-cm"))


def describe_results(metrics, results, y, name=None):
    print("\n    - Dataset", name)

    for metric, value in zip(metrics, results):
        if metric in ["tp", "fn", "tn", "fp"]:
            if metric in ["tp", "fn"]:
                c = 1
            else:
                c = 0

            n = np.sum(np.equal(y, c))
            rate = 100.0 * value / n

            ratio_str = "{}/{}".format(int(value), int(n))
            print(
                f"        - {metric.upper()}:",
                "{:<15}{:>8.2f} %".format(ratio_str, rate),
            )
        elif metric == "mcc":
            mcc_tf = float(value)
            print(f"        - {metric.capitalize()}:", "{:.4f}".format(mcc_tf))
        else:
            print(f"        - {metric.capitalize()}:", "{:.4f}".format(value))


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
    print(f"    - {name}:", "X, Y", "{:>20}, {}".format(str(x.shape), str(y.shape)))
    print("        - Positive labels:", "{:.2f} %".format(100 * y.mean()))
    print("        - Negative labels:", "{:.2f} %".format(100 * (1 - y).mean()))


class MatthewsCorrelationCoefficient(tf.keras.metrics.Metric):
    """
        Implementation following: https://scikit-learn.org/stable/modules/model_evaluation.html#matthews-corrcoef.
    """

    def __init__(self, threshold=0.5, name="mcc", **kwargs):
        super(MatthewsCorrelationCoefficient, self).__init__(name=name, **kwargs)

        self.threshold = threshold

        self.tps = tf.keras.metrics.TruePositives(thresholds=threshold, name="mcc-tp")
        self.fps = tf.keras.metrics.FalsePositives(thresholds=threshold, name="mcc-fp")
        self.tns = tf.keras.metrics.TrueNegatives(thresholds=threshold, name="mcc-tn")
        self.fns = tf.keras.metrics.FalseNegatives(thresholds=threshold, name="mcc-fn")

    def update_state(self, y_true, y_pred, sample_weight=None):
        self.tps.update_state(y_true, y_pred, sample_weight)
        self.fps.update_state(y_true, y_pred, sample_weight)
        self.tns.update_state(y_true, y_pred, sample_weight)
        self.fns.update_state(y_true, y_pred, sample_weight)

    def result(self):
        tp = self.tps.result()
        fp = self.fps.result()
        tn = self.tns.result()
        fn = self.fns.result()

        numerator = tp * tn - fp * fn
        denominator = tf.math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

        result = numerator / denominator
        return result

    def reset_states(self):
        self.tps.reset_states()
        self.fps.reset_states()
        self.tns.reset_states()
        self.fns.reset_states()


class ResidulaFCBlock(tf.keras.layers.Layer):
    def __init__(self, n_hidden, activation="relu", name=None, **kwargs):
        super(ResidulaFCBlock, self).__init__(name=name)

        self.dense_1 = tf.keras.layers.Dense(n_hidden, activation=None, **kwargs)
        self.dense_2 = tf.keras.layers.Dense(n_hidden, activation=None, **kwargs)

        self.activation = tf.keras.layers.Activation(activation)

    def call(self, input_tensor, training=False):
        x = self.dense_1(input_tensor)
        x = self.activation(x)

        x = self.dense_2(x)
        x = x + input_tensor
        x = self.activation(x)
        return x


"""
    Parameters
"""

random_seed = 0

# Data
test_frac = 0.10
val_frac = 0.10

downsampling_rate = 0.20

n_window_targets = 20
n_window_history = 2
n_window_forecasts = 1

use_forecasts = True
use_actions = True

# Model
model_type = "res"  # "fc" or "res"
dropout_rate = 0.2
l2_reg = 1e-4
n_hidden = 512
n_hidden_layers = 8

# Training
learning_rate = 1e-3
n_batch = 512
n_epochs = 200

# Prediction
threshold = 0.50

data = np.load(f"fc-data-h{n_window_history}-f{n_window_forecasts}")
X_all, Y_all = data["X_all"], data["Y_all"]
X, Y, mask_targets = data["X"], data["Y"], data["mask_targets"]

np.random.seed(random_seed)
tf.random.set_seed(random_seed)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=val_frac, random_state=random_seed
)

mask_test_neg = np.logical_and(~mask_targets, np.random.binomial(1, 0.08, mask_targets.size).astype(np.bool))
X_test = np.concatenate((X_val, X_all[mask_test_neg, :]))
Y_test = np.concatenate((Y_val, Y_all[mask_test_neg]))

class_weight, initial_bias = compute_weight_bias(Y)

print_dataset(X_all, Y_all, "All data")
print_dataset(X, Y, "Data")
print_dataset(X_train, Y_train, "Train")
print_dataset(X_val, Y_val, "Validation")
print_dataset(X_test, Y_test, "Test")
print_class_weights(class_weight)
print("Initial bias:", "{:.4f}".format(float(initial_bias)))
"""
    Model
"""

metrics = [
    tf.keras.metrics.TruePositives(thresholds=threshold, name="tp"),
    tf.keras.metrics.FalsePositives(thresholds=threshold, name="fp"),
    tf.keras.metrics.TrueNegatives(thresholds=threshold, name="tn"),
    tf.keras.metrics.FalseNegatives(thresholds=threshold, name="fn"),
    tf.keras.metrics.BinaryAccuracy(threshold=threshold, name="accuracy"),
    tf.keras.metrics.Precision(thresholds=threshold, name="precision"),
    tf.keras.metrics.Recall(thresholds=threshold, name="recall"),
    MatthewsCorrelationCoefficient(threshold=threshold, name="mcc"),
]

if l2_reg > 0:
    kwargs_reg = {
        "kernel_regularizer": tf.keras.regularizers.L2(l2=l2_reg),
        "bias_regularizer": tf.keras.regularizers.L2(l2=l2_reg),
    }
else:
    kwargs_reg = {}

input_dim = X.shape[-1]

tf.random.set_seed(random_seed)
if model_type == "fc":
    hidden_layers = [
        (
            tf.keras.layers.Dense(n_hidden, activation="relu", **kwargs_reg),
            tf.keras.layers.Dropout(dropout_rate),
        )
        for _ in range(n_hidden_layers)
    ]
    hidden_layers = list(itertools.chain(*hidden_layers))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                n_hidden, activation="relu", input_shape=(input_dim,), **kwargs_reg
            ),
            tf.keras.layers.Dropout(dropout_rate),
            *hidden_layers,
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                bias_initializer=tf.keras.initializers.Constant(initial_bias),
                **kwargs_reg,
            ),
        ]
    )
else:
    hidden_layers = [
        (
            ResidulaFCBlock(n_hidden, activation="relu", **kwargs_reg),
            tf.keras.layers.Dropout(dropout_rate),
        )
        for _ in range(n_hidden_layers // 2)
    ]
    hidden_layers = list(itertools.chain(*hidden_layers))

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Dense(
                n_hidden, activation="relu", input_shape=(input_dim,), **kwargs_reg
            ),
            tf.keras.layers.Dropout(dropout_rate),
            *hidden_layers,
            tf.keras.layers.Dropout(dropout_rate),
            tf.keras.layers.Dense(
                1,
                activation="sigmoid",
                bias_initializer=tf.keras.initializers.Constant(initial_bias),
                **kwargs_reg,
            ),
        ]
    )

model.compile(
    optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=metrics,
)

model_dir = None

"""
    Training
"""

training = model.fit(
    X_train,
    Y_train,
    epochs=n_epochs,
    batch_size=n_batch,
    class_weight=class_weight,
    validation_data=(X_val, Y_val),
    verbose=1,
)

"""
    Results
"""

model.summary()
plot_metrics(training, Y_train, Y_val, save_dir=model_dir)

results_train = model.evaluate(X_train, Y_train, batch_size=n_batch, verbose=0)
results_val = model.evaluate(X_val, Y_val, batch_size=n_batch, verbose=0)
results_test = model.evaluate(X_test, Y_test, batch_size=n_batch, verbose=0)
results_all = model.evaluate(X_all, Y_all, batch_size=n_batch, verbose=0)

Y_train_pred = model.predict(X_train, batch_size=n_batch)
Y_val_pred = model.predict(X_val, batch_size=n_batch)
Y_test_pred = model.predict(X_test, batch_size=n_batch)
Y_all_pred = model.predict(X_all, batch_size=n_batch, verbose=0)

describe_results(model.metrics_names, results_train, Y_train, name="Train")
describe_results(model.metrics_names, results_val, Y_val, name="Validation")
describe_results(model.metrics_names, results_test, Y_test, name="Test")
describe_results(model.metrics_names, results_all, Y_all, name="All")

plot_cm(Y_train, Y_train_pred, "Training", save_dir=model_dir)
plot_cm(Y_val, Y_val_pred, "Validation", save_dir=model_dir)
plot_cm(Y_test, Y_test_pred, "Test", save_dir=model_dir)
plot_cm(Y_all, Y_all_pred, "All", save_dir=model_dir)

plot_roc(
    [
        ("Training", Y_train, Y_train_pred),
        ("Validation", Y_val, Y_val_pred),
        ("Test", Y_test, Y_test_pred),
        ("All", Y_all, Y_all_pred),
    ],
    save_dir=model_dir,
)
