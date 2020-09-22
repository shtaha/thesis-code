import os
import time

import tensorflow as tf
from graph_nets import utils_tf

from experience import load_experience
from lib.action_space import is_do_nothing_action
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.gns import (
    obses_to_cgraphs,
    tf_batched_graph_dataset,
    get_graph_feature_dimensions,
    GraphNetworkSwitching,
)
from lib.run_utils import create_logger
from lib.visualizer import Visualizer, pprint

Visualizer()

experience_dir = make_dir(os.path.join(Const.EXPERIENCE_DIR, "data-aug"))
results_dir = make_dir(os.path.join(Const.RESULTS_DIR, "binary-linear"))

agent_name = "agent-mip"
case_name = "l2rpn_2019_art"
env_dc = True
verbose = False

case_results_dir = make_dir(os.path.join(results_dir, f"{case_name}-{env_pf(env_dc)}"))
create_logger(logger_name=f"{case_name}-{env_pf(env_dc)}", save_dir=case_results_dir)

case, collector = load_experience(case_name, agent_name, experience_dir, env_dc=env_dc)
obses, actions, rewards, dones = collector.aggregate_data()

"""
    Parameters
"""
n_window = 2
n_batch = 32
n_epochs = 20

"""
    Datasets
"""
max_length = 100
cgraphs = obses_to_cgraphs(obses, dones, case, n_window=n_window)
labels = is_do_nothing_action(actions, case.env)

graph_dims = get_graph_feature_dimensions(cgraphs=cgraphs)
graph_dataset = tf_batched_graph_dataset(cgraphs, n_batch=n_batch, **graph_dims)
label_dataset = tf.data.Dataset.from_tensor_slices(labels).batch(n_batch)
dataset = tf.data.Dataset.zip((graph_dataset, label_dataset))

"""
    Signatures
"""

graphs_sig = utils_tf.specs_from_graphs_tuple(
    next(iter(graph_dataset)), dynamic_num_graphs=True
)
labels_sig = tf.TensorSpec(shape=[None], dtype=tf.dtypes.float64)

"""
    Model
"""
tf.random.set_seed(0)
model = GraphNetworkSwitching(
    pos_class_weight=1 / labels.mean(),
    n_hidden=(128, 128, 128, 128),
    graphs_signature=graphs_sig,
    labels_signature=labels_sig,
    n_nodes=case.env.n_sub,
    n_edges=2 * case.env.n_line,
    **graph_dims,
)

model_dir = os.path.join(case_results_dir, "model-05")
checkpoint_path = os.path.join(model_dir, "ckpts")

ckpt = tf.train.Checkpoint(model=model, optimizer=model.optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=2)

if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    pprint(f"Restoring checkpoint from:", ckpt_manager.latest_checkpoint)

"""
    Training
"""

# Epoch Metrics
metric_recall_e = tf.keras.metrics.Recall()
metric_precision_e = tf.keras.metrics.Precision()
metric_accuracy_e = tf.keras.metrics.Accuracy()
metric_bce_e = tf.keras.metrics.BinaryCrossentropy(from_logits=False)
recall_e = []
precision_e = []
accuracy_e = []
bce_e = []

# Batch Metrics
metric_accuracy_b = tf.metrics.binary_accuracy
metric_bce_b = tf.metrics.binary_crossentropy
losses = []
accuracy = []

for epoch in range(n_epochs):
    start = time.time()

    # Reset epoch metrics
    metric_recall_e.reset_states()
    metric_precision_e.reset_states()
    metric_accuracy_e.reset_states()
    metric_bce_e.reset_states()

    for batch, (graph_batch, label_batch) in enumerate(dataset):
        (
            output_graphs,
            loss,
            probabilities,
            predicted_labels,
            gradients,
        ) = model.train_step(graph_batch, label_batch)

        # Batch Metric
        bce = metric_bce_b(label_batch, probabilities)  # Control
        acc = metric_accuracy_b(
            label_batch, tf.cast(predicted_labels, dtype=tf.float64)
        )

        # Epoch Metrics
        metric_recall_e(label_batch, predicted_labels)
        metric_precision_e(label_batch, predicted_labels)
        metric_accuracy_e(label_batch, predicted_labels)
        metric_bce_e(label_batch, probabilities)

        losses.append(loss.numpy())
        accuracy.append(acc.numpy())

        if batch % 100 == 0:
            pprint(
                "        - Batch/Epoch:",
                f"{batch}/{epoch}",
                "loss = {:.4f}".format(loss.numpy()),
                "bce = {:.4f}".format(bce.numpy()),
                "acc = {} %".format(int(100 * acc.numpy())),
            )

    recall_e.append(metric_recall_e.result().numpy())
    precision_e.append(metric_precision_e.result().numpy())
    accuracy_e.append(metric_accuracy_e.result().numpy())
    bce_e.append(metric_bce_e.result().numpy())

    if (epoch + 1) % 10 == 0:
        ckpt_save_path = ckpt_manager.save()
        pprint(f"            - Saving checkpoint to:", ckpt_save_path)
        pprint(f"            - Time taken for epoch:", f"{time.time() - start} secs")

ckpt_save_path = ckpt_manager.save()
pprint(f"    - Saving checkpoint to:", ckpt_save_path)
