import os

import numpy as np
import tensorflow as tf
from graph_nets import utils_tf

from experience import ExperienceCollector
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters
from lib.gns import (
    obses_to_combined_graphs_dict_list,
    tf_graph_dataset,
    get_graph_feature_dimensions,
    stack_graphs,
    GraphNetworkSwitching,
)
from lib.run_utils import create_logger
from lib.visualizer import Visualizer, pprint

visualizer = Visualizer()

experience_data_dir = make_dir(os.path.join(Const.EXPERIENCE_DIR, "data"))
results_dir = make_dir(os.path.join(Const.RESULTS_DIR, "binary-classification"))

agent_name = "agent-mip"
# case_name = "rte_case5_example"
case_name = "l2rpn_2019"

env_dc = True
verbose = False

case_experience_data_dir = make_dir(
    os.path.join(experience_data_dir, f"{case_name}-{env_pf(env_dc)}")
)
case_results_dir = make_dir(os.path.join(results_dir, f"{case_name}-{env_pf(env_dc)}"))
create_logger(logger_name=f"{case_name}-{env_pf(env_dc)}", save_dir=case_results_dir)

parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
case = load_case(case_name, env_parameters=parameters)
env = case.env

"""
    Load dataset
"""

collector = ExperienceCollector(save_dir=case_experience_data_dir)
collector.load_data(agent_name=agent_name, env=env)

observations, actions, rewards, dones = collector.aggregate_data()
labels = np.array(
    [action != env.action_space({}) for action in actions], dtype=np.float
)

combined_graphs_dict_list = obses_to_combined_graphs_dict_list(
    observations, dones, case
)

"""
    Construct TF dataset
"""

label_dataset = tf.data.Dataset.from_tensor_slices(labels)
graph_dataset = tf_graph_dataset(combined_graphs_dict_list)

sample_graph = next(iter(graph_dataset))
graphs_signature = utils_tf.specs_from_graphs_tuple(
    sample_graph, dynamic_num_graphs=True
)
labels_signature = tf.TensorSpec(shape=[None], dtype=tf.dtypes.float64)

model = GraphNetworkSwitching(
    pos_class_weight=1 / labels.mean(),
    n_hidden=(32, 32, 32, 32),
    graphs_signature=graphs_signature,
    labels_signature=labels_signature,
    **get_graph_feature_dimensions(sample_graph),
)

"""
    Training
"""
n_epochs = 20
n_batch = 32

tf.random.set_seed(0)
dataset = tf.data.Dataset.zip((graph_dataset, label_dataset))

dataset = dataset.shuffle(len(labels))
dataset = dataset.repeat(n_epochs)
dataset = dataset.batch(n_batch)

# Epoch
recall_fn = tf.keras.metrics.Recall()
false_net_fn = tf.keras.metrics.FalseNegatives()
true_pos_fn = tf.keras.metrics.TruePositives()

# Batch
accuracy_fn = tf.metrics.binary_accuracy
bce_fn = tf.metrics.binary_crossentropy

losses = []
cross_entropies = []
accuracies = []
recalls = []
false_negs = []
true_poss = []

for batch_idx, (graph_batch, label_batch) in enumerate(dataset):
    graph_batch = stack_graphs(graph_batch)
    output_graphs, loss, probabilities, predicted_labels, gradients = model.train_step(
        graph_batch, label_batch
    )

    bce = bce_fn(label_batch, probabilities)
    acc = accuracy_fn(label_batch, tf.cast(predicted_labels, dtype=tf.float64))
    rec = recall_fn(label_batch, predicted_labels)
    fns = false_net_fn(label_batch, predicted_labels)
    true_pos = true_pos_fn(label_batch, predicted_labels)

    losses.append(loss)
    cross_entropies.append(bce)
    accuracies.append(acc)
    recalls.append(rec)
    false_negs.append(fns)
    true_poss.append(true_pos)

    if batch_idx % 200 == 0:
        pprint(
            "Batch:",
            batch_idx,
            "loss = {:.4f}".format(loss.numpy()),
            "bce = {:.4f}".format(bce.numpy()),
            "acc = {:.4f}".format(acc.numpy()),
        )
        if acc.numpy() < 0.1 or (0.9 < acc.numpy() < 1.0):
            pprint("Labels:", label_batch.numpy().astype(int))
            pprint("Predictions:", predicted_labels.numpy().astype(int))
