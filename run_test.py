import os

import graph_nets as gns
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from bclassification.utils_base import (
    print_class_weights,
    compute_weight_bias,
    print_dataset,
)
from bclassification.utils_gnsn import create_dataset
from experience import load_experience
from lib.constants import Constants as Const
from lib.data_utils import (
    make_dir,
    env_pf,
    create_results_dir,
)
from lib.gns import get_graph_feature_dimensions
from lib.visualizer import Visualizer, pprint

Visualizer()

# experience_dir = make_dir(os.path.join(Const.EXPERIENCE_DIR, "data-aug"))
experience_dir = make_dir(os.path.join(Const.RESULTS_DIR, "performance-aug"))
results_dir = make_dir(os.path.join(Const.RESULTS_DIR, "_bc-gnsn"))

agent_name = "agent-mip"
case_name = "l2rpn_2019_art"
env_dc = True
verbose = False

case_results_dir = make_dir(os.path.join(results_dir, f"{case_name}-{env_pf(env_dc)}"))
case, collector = load_experience(case_name, agent_name, experience_dir, env_dc=env_dc)

"""
    Parameters
"""
random_seed = 0

mode = "structured"

n_window_targets = 0
n_window_history = 1
downsampling_rate = 0.1
feature_scaling = True
val_frac = 0.10

# Model
model_type = "res"  # "fc" or "res"
dropout_rate = 0.0
l2_reg = 0.0
n_hidden = 256
n_hidden_layers = 4
threshold = 0.50

# Training
learning_rate = 1e-5
n_batch = 512
n_epochs = 1000

"""
    Dataset
"""

np.random.seed(random_seed)
tf.random.set_seed(random_seed)

X, Y, mask_targets, X_all, Y_all = create_dataset(
    case,
    collector,
    mode=mode,
    n_window_history=n_window_history,
    n_window_targets=n_window_targets,
    downsampling_rate=downsampling_rate,
    feature_scaling=feature_scaling,
)

class_weight, initial_bias = compute_weight_bias(Y)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y, test_size=val_frac, random_state=random_seed
)

mask_test_neg = np.logical_and(
    ~mask_targets, np.random.binomial(1, val_frac, mask_targets.size).astype(np.bool)
)
X_test = np.concatenate((X_val, X_all[mask_test_neg, :]))
Y_test = np.concatenate((Y_val, Y_all[mask_test_neg]))

print_dataset(X_all, Y_all, "All data")
print_dataset(X, Y, "Data")
print_dataset(X_train, Y_train, "Train")
print_dataset(X_val, Y_val, "Validation")
print_dataset(X_test, Y_test, "Test")
print_class_weights(class_weight)
pprint("Initial bias:", "{:.4f}".format(float(initial_bias)))

model_dir = create_results_dir(case_results_dir, model_name=model_type)

graph_dims = get_graph_feature_dimensions(lgraphs=lgraphs.tolist())
lgraph_dims = {**graph_dims, "n_nodes": case.env.n_sub, "n_edges": 2 * case.env.n_line}

"""
    Signatures
"""

graphs_sig = gns.utils_tf.specs_from_graphs_tuple(
    dgraphs_to_graphs(next(iter(X_train))), dynamic_num_graphs=True
)
labels_sig = tf.TensorSpec(shape=[None], dtype=tf.dtypes.float64)
