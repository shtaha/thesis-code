import os
import sys
import unittest

import tensorflow as tf
from graph_nets import utils_tf

from experience import load_experience
from lib.action_space import is_do_nothing_action
from lib.gns import equal_graphs
from lib.gns import (
    obses_to_lgraphs,
    lgraphs_to_cgraphs,
    get_graph_feature_dimensions,
    tf_batched_graph_dataset,
)
from lib.visualizer import pprint, print_matrix


class TestGNs(unittest.TestCase):
    def test_gns_dataset(self):
        if sys.platform != "win32":
            self.assertTrue(True)
            return

        experience_dir = os.path.join("../experience", "data")

        case_name = "rte_case5_example"
        agent_name = "agent-mip"

        env_dc = True

        case, collector = load_experience(
            case_name, agent_name, experience_dir, env_dc=env_dc
        )
        obses, actions, rewards, dones = collector.aggregate_data()

        n_batch = 16
        max_length = 10 * n_batch + 1
        n_window = 2

        graphs_dict_list = obses_to_lgraphs(
            obses, dones, case, max_length=max_length, n_window=n_window
        )
        cgraphs = lgraphs_to_cgraphs(graphs_dict_list)
        labels = is_do_nothing_action(actions, case.env)

        graph_dims = get_graph_feature_dimensions(cgraphs=cgraphs)
        graph_dataset = tf_batched_graph_dataset(cgraphs, n_batch=n_batch, **graph_dims)
        label_dataset = tf.data.Dataset.from_tensor_slices(labels).batch(n_batch)
        dataset = tf.data.Dataset.zip((graph_dataset, label_dataset))
        dataset = dataset.repeat(1)

        for batch_idx, (graph_batch, label_batch) in enumerate(dataset):
            graph_batch_from_list = utils_tf.data_dicts_to_graphs_tuple(
                graphs_dict_list[(n_batch * batch_idx) : (n_batch * (batch_idx + 1))]
            )

            check = tf.squeeze(equal_graphs(graph_batch, graph_batch_from_list)).numpy()

            pprint("Batch:", batch_idx, check)

            if not check:
                for field in [
                    "globals",
                    "nodes",
                    "edges",
                ]:
                    print_matrix(getattr(graph_batch, field))
                    print_matrix(getattr(graph_batch_from_list, field))

            self.assertTrue(check)
