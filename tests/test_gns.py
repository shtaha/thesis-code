import os
import unittest

import numpy as np
import tensorflow as tf
from graph_nets import utils_tf

from experience import ExperienceCollector
from lib.data_utils import make_dir, env_pf
from lib.dc_opf import load_case, CaseParameters, GridDCOPF
from lib.gns import (
    obses_to_graphs_dict_list,
    dict_list_to_combined_dict,
    tf_graph_dataset,
)
from lib.gns import stack_graphs, equal_graphs
from lib.visualizer import pprint, print_matrix


class TestGNs(unittest.TestCase):
    def test_gns_dataset(self):
        save_dir = os.path.join("../experience", "data")

        case_name = "rte_case5_example"
        agent_name = "agent-mip"

        env_dc = True
        case_save_dir = make_dir(
            os.path.join(save_dir, f"{case_name}-{env_pf(env_dc)}")
        )

        parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
        case = load_case(case_name, env_parameters=parameters)
        env = case.env

        collector = ExperienceCollector(save_dir=case_save_dir)
        collector.load_data(agent_name=agent_name, env=env)
        observations, actions, rewards, dones = collector.aggregate_data()

        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        labels = np.array(
            list(map(lambda action: int(action != env.action_space({})), actions)),
            dtype=np.int,
        )

        batch_size = 16
        max_length = 10 * batch_size + 1

        graphs_dict_list = obses_to_graphs_dict_list(
            observations, dones, grid, max_length=max_length
        )
        combined_graphs_dict_list = dict_list_to_combined_dict(graphs_dict_list)

        graph_dataset = tf_graph_dataset(combined_graphs_dict_list)
        label_dataset = tf.data.Dataset.from_tensor_slices(labels[:max_length])
        dataset = tf.data.Dataset.zip((graph_dataset, label_dataset))

        dataset = dataset.repeat(1)
        dataset = dataset.batch(batch_size)

        for batch_idx, (graph_batch, label_batch) in enumerate(dataset):
            graph_batch = stack_graphs(graph_batch)
            graph_batch_from_list = utils_tf.data_dicts_to_graphs_tuple(
                graphs_dict_list[
                    (batch_size * batch_idx) : (batch_size * (batch_idx + 1))
                ]
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
