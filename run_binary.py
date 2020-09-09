import os

import numpy as np

from experience import ExperienceCollector
from lib.constants import Constants as Const
from lib.data_utils import make_dir
from lib.dc_opf import load_case, CaseParameters, GridDCOPF
from lib.gns import obses_to_combined_graphs_dict_list
from lib.gns import obses_to_graphs_dict_list
from lib.run_utils import create_logger
from lib.visualizer import Visualizer

visualizer = Visualizer()

save_dir = make_dir(os.path.join(Const.EXPERIENCE_DIR, "data"))

env_dc = True
verbose = False

kwargs = dict()

# for case_name in ["rte_case5_example", "l2rpn_2019", "l2rpn_wcci_2020"]:
for case_name in ["rte_case5_example"]:
    env_pf = "dc" if env_dc else "ac"
    case_save_dir = make_dir(os.path.join(save_dir, f"{case_name}-{env_pf}"))

    create_logger(logger_name=f"{case_name}-{env_pf}", save_dir=case_save_dir)

    """
        Initialize environment.
    """
    parameters = CaseParameters(case_name=case_name, env_dc=env_dc)
    case = load_case(case_name, env_parameters=parameters)
    env = case.env

    for agent_name in [
        "agent-mip",
        # "agent-multistep-mip",
    ]:
        """
            Initialize agent.
        """
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        collector = ExperienceCollector(save_dir=case_save_dir)
        collector.load_data(agent_name=agent_name, env=env)

        observations, actions, rewards, dones = collector.aggregate_data()
        targets = np.array(
            list(map(lambda action: int(action != env.action_space({})), actions)),
            dtype=np.int,
        )

        graphs_dict_list = obses_to_graphs_dict_list(observations, dones, grid)
        combined_graphs_dict_list = obses_to_combined_graphs_dict_list(
            observations, dones, grid
        )

        data = tf.data.Dataset.from_tensor_slices((combined_graphs_dict_list, targets))
        data = data.map(graph_dict_to_graph)
        data = data.repeat(n_epochs).batch(batch_size)

        for i, (x, y) in enumerate(data):
            x = stack_graphs(x)
