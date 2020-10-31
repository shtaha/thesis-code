import argparse

import grid2op

from lib.constants import Constants as Const
from lib.data_utils import create_results_dir, get_grid_info
from lib.dc_opf import get_dc_opf_environment_parameters, get_dc_opf_observation_parameters, get_topology_info, dc_opf
from lib.visualizer import describe_environment


def parse_arguments():
    parser = argparse.ArgumentParser()
    # parser.add_argument("--env_name", default=Const.ENV_NAME, type=str, help="Environment name.")
    parser.add_argument("--env_name", default="rte_case5_example", type=str, help="Environment name.")
    # parser.add_argument("--env_name", default="l2rpn_2019", type=str, help="Environment name.")

    parser.add_argument(
        "--n_bus",
        default=1,
        type=int,
        help="Number of buses per substation. Tested only for 2.",
    )
    parser.add_argument(
        "-v", "--verbose", help="Set verbosity level.", action="store_false"
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    # save_dir = create_results_dir(Const.RESULTS_DIR)

    env_name = args.env_name
    env = grid2op.make(dataset=env_name)
    action_space = env.action_space
    describe_environment(env)

    env_info = get_grid_info(env_name, verbose=True)

    obs = env.reset()

    # print(env.gen_pmax)
    # print(env.gen_pmin)

    # Get parameters
    obs_params = get_dc_opf_observation_parameters(obs, verbose=args.verbose)
    env_params = get_dc_opf_environment_parameters(env_info, verbose=args.verbose)
    topology_info = get_topology_info(env, obs, n_bus=args.n_bus, verbose=args.verbose)
    dc_opf_params = {**env_params, **obs_params, **topology_info}

    dc_opf(dc_opf_params)
