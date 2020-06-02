import grid2op

from lib.dc_opf import DCOptimalPowerFlow, update_backend
from lib.visualizer import render_and_save

env_name = "rte_case5_example"

# for env_name in ["rte_case5_example", "l2rpn_2019", "rte_case14_realistic", "l2rpn_wcci_2020"]:
# for env_name in ["rte_case5_example", "l2rpn_2019"]:
for env_name in ["rte_case5_example"]:
    env = grid2op.make(dataset=env_name)
    update_backend(env, verbose=True)

    opf = DCOptimalPowerFlow(env)

    obs = env.reset()
    # render_and_save(env)
