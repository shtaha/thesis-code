import grid2op

from lib.constants import Constants as Const
from lib.data_utils import create_results_dir
from lib.visualizer import (
    render_and_save
)

env_name = "l2rpn_wcci_2020"
save_dir = create_results_dir(Const.RESULTS_DIR)

env = grid2op.make(dataset=env_name)
action_space = env.action_space

obs = env.reset()
render_and_save(env, save_dir, env_name)

print(obs.topo_vect)
