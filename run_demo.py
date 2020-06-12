import grid2op
from grid2op.Agent import RandomAgent, DoNothingAgent
from grid2op.Episode import EpisodeReplay
from grid2op.Runner import Runner
from tqdm import tqdm

from lib.constants import Constants as Const
from lib.data_utils import create_results_dir
from lib.visualizer import (
    render_and_save, print_dict, describe_environment
)

save_dir = create_results_dir(Const.RESULTS_DIR)

# env_name = "l2rpn_2019"
env_name = "rte_case14_realistic"
# env_name = "rte_case5_example"

# n_episodes = 2
#
#
# env = grid2op.make(dataset=env_name)
#
# runner = Runner(**env.get_params_for_runner(), agentClass=DoNothingAgent)
# result = runner.run(nb_episode=n_episodes, path_save=save_dir, pbar=tqdm)
#
# plot_epi = EpisodeReplay(save_dir)
# for e in n_episodes:
#     plot_epi.replay_episode(result[e][1], fps=4, gif_name=f"episode-{e}")

for env_name in ["rte_case5_example", "rte_case14_realistic", "l2rpn_2019", "l2rpn_wcci_2020"]:
    env = grid2op.make(dataset=env_name)
    describe_environment(env)

    obs = env.reset()
    obs_vect = obs.to_vect()

    action = env.action_space({})
    action_vect = action.to_vect()

    print("obs_vect", type(obs_vect), obs_vect.shape)
    print(obs_vect)
    print("action_vect", type(action_vect), action_vect.shape)
    print(action_vect)

    render_and_save(env, save_dir, env_name)
