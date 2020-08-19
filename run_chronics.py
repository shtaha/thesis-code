import os

import grid2op
import matplotlib.pyplot as plt
import numpy as np
from grid2op.Environment import Environment

from lib.constants import Constants as Const
from lib.data_utils import read_bz2_to_dataframe
from lib.dc_opf import CaseParameters
from lib.visualizer import Visualizer
from lib.dc_opf import Forecasts

visualizer = Visualizer()

# case_name = "rte_case5_example"
case_name = "l2rpn_2019"
# case_name = "l2rpn_wcci_2020"

parameters = CaseParameters(case_name=case_name, env_dc=False)

env: Environment = grid2op.make_from_dataset_path(
    dataset_path=os.path.join(os.path.expanduser("~"), "data_grid2op", case_name),
    backend=grid2op.Backend.PandaPowerBackend(),
    action_class=grid2op.Action.TopologyAction,
    observation_class=grid2op.Observation.CompleteObservation,
    reward_class=grid2op.Reward.L2RPNReward,
)

print(env.chronics_handler.path)
print(env.chronics_handler.get_id())
print(env.chronics_handler.time_interval)

colors = Const.COLORS
for e in range(20):
    env.reset()
    print("\ne", e, env.chronics_handler.get_name())

    if env.chronics_handler.get_name() != "0018":
        continue

    forecasts = Forecasts(
        env=env, t=env.chronics_handler.real_data.data.current_index, horizon=1,
    )

    load_p_chronics = env.chronics_handler.real_data.data.load_p
    prod_p_chronics = env.chronics_handler.real_data.data.prod_p
    t_chronics = np.arange(load_p_chronics.shape[0])

    chronics_dir = env.chronics_handler.get_id()
    load_p_file = [
        f
        for f in os.listdir(chronics_dir)
        if ("load_p" in f or "loads_p" in f)
        and "planned" not in f
        and "forecasted" not in f
    ]
    prod_p_file = [
        f
        for f in os.listdir(chronics_dir)
        if ("prod_p" in f or "prods_p" in f)
        and "planned" not in f
        and "forecasted" not in f
    ]

    print(load_p_file, prod_p_file)
    assert len(load_p_file) == 1
    assert len(prod_p_file) == 1

    load_p_from_file = read_bz2_to_dataframe(
        os.path.join(chronics_dir, load_p_file[0]), sep=";"
    )
    prod_p_from_file = read_bz2_to_dataframe(
        os.path.join(chronics_dir, prod_p_file[0]), sep=";"
    )

    if env.names_chronics_to_backend:
        if "loads" in env.names_chronics_to_backend:
            load_p_from_file.rename(
                columns=env.names_chronics_to_backend["loads"], inplace=True,
            )
            load_p_from_file = load_p_from_file.reindex(
                sorted(load_p_from_file.columns, key=lambda x: int(x.split("_")[-1])),
                axis=1,
            )
        if "prods" in env.names_chronics_to_backend:
            prod_p_from_file.rename(
                columns=env.names_chronics_to_backend["prods"], inplace=True,
            )

            prod_p_from_file = prod_p_from_file.reindex(
                sorted(prod_p_from_file.columns, key=lambda x: int(x.split("_")[-1])),
                axis=1,
            )

    load_p_from_file = load_p_from_file.values
    prod_p_from_file = prod_p_from_file.values
    t_from_file = np.arange(load_p_from_file.shape[0])

    load_p_step = np.empty_like(load_p_chronics)
    prod_p_step = np.empty_like(prod_p_chronics)
    t_step = np.empty_like(t_chronics)

    t = 0
    done = False
    while not done:
        # t = t + 1
        t = env.chronics_handler.real_data.data.current_index

        obs, reward, done, _ = env.step(env.action_space({}))
        load_p_step[t, :] = obs.load_p
        prod_p_step[t, :] = obs.prod_p

        forecasts.t = forecasts.t + 1

        # if t > 500:
        #     break

    load_p_step = load_p_step[:t, :]
    prod_p_step = prod_p_step[:t, :]
    t_step = np.arange(t)

    fig, ax = plt.subplots()
    fig.suptitle(
        f"Chronic {env.chronics_handler.get_name().replace('_', ' ')} - Load Active Power Demand"
    )
    for load_id in range(env.n_load):
        color = colors[load_id % len(colors)]
        ax.plot(
            t_step,
            load_p_step[:, load_id],
            label=f"Load-{load_id} S",
            c=color,
            linestyle="-",
            linewidth=1,
        )
        ax.plot(
            t_chronics,
            forecasts.data.load_p[:, load_id],
            # load_p_chronics[:, load_id],
            label=f"Load-{load_id} C",
            c=color,
            linestyle="-.",
            linewidth=1,
        )

    ax.set_xlim([0, t])
    ax.set_xlabel("Time step t")
    ax.set_ylabel("P [p.u.]")
    if env.n_load < 4:
        ax.legend()
    fig.show()

    fig, ax = plt.subplots()
    fig.suptitle(
        f"Chronic {env.chronics_handler.get_name().replace('_', ' ')} - Generator Active Power Production"
    )
    for gen_id in range(env.n_gen):
        color = colors[gen_id % len(colors)]
        ax.plot(
            t_step,
            prod_p_step[:, gen_id],
            label=f"Gen-{gen_id} S",
            c=color,
            linestyle="-",
            linewidth=1,
        )
        ax.plot(
            t_chronics,
            forecasts.data.prod_p[:, gen_id],
            # prod_p_chronics[:, gen_id],
            label=f"Gen-{gen_id} C",
            c=color,
            linestyle="-.",
            linewidth=1,
        )

    ax.set_xlim([0, t])
    ax.set_xlabel("Time step t")
    ax.set_ylabel("P [p.u.]")
    if env.n_gen < 4:
        ax.legend()
    fig.show()

    print(env.chronics_handler.real_data.data.max_iter, env.chronics_handler.get_id())
    print(load_p_from_file.shape, prod_p_from_file.shape)
    print(t_chronics.shape, load_p_chronics.shape, prod_p_chronics.shape)
    print(t_step.shape, load_p_step.shape, prod_p_step.shape)

    assert (
        np.linalg.norm(load_p_from_file - load_p_chronics)
        / np.linalg.norm(load_p_from_file)
        < 1e-6
    )
    assert (
        np.linalg.norm(prod_p_from_file - prod_p_chronics)
        / np.linalg.norm(prod_p_from_file)
        < 1e-6
    )
    assert (
        np.linalg.norm(load_p_step - load_p_chronics[:t, :])
        / np.linalg.norm(load_p_from_file)
        < 0.2
    )
    assert (
        np.linalg.norm(prod_p_step - prod_p_chronics[:t, :])
        / np.linalg.norm(prod_p_from_file)
        < 0.2
    )
