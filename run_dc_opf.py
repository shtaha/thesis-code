import grid2op

from lib.data_utils import update_backend
from lib.dc_opf import DCOptimalPowerFlow

env_name = "rte_case5_example"

# for env_name in ["rte_case5_example", "l2rpn_2019", "rte_case14_realistic", "l2rpn_wcci_2020"]:
# for env_name in ["rte_case5_example", "l2rpn_2019"]:
for env_name in ["rte_case5_example"]:
    # for env_name in ["l2rpn_2019"]:
    env = grid2op.make(dataset=env_name)
    update_backend(env, verbose=True)

    # opf = DCOptimalPowerFlow(env, verbose=False)
    #
    # # gen_costs = 1 + np.arange(1, env.n_gen + 1) * 0.1
    # gen_costs = np.random.uniform(low=1.0, high=2.0, size=(env.n_gen, ))
    # opf.set_gen_cost(gen_costs)
    #
    # opf.solve_dc_opf()
    # opf.solve_dc_opf_backend()

    obs = env.reset()
    print("obs", obs.load_p)
    action = env.action_space({"set_bus": {
        "lines_ex_id": [(3, 2)]
    }})

    print(env.backend._grid.load.to_string())
    print(env.backend._grid.line.to_string())
    obs_next, reward, done, info = env.step(action)
    print(env.backend._grid.load.to_string())
    print(env.backend._grid.line.to_string())
    opf = DCOptimalPowerFlow(env, verbose=False)
    env.render()

    for i in range(10):
        obs_next, reward, done, info = env.step(action)
        print(i)
        print(obs_next.load_p, obs_next.load_p.sum(), obs_next.load_q)
        print(obs_next.prod_p, obs_next.prod_p.sum(), obs_next.prod_q)