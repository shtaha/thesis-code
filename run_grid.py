import grid2op
import pandapower as pp
import pandas as pd

# env_name = "rte_case5_example"
# env_name = "rte_case14_realistic"
env_name = "l2rpn_2019"
# env_name = "l2rpn_wcci_2020"

env = grid2op.make(dataset=env_name)
obs = env.reset()
# print(env.n_gen)
# print(env.gen_pmax)
# print(env.gen_pmin)
# print(env.backend.generators_info())
# print(env.backend.loads_info())
# print(env.get_thermal_limit())

print(env.name)

grid = env.backend._grid
backend = env.backend
print(env.backend._grid)

grid.load["controllable"] = False

grid.gen["controllable"] = True
grid.gen["min_p_mw"] = env.gen_pmin
grid.gen["max_p_mw"] = env.gen_pmax

# df["type"] = env.gen_type
# df["gen_redispatchable"] = env.gen_redispatchable
# df["gen_max_ramp_up"] = env.gen_max_ramp_up
# df["gen_max_ramp_down"] = env.gen_max_ramp_down
# df["gen_min_uptime"] = env.gen_min_uptime
# df["gen_min_downtime"] = env.gen_min_downtime
# df["gen_cost_per_mw"] = 0
# grid.gen = df

print(grid.bus.to_string())
print(grid.load.to_string())
print(grid.gen.to_string())
print(grid.line.to_string())

pp.runopp(grid, verbose=False)
print(grid.res_cost)
print(grid.res_gen)
print(grid.res_load)

print("\n\n\n\n")
pp.rundcopp(grid, verbose=False)
print(grid.res_cost)
print(grid.res_gen)
print(grid.res_load)