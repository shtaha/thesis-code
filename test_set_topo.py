import numpy as np
from lib.dc_opf import load_case, TopologyConverter
from lib.visualizer import pprint

case_name = "l2rpn_2019"
case = load_case(case_name)
env = case.env
tc = TopologyConverter(env)

obs = env.reset()
topo_vect = np.ones((env.dim_topo,), dtype=np.int)

pprint("    - topo:", topo_vect)
pprint("    - env:", env.current_obs.topo_vect)

sub_id = 4
sub_topo = np.array([1, 2, 2, 1, 1])
sub_mask = tc.substation_topology_mask[sub_id, :]
topo_vect[sub_mask] = sub_topo

env.current_obs.topo_vect = topo_vect

pprint("    - mask:", sub_mask.astype(np.int))
pprint("    - topo:", topo_vect)
pprint("    - env:", env.current_obs.topo_vect)

fig = env.render()

action = env.action_space({})
env.step(action)

fig = env.render()
