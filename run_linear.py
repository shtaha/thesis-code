import os

import numpy as np

from experience import load_experience
from lib.action_space import is_do_nothing_action
from lib.constants import Constants as Const
from lib.data_utils import make_dir, env_pf, extract_target_windows
from lib.run_utils import create_logger
from lib.visualizer import Visualizer, pprint

Visualizer()

experience_dir = make_dir(os.path.join(Const.EXPERIENCE_DIR, "data-aug"))
results_dir = make_dir(os.path.join(Const.RESULTS_DIR, "binary-linear"))

agent_name = "agent-mip"
case_name = "l2rpn_2019"
env_dc = True
verbose = False

case_results_dir = make_dir(os.path.join(results_dir, f"{case_name}-{env_pf(env_dc)}"))
create_logger(logger_name=f"{case_name}-{env_pf(env_dc)}", save_dir=case_results_dir)

case, collector = load_experience(case_name, agent_name, experience_dir, env_dc=env_dc)
obses, actions, rewards, dones = collector.aggregate_data()

Y = is_do_nothing_action(actions, case.env)


X = np.vstack(
    [
        np.concatenate((obs.rho, obs.prod_p, obs.load_p, obs.p_or, obs.p_ex))
        for obs in obses
    ]
)

n_window = 1
mask = extract_target_windows(Y, mask=~dones, n_window=n_window)
X = X[mask, :]
Y = Y[mask]

pprint("    - Data:", f"X {X.shape}", f"Y {Y.shape}")
pprint("    - Labels:", "{:.2f} %".format(100 * Y.mean()))

pprint("    - Data:", f"X {X.shape}", f"Y {Y.shape}")
pprint("    - Labels:", "{:.2f} %".format(100 * Y.mean()))
