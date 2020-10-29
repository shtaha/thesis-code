import numpy as np

from lib.data_utils import extract_target_windows

np.random.seed(1)
n = 10
k = 2
p = 0.3
actions = np.random.binomial(k, p, n)
obses = np.arange(n)

print("\t".join(actions.astype(str)))
print("\t".join(obses.astype(str)))

n_window = 2
window = extract_target_windows(actions, n_window=n_window)
print(obses[window])
print("\t".join(window.astype(int).astype(str)))
