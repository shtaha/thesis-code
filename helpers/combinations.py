import itertools
import numpy as np

from lib.data_utils import indices_to_hot

n = 5
m = 5
line_ids = np.arange(0, m)
combinations = list()
line_statuses = list()

for i in range(n + 1):
    combinations.extend([list(x) for x in itertools.combinations(line_ids, i) if list(x)])

for line_status in combinations:
    line_status = indices_to_hot(line_status, length=m, dtype=np.int)
    print(" ".join(line_status.astype(str)))
