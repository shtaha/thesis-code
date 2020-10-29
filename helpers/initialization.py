import numpy as np

v = dict()
v[1, 1] = 9
v[2, 2] = 16
v[3, 3] = 25

print(v)


def create_map_dual_ids_to_values(ids_first, ids_second, values):
    mapping = dict()
    for j, idx_second in enumerate(ids_second):
        for i, idx_first in enumerate(ids_first):
            value = values[i, j]
            mapping[(idx_first, idx_second)] = value
    return mapping


data = np.random.randint(0, 5, size=(3, 2))

print(data)
print(data.shape)

map_ = create_map_dual_ids_to_values(
    [1, 2, 3],
    [4, 5],
    data,
)

for key in map_:
    print(key, map_[key])
