from collections import deque

import numpy as np

from lib.data_utils import is_nonetype, extract_target_windows, moving_window, extract_history_windows
from lib.visualizer import pprint, format_matrix

obses = np.arange(0, 25).tolist()
n_window = 3

first_method = False
if first_method:
    for i in range(len(obses)):
        start_idx = np.maximum(0, i - n_window + 1)
        end_idx = i + 1

        obses_padding = [None] * (n_window - (end_idx - start_idx))

        obses_window = obses_padding + obses[start_idx:end_idx]
        print("{:>6}".format(f"{start_idx}-{end_idx}"),
              len(obses_window),
              ", ".join(["{:>4}".format(obs if not is_nonetype(obs) else -1) for obs in obses_window]))

second_method = False
if second_method:
    padding = [None] * n_window
    queue = deque(padding)

    for i in range(7):
        queue.popleft()
        queue.append(i)
        print([item for item in queue])
        print(queue[-1])

n = 25
np.random.seed(10)
obses = np.arange(0, n)
targets = np.random.binomial(1, 0.1, n).astype(np.bool)

n_window_targets = 3
n_window_history = 1
downsampling_rate = 1.0

mask_positives = extract_target_windows(targets, n_window=n_window_targets)
mask_negatives = np.logical_and(np.random.binomial(1, downsampling_rate, len(targets)).astype(np.bool), ~mask_positives)
mask_targets = np.logical_or(targets, mask_negatives)

mask_history = extract_history_windows(mask_targets, n_window=n_window_history)

history = moving_window(obses, mask_targets,
                        n_window=n_window_history,
                        process_fn=lambda x: np.square(x),
                        combine_fn=lambda x: x,
                        padding=0
                        )

pprint("obses", format_matrix(obses)[0])
pprint("targets", format_matrix(targets)[0])

pprint("mask_positives", format_matrix(mask_positives.astype(int))[0])
pprint("mask_negative", format_matrix(mask_negatives.astype(int))[0])
pprint("mask_targets", format_matrix(mask_targets.astype(int))[0])
pprint("mask_history", format_matrix(mask_history.astype(int))[0])

# print("\n")
# for i, h in enumerate(history):
#     pprint(i, format_matrix(h)[0])
