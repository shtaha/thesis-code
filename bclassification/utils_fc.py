import numpy as np

from lib.action_space import is_do_nothing_action
from lib.data_utils import (
    moving_window,
    extract_history_windows,
    indices_to_hot,
    backshift_and_hstack,
)
from lib.dc_opf import TopologyConverter
from lib.visualizer import pprint


def obs_to_vects(obs, tc):
    lines_or_to_sub_bus = tc.lines_or_to_sub_bus(obs)
    lines_ex_to_sub_bus = tc.lines_ex_to_sub_bus(obs)
    gens_to_sub_bus = tc.gens_to_sub_bus(obs)
    loads_to_sub_bus = tc.loads_to_sub_bus(obs)

    prod_1 = np.multiply(tc.sub_bus_mask(gens_to_sub_bus, 1, np.float), obs.prod_p)
    prod_2 = np.multiply(tc.sub_bus_mask(gens_to_sub_bus, 2, np.float), obs.prod_p)
    prod_vect = np.concatenate((prod_1, prod_2))

    load_1 = np.multiply(tc.sub_bus_mask(loads_to_sub_bus, 1, np.float), obs.load_p)
    load_2 = np.multiply(tc.sub_bus_mask(loads_to_sub_bus, 2, np.float), obs.load_p)
    load_vect = np.concatenate((load_1, load_2))

    sub_vect = obs.time_before_cooldown_sub
    inj_vect = np.concatenate((prod_vect, load_vect, sub_vect))
    inj_vect = np.nan_to_num(inj_vect)

    p_ors = []
    for sub_bus_or in [1, 2]:
        for sub_bus_ex in [1, 2]:
            mask = np.multiply(
                tc.sub_bus_mask(lines_or_to_sub_bus, sub_bus_or, np.float),
                tc.sub_bus_mask(lines_ex_to_sub_bus, sub_bus_ex, np.float),
            )
            p_or = np.multiply(obs.p_or, mask)
            p_ors.append(p_or)
    p_ors.append(obs.rho)

    # Add status, cooldown, maintenance, overflow
    p_ors.append(obs.line_status)
    p_ors.append(obs.timestep_overflow)
    p_ors.append(obs.time_next_maintenance)
    p_ors.append(obs.duration_next_maintenance)
    p_ors.append(obs.time_before_cooldown_line)

    line_vect = np.concatenate(p_ors)
    line_vect = np.nan_to_num(line_vect)

    return line_vect.astype(np.float), inj_vect.astype(np.float)


def obs_to_vect(obs, tc):
    return np.concatenate(obs_to_vects(obs, tc))


def obs_to_vects_with_tc(tc):
    return lambda obs: obs_to_vects(obs, tc)


def obs_to_vect_with_tc(tc):
    return lambda obs: obs_to_vect(obs, tc)


def obs_vects_to_vect(obs_vects):
    return np.concatenate(obs_vects)


def action_to_vect(action):
    return indices_to_hot([int(action)], length=2, dtype=np.float)


def action_vects_to_vect(action_vects):
    return np.concatenate(action_vects)


def create_datasets(
    case,
    collector,
    n_window_targets=0,
    n_window_history=0,
    n_window_forecasts=0,
    use_actions=True,
    use_forecasts=True,
    feature_scaling=True,
    downsampling_rate=1.0,
):
    mask_targets = []
    Y_all = []
    X_all = []

    obs_to_vect = obs_to_vect_with_tc(TopologyConverter(case.env))

    for chronic_idx, chronic_data in collector.data.items():
        chronic_obses = chronic_data["obses"][:-1]
        chronic_labels = is_do_nothing_action(
            chronic_data["actions"], case.env, dtype=np.bool
        )
        chronic_len = len(chronic_obses)

        mask_positives = extract_history_windows(
            chronic_labels, n_window=n_window_targets
        )
        mask_negatives = np.logical_and(
            np.random.binomial(1, downsampling_rate, len(chronic_labels)).astype(
                np.bool
            ),
            ~mask_positives,
        )
        chronic_mask_targets = np.logical_or(chronic_labels, mask_negatives)

        chronic_X_obses = moving_window(
            chronic_obses,
            n_window=n_window_history,
            process_fn=obs_to_vect,
            combine_fn=obs_vects_to_vect,
            padding=np.zeros_like(obs_to_vect(chronic_obses[0])),
        )

        chronic_X = chronic_X_obses

        if use_actions:
            chronic_actions = np.roll(chronic_labels, 1).astype(np.float)
            chronic_actions[0] = 0.0
            chronic_X_actions = moving_window(
                chronic_actions,
                n_window=n_window_history,
                process_fn=action_to_vect,
                combine_fn=action_vects_to_vect,
                padding=np.zeros_like(action_to_vect(chronic_labels[0])),
            )
            chronic_X = np.hstack((chronic_X, chronic_X_actions))

        if use_forecasts:
            prods, loads = collector.load_forecasts(case.env, chronic_idx)
            chronic_X_forecasts = np.concatenate(
                (prods[1 : chronic_len + 1], loads[1 : chronic_len + 1]), axis=1
            )

            chronic_X_forecasts = backshift_and_hstack(
                chronic_X_forecasts, max_shift=n_window_forecasts
            )
            chronic_X = np.hstack((chronic_X, chronic_X_forecasts))

        mask_targets.extend(chronic_mask_targets)
        X_all.extend(chronic_X)
        Y_all.extend(chronic_labels)

    mask_targets = np.array(mask_targets)
    X_all = np.vstack(X_all)
    Y_all = np.array(Y_all)

    X_std = X_all.std(axis=0)
    mask_zero = np.equal(X_std, 0.0)
    X_all = X_all[:, ~mask_zero]

    pprint("    - Removed columns:", mask_zero.sum())

    if feature_scaling:
        X_all = X_all / X_std[~mask_zero]

    X = X_all[mask_targets, :]
    Y = Y_all[mask_targets]

    pprint(
        "    - Labels:",
        f"{Y_all.sum()}/{Y_all.size}",
        "{:.2f} %".format(100 * Y_all.mean()),
    )

    return X, Y, mask_targets, X_all, Y_all
