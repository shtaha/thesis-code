import json
import os

import grid2op
import numpy as np
import pandas as pd

from lib.visualizer import print_matrix, print_topology_hot_line, print_topology_line, get_topology_to_bus_ids


def get_grid_info(env_name, verbose=False):
    env_path = os.path.join(grid2op.MakeEnv.PathUtils.DEFAULT_PATH_DATA, env_name)

    with open(os.path.join(env_path, "grid.json"), mode="r") as f:
        grid_json = json.loads(f.read())["_object"]

    info = dict()

    # Parse json for DataFrame objects
    for item in grid_json:
        if isinstance(grid_json[item], dict):
            if "_class" in grid_json[item]:
                if grid_json[item]["_class"] == "DataFrame":
                    item_dict = json.loads(grid_json[item]["_object"])
                    item_df = pd.DataFrame(**item_dict)

                    if item_df.size > 0:
                        info[item] = item_df

                        if verbose:
                            print(item + "\n" + item_df.to_string() + "\n")
    return info


def get_topology_info(environment, observation, n_bus=2, verbose=False):
    sub_info = environment.sub_info
    sub_ids = np.arange(len(sub_info))

    bus_ids = np.arange(n_bus * len(sub_info))

    bus_to_sub_id = np.array([np.ones((n_bus,), dtype=np.int) * sub_id for sub_id in sub_ids]).flatten()
    bus_to_sub_pos = np.array([np.arange(1, n_bus + 1) for _ in sub_ids]).flatten()
    sub_to_bus_ids = np.array([bus_ids[np.equal(bus_to_sub_id, sub_id)] for sub_id in sub_ids])

    topology_vector = observation.topo_vect
    topology_dim = len(topology_vector)

    line_or_topo_pos = observation.line_or_pos_topo_vect
    line_ex_topo_pos = observation.line_ex_pos_topo_vect
    line_topo_pos = np.concatenate((line_or_topo_pos, line_ex_topo_pos), axis=0)
    gen_topo_pos = observation.gen_pos_topo_vect
    load_topo_pos = observation.load_pos_topo_vect

    line_or_topo_one_hot = hot_vector(line_or_topo_pos, length=topology_dim)
    line_ex_topo_one_hot = hot_vector(line_ex_topo_pos, length=topology_dim)
    line_topo_one_hot = hot_vector(line_topo_pos, length=topology_dim)
    gen_topo_one_hot = hot_vector(gen_topo_pos, length=topology_dim)
    load_topo_one_hot = hot_vector(load_topo_pos, length=topology_dim)

    line_or_to_sub_id = observation.line_or_to_subid
    line_ex_to_sub_id = observation.line_ex_to_subid
    line_to_sub_id = np.concatenate((line_or_to_sub_id, line_ex_to_sub_id), axis=0)
    gen_to_sub_id = observation.gen_to_subid
    load_to_sub_id = observation.load_to_subid

    # Sort arrays
    line_topo_sort_indices = np.argsort(line_topo_pos)
    gen_topo_sort_indices = np.argsort(gen_topo_pos)
    load_topo_sort_indices = np.argsort(load_topo_pos)

    # line_topo_pos_sorted = line_topo_pos[line_topo_sort_indices]
    # load_topo_pos_sorted = load_topo_pos[load_topo_sort_indices]
    # gen_topo_pos_sorted = gen_topo_pos[gen_topo_sort_indices]

    line_to_sub_id_sorted = line_to_sub_id[line_topo_sort_indices]
    gen_to_sub_id_sorted = gen_to_sub_id[gen_topo_sort_indices]
    load_to_sub_id_sorted = load_to_sub_id[load_topo_sort_indices]

    line_ids = np.arange(len(line_or_to_sub_id))
    gen_ids = np.arange(len(gen_to_sub_id))
    load_ids = np.arange(len(load_to_sub_id))

    topology_to_sub_id = -np.ones(shape=(topology_dim,), dtype=np.int)
    topology_to_sub_id[line_topo_one_hot] = line_to_sub_id_sorted
    topology_to_sub_id[gen_topo_one_hot] = gen_to_sub_id_sorted
    topology_to_sub_id[load_topo_one_hot] = load_to_sub_id_sorted

    # Grid elements do not change substations, but change buses within substation.
    # This varies by every action on grid topology.
    topology_to_bus_id = get_topology_to_bus_ids(topology_vector, topology_to_sub_id, sub_to_bus_ids)

    # Grid elements to bus ids
    line_or_to_bus_id = [topology_to_bus_id[line_or_topo_pos[line_id]] for line_id in line_ids]
    line_ex_to_bus_id = [topology_to_bus_id[line_ex_topo_pos[line_id]] for line_id in line_ids]
    gen_to_bus_id = [topology_to_bus_id[gen_topo_pos[gen_id]] for gen_id in gen_ids]
    load_to_bus_id = [topology_to_bus_id[load_topo_pos[load_id]] for load_id in load_ids]

    topology_info = dict()
    # Substation and bus ids
    topology_info["sub_ids"] = sub_ids
    topology_info["bus_ids"] = bus_ids
    topology_info["line_ids"] = line_ids
    topology_info["gen_ids"] = gen_ids
    topology_info["load_ids"] = load_ids

    # Map bus id to sub id
    topology_info["bus_to_sub_id"] = bus_to_sub_id

    # Map sub id to bus ids on that substation
    topology_info["sub_to_bus_ids"] = sub_to_bus_ids

    # Map bus id to substation position
    topology_info["bus_to_sub_pos"] = bus_to_sub_pos

    # Grid elements to substation topology vector positions
    topology_info["line_topo_pos"] = line_topo_pos
    topology_info["gen_topo_pos"] = gen_topo_pos
    topology_info["load_topo_pos"] = load_topo_pos

    topology_info["line_topo_one_hot"] = line_topo_one_hot
    topology_info["gen_topo_one_hot"] = gen_topo_one_hot
    topology_info["load_topo_one_hot"] = load_topo_one_hot

    # Grid elements to substation ids
    topology_info["line_or_to_sub_id"] = line_or_to_sub_id
    topology_info["line_ex_to_sub_id"] = line_ex_to_sub_id
    topology_info["line_to_sub_id"] = line_to_sub_id
    topology_info["gen_to_sub_id"] = gen_to_sub_id
    topology_info["load_to_sub_id"] = load_to_sub_id

    # Sorted arrays
    topology_info["line_to_sub_id_sorted"] = line_to_sub_id_sorted
    topology_info["gen_to_sub_id_sorted"] = gen_to_sub_id_sorted
    topology_info["load_to_sub_id_sorted"] = load_to_sub_id_sorted

    # Topology position to substation id
    topology_info["topology_to_sub_id"] = topology_to_sub_id

    # Topology position to bus id
    topology_info["topology_to_bus_id"] = topology_to_bus_id

    topology_info["line_or_to_bus_id"] = line_or_to_bus_id
    topology_info["line_ex_to_bus_id"] = line_ex_to_bus_id
    topology_info["gen_to_bus_id"] = gen_to_bus_id
    topology_info["load_to_bus_id"] = load_to_bus_id

    if verbose:
        print_topology_line(np.ones((topology_dim,), dtype=np.bool), list(range(topology_dim)), "topology pos")
        print_topology_hot_line(line_or_topo_one_hot, "line_or")
        print_topology_hot_line(line_ex_topo_one_hot, "line_ex")
        print_topology_hot_line(line_topo_one_hot, "line")
        print_topology_hot_line(gen_topo_one_hot, "gen")
        print_topology_hot_line(load_topo_one_hot, "load")
        print_topology_line(line_topo_one_hot, topology_to_sub_id, "line sub id")
        print_topology_line(gen_topo_one_hot, topology_to_sub_id, "gen sub id")
        print_topology_line(load_topo_one_hot, topology_to_sub_id, "load sub id")
        print_topology_line(np.ones((topology_dim,), dtype=np.bool), topology_to_sub_id, "topology sub ids")
        print_topology_line(line_topo_one_hot, topology_to_bus_id, "line bus id")
        print_topology_line(gen_topo_one_hot, topology_to_bus_id, "gen bus id")
        print_topology_line(load_topo_one_hot, topology_to_bus_id, "load bus id")
        print_topology_line(np.ones((topology_dim,), dtype=np.bool), topology_to_bus_id, "topology bus ids")

    return topology_info


def get_dc_opf_environment_parameters(environment_info, verbose=False):
    parameters = dict()

    # Generators
    gen_info = environment_info["gen"]
    n_gens = gen_info.shape[0]
    if "max_p_mw" in gen_info.columns:
        gen_active_power_max = gen_info["max_p_mw"].values * 1e6
    else:
        gen_active_power_max = np.full((n_gens,), fill_value=np.inf)

    if "min_p_mw" in gen_info.columns:
        gen_active_power_min = gen_info["min_p_mw"].values
    else:
        gen_active_power_min = np.full((n_gens,), fill_value=0)

    # Lines
    line_info = environment_info["line"]
    n_lines = line_info.shape[0]
    line_length = line_info["length_km"].values
    line_resistance = np.multiply(line_info["r_ohm_per_km"].values, line_length)  # Resistance in Ohms
    line_reactance = np.multiply(line_info["x_ohm_per_km"].values, line_length)  # Reactance in Ohms
    line_inverse_reactance = np.divide(1, line_reactance)  # Negative susceptance in Siemens/Mhos

    # Thermal limits
    line_i_max = line_info["max_i_ka"].values * 1000  # Thermal limits in Amperes
    line_p_max = np.multiply(np.square(line_i_max), line_resistance)  # Thermal limits in Watts

    # Loads
    loads_info = environment_info["load"]
    n_loads = loads_info.shape[0]

    # DC-OPF Environment parameters
    parameters["n_lines"] = n_lines
    parameters["line_i_max"] = line_i_max
    parameters["line_p_max"] = line_p_max
    parameters["line_reactance"] = line_reactance
    parameters["line_inverse_reactance"] = line_inverse_reactance

    parameters["n_gens"] = n_gens
    parameters["gen_active_power_min"] = gen_active_power_min
    parameters["gen_active_power_max"] = gen_active_power_max
    parameters["n_loads"] = n_loads

    if verbose:
        print_matrix(line_i_max, "line_i_max")
        print_matrix(line_p_max, "line_p_max")

        print_matrix(line_resistance, "line_resistance")
        print_matrix(line_reactance, "line_reactance")
        print_matrix(line_inverse_reactance, "line_inverse_reactance")

        print_matrix(gen_active_power_max, "gen_active_power_max")
        print_matrix(gen_active_power_min, "gen_active_power_min")

    return parameters


def get_dc_opf_observation_parameters(observation, verbose=False):
    if not isinstance(observation, dict):
        observation = observation.to_dict()

    parameters = dict()
    loads_active_power = observation["loads"]["p"]

    # DC-OPF Observation parameters
    parameters["loads_active_power"] = loads_active_power

    if verbose:
        print_matrix(loads_active_power, name="loads_active_power")

    return parameters


def get_bus_susceptance_matrix(dc_opf_params, verbose=False):
    n_buses = len(dc_opf_params["bus_ids"])
    susceptance_matrix = np.zeros((n_buses, n_buses))

    line_or_to_bus_id = dc_opf_params["line_or_to_bus_id"]
    line_ex_to_bus_id = dc_opf_params["line_ex_to_bus_id"]
    line_inverse_reactance = dc_opf_params["line_inverse_reactance"]

    for line_id in dc_opf_params["line_ids"]:
        bus_or = line_or_to_bus_id[line_id]
        bus_ex = line_ex_to_bus_id[line_id]
        inv_reactance = line_inverse_reactance[line_id]

        # Non-diagonal entries
        susceptance_matrix[bus_or, bus_ex] += -inv_reactance
        susceptance_matrix[bus_ex, bus_or] += -inv_reactance

        # Diagonal entries
        susceptance_matrix[bus_or, bus_or] += inv_reactance
        susceptance_matrix[bus_ex, bus_ex] += inv_reactance

    if verbose:
        print(n_buses)
        print_matrix(susceptance_matrix, "B", spacing=10)

    return susceptance_matrix
