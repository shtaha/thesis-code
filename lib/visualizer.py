import json
import os
import warnings
import itertools
import numpy as np


def custom_formatwarning(msg, *args, **kwargs):
    del args, kwargs  # Unused.
    return str(msg) + '\n'


warnings.formatwarning = custom_formatwarning


def print_action(action):
    representation = action.__str__()
    representation = [line.strip() for line in representation.split("\n")[1:] if "NOT" not in line]
    if len(representation) > 0:
        assert not action.is_ambiguous()[0]
        print("action", "\n".join(representation))
    else:
        print("action: do nothing")


def print_matrix(matrix, name=None, spacing=None, decimals=4):
    if name:
        if type(matrix) == np.ndarray:
            shape = matrix.shape
        else:
            shape = len(matrix)

        print(name, "=", str(shape), str(type(matrix)))

    lines = []
    matrix = np.squeeze(matrix)
    matrix = np.atleast_2d(matrix)

    max_value = np.max(np.abs(matrix))
    if spacing:
        spacing = int(np.log10(max_value)) + 2

    for row in matrix:
        line = ""
        for cell in row:
            if cell == 0 or np.abs(int(cell) - cell) < 1e-12:
                pattern = "{:>" + str(int(spacing)) + "}"
                line = line + pattern.format(int(cell))
            else:
                pattern = "{:>" + str(int(spacing)) + "." + str(int(decimals)) + "}"
                line = line + pattern.format(cell)

        lines.append(line)
    print("\n".join(lines))
    print("\n")


def print_trainable_variables(model):
    for var in model.trainable_variables:
        print(var.name, var.shape, np.linalg.norm(var.numpy()))


def render_and_save(environment, save_dir, fig_title):
    fig = environment.render()
    fig.suptitle(fig_title)
    fig.savefig(os.path.join(save_dir, fig_title))
    fig.show()


def get_line_status(observation):
    line_status = [f"Line id {i}: {int(observation.line_status[i])}" for i in range(observation.n_line)]
    return line_status


def get_line_topology(observation):
    topology_vect = observation.topo_vect
    line_topology = [f"Line id {i}: {topology_vect[pos_or]}, {topology_vect[pos_ex]}" for i, (pos_or, pos_ex) in
                     enumerate(zip(observation.line_or_pos_topo_vect, observation.line_ex_pos_topo_vect))]

    return line_topology


def get_gen_topology(observation):
    topology_vect = observation.topo_vect
    gen_topology = [f"Gen id {i}: {topology_vect[pos]}" for i, pos in enumerate(observation.gen_pos_topo_vect)]
    return gen_topology


def get_load_topology(observation):
    topology_vect = observation.topo_vect
    load_topology = [f"Load id {i}: {topology_vect[pos]}" for i, pos in enumerate(observation.load_pos_topo_vect)]
    return load_topology


def print_topology_changes(observation, observation_next, p_line_status=False, p_line_topology=False,
                           p_gen_topology=False, p_load_topology=False):
    def before_after(inputs, inputs_next):
        changes = list()
        if len(inputs - inputs_next):
            changes.append("BEFORE:" + "\t|\t".join([f"{status}" for status in list(inputs - inputs_next)]))
        if len(inputs_next - inputs):
            changes.append("AFTER:" + "\t|\t".join([f"{status}" for status in list(inputs_next - inputs)]))
        return changes

    if p_line_status:
        line_status = set(get_line_status(observation))
        line_status_next = set(get_line_status(observation_next))

        line_changes = before_after(line_status, line_status_next)

        if line_changes:
            print("Line Status changes:\n" + "\n".join(line_changes))
        else:
            print("Line Status changes: None")

    if p_line_topology:
        line_topology = set(get_line_topology(observation))
        line_topology_next = set(get_line_topology(observation_next))

        topology = before_after(line_topology, line_topology_next)

        if topology:
            print("Line topology changes:\n" + "\n".join(topology))
        else:
            print("Line topology changes: None")

    if p_gen_topology:
        gen_topology = set(get_gen_topology(observation))
        gen_topology_next = set(get_gen_topology(observation_next))

        topology = before_after(gen_topology, gen_topology_next)

        if topology:
            print("Gen topology changes:\n" + "\n".join(topology))

    if p_load_topology:
        load_topology = set(get_load_topology(observation))
        load_topology_next = set(get_load_topology(observation_next))

        topology = before_after(load_topology, load_topology_next)

        if topology:
            print("Load topology changes:\n" + "\n".join(topology))


def print_info(info, done, reward):
    is_illegal = info["is_illegal"]
    is_ambiguous = info["is_ambiguous"]
    # is_dispatching_illegal = info["is_dispatching_illegal"]
    # is_illegal_reco = info["is_illegal_reco"]
    exceptions = info["exception"]

    print(f"Done: {done} with Reward {reward}")
    print(f"Action: ILLEGAL = {is_illegal} AMBIGUOUS {is_ambiguous}")
    if exceptions:
        for exception in exceptions:
            warnings.warn(f"Exception raised: {exception}")


def print_rho(observation):
    rho = "\t|\t".join(
        ["Line id {}: {:.2f}".format(i, r) for i, r in enumerate(observation.rho) if r >= 0.8 or r == 0.0])
    if rho:
        print("Line rho:\n" + rho)


def print_parameters(environment):
    print(json.dumps(environment.get_params_for_runner()["parameters_path"], indent=2))


def describe_substation(subid, environment):
    n_elements = environment.sub_info[subid]
    gens = [gen for gen, sub in enumerate(environment.gen_to_subid) if sub == subid]
    loads = [load for load, sub in enumerate(environment.load_to_subid) if sub == subid]
    lines_or = [line for line, sub in enumerate(environment.line_or_to_subid) if sub == subid]
    lines_ex = [line for line, sub in enumerate(environment.line_ex_to_subid) if sub == subid]

    pos_gens = [pos for gen, pos in enumerate(environment.gen_to_sub_pos) if gen in gens]
    pos_loads = [pos for load, pos in enumerate(environment.load_to_sub_pos) if load in loads]
    pos_lines_or = [pos for line, pos in enumerate(environment.line_or_to_sub_pos) if line in lines_or]
    pos_lines_ex = [pos for line, pos in enumerate(environment.line_ex_to_sub_pos) if line in lines_ex]

    elements = list(itertools.chain(lines_or, lines_ex, gens, loads))
    positions = list(itertools.chain(pos_lines_or, pos_lines_ex, pos_gens, pos_loads))

    if n_elements != len(gens) + len(loads) + len(lines_or) + len(lines_ex):
        raise ValueError("Element counts do not match.")

    print(f"substation id: {subid} {n_elements}")
    print(f"ids: lines_or {lines_or} lines_ex {lines_ex} gens {gens} loads {loads}")
    print(f"pos: lines_or {pos_lines_or} lines_ex {pos_lines_ex} gens {pos_gens} loads {pos_loads}")


def describe_environment(environment):
    print(f"obs_space {environment.observation_space.size()}")
    print(f"action_space {environment.action_space.n}")

    print(f"n_gen {environment.n_gen}")
    print(f"n_load {environment.n_load}")
    print(f"n_line {environment.n_line}")
    print(f"n_sub {len(environment.sub_info)}")

    sub_info = ", ".join(["{}:{:>2}".format(i, sub) for i, sub in enumerate(environment.sub_info)])
    print(f"sub_info {sub_info}")
    print(f"dim_topo {environment.dim_topo}")

    print(f"load_to_subid {environment.action_space.load_to_subid}")
    print(f"gen_to_subid {environment.action_space.gen_to_subid}")

    line_or_to_subid = ", ".join(["{:>3}".format(subid) for subid in environment.line_or_to_subid])
    line_ex_to_subid = ", ".join(["{:>3}".format(subid) for subid in environment.line_ex_to_subid])
    print(f"line_or_to_subid {line_or_to_subid}")
    print(f"line_ex_to_subid {line_ex_to_subid}")
