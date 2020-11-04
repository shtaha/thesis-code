import graph_nets as gn
import matplotlib.pyplot as plt
import networkx as nx
import sonnet as snt
from graph_nets import utils_np
from graph_nets import utils_tf
import tensorflow as tf

from lib.dc_opf import load_case, GridDCOPF
from lib.gns import obs_to_graph_dict_by_grid, print_graphs_tuple
from lib.tf_utils import print_variables

render = False

case_name = "rte_case5_example"
case = load_case(case_name)
env = case.env
grid = GridDCOPF(case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p)

obs = env.reset()

graph_dict = obs_to_graph_dict_by_grid(obs, grid)
graphs_dict_list = [graph_dict]

graphs_tuple = utils_np.data_dicts_to_graphs_tuple(graphs_dict_list)

graphs_nx = utils_np.graphs_tuple_to_networkxs(graphs_tuple)
recovered_graph_dict_list = utils_np.graphs_tuple_to_data_dicts(graphs_tuple)

if render:
    env.render()
    if len(graphs_tuple.n_node) > 1:
        fig, axs = plt.subplots(ncols=len(graphs_tuple.n_node), figsize=(16, 9))
        for iax, (graph_nx, ax) in enumerate(zip(graphs_nx, axs)):
            nx.draw(graph_nx, ax=ax)
            ax.set_title("Graph {}".format(iax))
        fig.show()
    else:
        fig, ax = plt.subplots(figsize=(16, 9))
        nx.draw(graphs_nx[0], ax=ax)
        ax.set_title("Graph {}".format(0))
        fig.show()

graph_network = gn.modules.GraphNetwork(
    edge_model_fn=lambda: snt.Linear(output_size=graphs_tuple.globals.shape[1]),
    node_model_fn=lambda: snt.Linear(output_size=graphs_tuple.nodes.shape[1]),
    global_model_fn=lambda: snt.Linear(output_size=graphs_tuple.edges.shape[1]),
)

input_graphs = utils_tf.data_dicts_to_graphs_tuple(graphs_dict_list)

num_recurrent_passes = 3
previous_graphs = input_graphs

for _ in range(num_recurrent_passes):
    previous_graphs = graph_network(previous_graphs)
output_graphs = previous_graphs

print_variables(graph_network.trainable_variables)
