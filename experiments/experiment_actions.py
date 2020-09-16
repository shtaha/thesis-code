import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np

from lib.constants import Constants as Const
from lib.action_space import get_actions_effects


def analyse_actions(actions, env, agent_name, save_dir=None):
    (
        action_do_nothing,
        action_unitary,
        action_set_bus,
        action_set_line_status,
    ) = get_actions_effects(actions, env)

    data = pd.DataFrame(
        {
            "do_nothing": action_do_nothing,
            "unitary": action_unitary,
            "set_bus": action_set_bus,
            "set_line_status": action_set_line_status,
        }
    )
    data["n_set_bus"] = data["set_bus"].apply(lambda x: len(x))
    data["sub_id"] = data["set_bus"].apply(
        lambda x: list(x.keys())[0] if x.keys() else np.nan
    )
    data["sub_topo"] = data["set_bus"].apply(
        lambda x: "-".join([str(i) for i in x[list(x.keys())[0]]])
        if x.keys()
        else np.nan
    )

    data["n_set_line_status"] = data["set_line_status"].apply(lambda x: len(x))
    data["line_id"] = data["set_line_status"].apply(
        lambda x: list(x.keys())[0] if x.keys() else np.nan
    )
    data["reconnect"] = data["set_line_status"].apply(
        lambda x: x[list(x.keys())[0]] == 1 if x.keys() else np.nan
    )

    data_dn = data[~data["do_nothing"]]

    """
        General plots
    """
    fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
    sns.histplot(
        data=data,
        x="do_nothing",
        discrete=True,
        ax=ax,
        shrink=0.8,
        stat="probability",
        alpha=1.0,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Switch", "Do-nothing"])
    ax.set_xlabel("Do-nothing or switch action")
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-do-nothing"))

    fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
    sns.histplot(
        data=data_dn,
        x="unitary",
        discrete=True,
        ax=ax,
        shrink=0.8,
        stat="probability",
        alpha=1.0,
    )
    ax.set_xticks([0, 1])
    ax.set_xticklabels([False, True])
    ax.set_xlabel("Unitary action")
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-unitary"))

    """
        Line status 
    """
    fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
    sns.histplot(
        data=data_dn,
        x="n_set_line_status",
        hue="unitary",
        discrete=True,
        ax=ax,
        shrink=0.8,
        stat="probability",
        multiple="dodge",
        alpha=1.0,
    )
    ax.set_xticks([0, np.max(data_dn["n_set_line_status"])])
    ax.set_xlabel("Lines switched in a (non-)unitary action")
    ax.legend([True, False], title="Unitary")
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-n-lines"))

    fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
    sns.histplot(
        data=data_dn,
        x="line_id",
        hue="reconnect",
        discrete=True,
        ax=ax,
        shrink=0.5,
        stat="count",
        multiple="dodge",
        alpha=1.0,
    )
    ax.set_xticks(np.arange(0, env.n_line))
    ax.set_xlim(left=-1, right=env.n_line)
    ax.set_xlabel("Line status set")
    ax.legend([True, False], title="Reconnection")
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-set-status-per-line"))

    """
        Substation topology
    """
    # Set bus and Set line status
    fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
    sns.histplot(
        data=data_dn,
        x="n_set_bus",
        hue="unitary",
        discrete=True,
        ax=ax,
        shrink=0.8,
        stat="probability",
        multiple="dodge",
        alpha=1.0,
    )
    ax.set_xticks([0, np.max(data_dn["n_set_bus"])])
    ax.set_xlabel("Substations switched in a (non-)unitary action")
    ax.legend([True, False], title="Unitary")
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-n-subs"))

    # Set bus action per substation
    fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
    sns.histplot(
        data=data_dn,
        x="sub_id",
        discrete=True,
        ax=ax,
        shrink=0.8,
        stat="probability",
        multiple="dodge",
        alpha=1.0,
    )
    ax.set_xticks(np.arange(0, len(env.sub_info)))
    ax.set_xlim(left=-1, right=len(env.sub_info))
    ax.set_xlabel("Substation set bus")
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-set-bus-per-sub"))

    # Substation topologies
    for sub_id in range(len(env.sub_info)):
        data_sub = data_dn[data_dn["sub_id"] == sub_id]
        if len(data_sub):
            counts = data_sub["sub_topo"].value_counts().sort_index()
            ref_topo = False
            if len(counts.index) > 1:
                fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
                ax.set_title(f"Substation {sub_id} topologies")
                for c, count in enumerate(counts):
                    if all([sub_bus == "1" for sub_bus in counts.index[c].split("-")]):
                        color = "tab:red"
                        label = True
                        ref_topo = True
                    else:
                        color = "tab:blue"
                        label = False
                    ax.bar(
                        c, count, width=0.8, color=color, edgecolor="black", label=label
                    )

                ticks = np.arange(0, len(counts))
                if len(counts) > 5:
                    labels = ticks.astype(str).tolist()
                else:
                    labels = list(counts.index)

                plt.xticks(ticks, labels)
                if ref_topo:
                    ax.legend([True, False], title="Ref. top.")

                if save_dir:
                    fig.savefig(
                        os.path.join(save_dir, agent_name + f"-set-sub-{sub_id}-topo")
                    )
