import os
from collections import Counter

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from lib.action_space import get_actions_effects
from lib.constants import Constants as Const
from lib.dc_opf import TopologyConverter
from lib.visualizer import pprint


def analyse_actions(actions, case, agent_name, save_dir=None):
    env = case.env
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

    sub_sep_str = " "
    data["n_set_bus"] = data["set_bus"].apply(lambda x: len(x))
    data["sub_id"] = data["set_bus"].apply(
        lambda x: list(x.keys())[0] if x.keys() else np.nan
    )
    data["sub_topo"] = data["set_bus"].apply(
        lambda x: sub_sep_str.join([str(i) for i in x[list(x.keys())[0]]])
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
    # ax.set_title("Do-nothing or switch action")
    ax.set_xlabel(None)
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-do-nothing"))
    plt.close(fig)

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
    # ax.set_title("Unitary action")
    ax.set_xlabel(None)
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-unitary"))
    plt.close(fig)

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
    ax.legend([True, False], title="Unitary")
    ax.set_xticks([0, np.max(data_dn["n_set_line_status"])])
    # ax.set_title("Lines switched in a (non-)unitary action")
    ax.set_xlabel(None)
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-n-lines"))
    plt.close(fig)

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
    ax.legend([True, False], title="Reconnection")
    # ax.set_title("Line status set")
    ax.set_xlabel(None)
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-set-status-per-line"))
    plt.close(fig)

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
    ax.legend([True, False], title="Unitary")
    # ax.set_title("Substations switched in a (non-)unitary action")
    ax.set_xlabel(None)
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-n-subs"))
    plt.close(fig)

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
    ax.set_xlabel(None)
    plt.tight_layout()
    if save_dir:
        fig.savefig(os.path.join(save_dir, agent_name + "-set-bus-per-sub"))
    plt.close(fig)

    # Substation topologies
    for sub_id in range(len(env.sub_info)):
        data_sub = data_dn[data_dn["sub_id"] == sub_id]
        if len(data_sub):
            counts = data_sub["sub_topo"].value_counts().sort_index()

            # Filter topologies with disconnections
            # counts = counts[[True if "-1" not in idx.split(" ") else False for idx in counts.index]]

            ref_topo = False
            if len(counts.index) > 3:
                fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
                ax.set_title(
                    r"Action target topologies at substation $s_{}$".format("{" + str(sub_id + 1) + "}")
                )
                for c, count in enumerate(counts):
                    if all(
                            [
                                sub_bus == "1"
                                for sub_bus in counts.index[c].split(sub_sep_str)
                            ]
                    ):
                        color = "tab:red"
                        ref_topo = True
                    elif any(
                            [
                                sub_bus == "-1"
                                for sub_bus in counts.index[c].split(sub_sep_str)
                            ]
                    ):
                        color = "tab:green"
                        ref_topo = False
                    else:
                        color = "tab:blue"

                    ax.bar(c, count, width=0.8, color=color, edgecolor="black")

                ticks = np.arange(0, len(counts))
                labels = list(counts.index)

                plt.xticks(ticks, labels, rotation=75)
                if ref_topo:
                    ax.legend(
                        title="Ref. top.",
                        handles=[
                            mpatches.Patch(color="tab:blue", label="False"),
                            mpatches.Patch(color="tab:red", label="True"),
                            mpatches.Patch(color="tab:green", label="-1"),
                        ],
                    )

                # Enumerate topologies
                ylim = ax.get_ylim()
                j = 0
                for i, x in enumerate(ticks):
                    if "-1" not in counts.index[i].split(sub_sep_str):
                        ax.text(
                            x,
                            counts.iloc[i] + ylim[1] * 0.02,
                            j,
                            horizontalalignment="center",
                        )
                        j = j + 1

                plt.tight_layout()
                if save_dir:
                    fig.savefig(
                        os.path.join(save_dir, agent_name + f"-set-sub-{sub_id}-topo")
                    )
                plt.close(fig)


def analyse_loading(obses, case, agent_name, save_dir=None):
    env = case.env

    rhos = np.vstack([obs.rho for obs in obses])

    critical_rho = 0.85
    means = rhos.mean(axis=0)
    stds = rhos.std(axis=0)
    criticals = np.greater(rhos, critical_rho).sum(axis=0)
    overloads = np.greater(rhos, 1.0).sum(axis=0)

    n_critical_all = criticals.sum()
    n_overloaded_all = overloads.sum()

    max_ids = np.argsort(criticals)

    for line_id in reversed(max_ids[-3:]):
        sub_or = env.line_or_to_subid[line_id]
        sub_ex = env.line_ex_to_subid[line_id]

        critical = np.greater(rhos[:, line_id], critical_rho)
        overloaded = np.greater(rhos[:, line_id], 1.0)

        n_critical = critical.sum()
        n_overloaded = overloaded.sum()

        if n_critical > 0:
            pprint(
                f"    - Line {line_id}",
                sub_or,
                sub_ex,
                "{:.3f} + {:.3f}".format(means[line_id], stds[line_id]),
            )
            pprint(
                "        - Critical:",
                n_critical,
                "{:.3f} % / {:.3f} %".format(
                    100 * critical.mean(), 100 * n_critical / n_critical_all
                ),
            )
            pprint(
                "        - Overloaded:",
                n_overloaded,
                "{:.3f} % / {:.3f} %".format(
                    100 * overloaded.mean(), 100 * n_overloaded / n_overloaded_all
                ),
            )

        fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
        sns.histplot(data=rhos[:, line_id], ax=ax)
        ax.set_xlabel(r"$\rho$")
        ax.set_ylabel(r"Count")
        ax.set_xlim(left=0.0, right=1.5)
        ax.set_title(f"Line {line_id}")
        # fig.suptitle(f"{case.name} - {agent_name}")

        plt.tight_layout()
        if save_dir:
            fig.savefig(os.path.join(save_dir, agent_name + f"-line-{line_id}-loading"))
        plt.close(fig)


def analyse_topologies(obses, case, agent_name, save_dir=None):
    tc = TopologyConverter(case.env)
    topologies = np.vstack([obs.topo_vect for obs in obses]).astype(np.int)

    sep_str = " "
    topo_str = [sep_str.join(t.astype(str)) for t in topologies]
    counts = Counter(topo_str)
    fig, ax = plt.subplots(figsize=Const.FIG_SIZE)

    ax.set_title("Grid topologies")

    ticks = []
    labels = []
    for i, (topo, count) in enumerate(counts.most_common(25)):
        if np.equal(np.array(topo.split(sep_str)).astype(int), 1).all():
            color = "tab:red"
        else:
            color = "tab:blue"

        ax.bar(i, count, width=0.8, color=color, edgecolor="black")
        ticks.append(i)
        labels.append(topo)

    plt.xticks(ticks, labels, rotation=90)
    plt.tight_layout()
    if save_dir:
        fig.savefig(
            os.path.join(save_dir, agent_name + f"-topo")
        )
    plt.close(fig)

    for sub_id in range(tc.n_sub):
        mask_sub = tc.substation_topology_mask[sub_id, :]
        topo_sub = topologies[:, mask_sub]

        topo_sub_str = [sep_str.join(t.astype(str)) for t in topo_sub]
        counts = Counter(topo_sub_str)

        if len(counts) > 4:
            fig, ax = plt.subplots(figsize=Const.FIG_SIZE)
            ax.set_title(
                r"Substation $s_{}$ topologies".format("{" + str(sub_id + 1) + "}")
            )

            ticks = []
            labels = []
            for i, (topo, count) in enumerate(counts.most_common(16)):
                if np.equal(np.array(topo.split(sep_str)).astype(int), 1).all():
                    color = "tab:red"
                elif np.equal(np.array(topo.split(sep_str)).astype(int), -1).any():
                    color = "tab:green"
                else:
                    color = "tab:blue"

                ax.bar(i, count, width=0.8, color=color, edgecolor="black")
                ticks.append(i)
                labels.append(topo)

            plt.xticks(ticks, labels, rotation=75)
            plt.tight_layout()
            if save_dir:
                fig.savefig(
                    os.path.join(save_dir, agent_name + f"-sub-{sub_id}-topo")
                )
            plt.close(fig)
