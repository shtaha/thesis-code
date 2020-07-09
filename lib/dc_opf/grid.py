import numpy as np
import pandas as pd

from .unit_converter import UnitConverter
from ..data_utils import hot_to_indices
from ..visualizer import describe_substation


def bus_names_to_sub_ids(bus_names):
    sub_ids = np.array([int(bus_name.split("-")[-1]) for bus_name in bus_names])

    # Start with substation with id 0
    sub_zero = int(bus_names[0].split("-")[-1])
    if sub_zero:
        sub_ids = sub_ids - sub_zero

    return sub_ids


class GridDCOPF(UnitConverter):
    def __init__(self, case, base_unit_v, base_unit_p=1e6):
        UnitConverter.__init__(self, base_unit_v=base_unit_v, base_unit_p=base_unit_p)
        self.case = case

        # Initialize grid elements
        self.sub = pd.DataFrame(
            columns=[
                "id",
                "bus",
                "line_or",
                "line_ex",
                "gen",
                "load",
                "ext_grid",
                "n_elements",
            ]
        )
        self.bus = pd.DataFrame(
            columns=[
                "id",
                "sub",
                "sub_bus",
                "v_pu",
                "line_or",
                "line_ex",
                "gen",
                "load",
                "ext_grid",
            ]
        )
        self.line = pd.DataFrame(
            columns=[
                "id",
                "sub_or",
                "sub_ex",
                "bus_or",
                "bus_ex",
                "sub_bus_or",
                "sub_bus_ex",
                "b_pu",
                "p_pu",
                "max_p_pu",
                "status",
                "trafo",
            ]
        )
        self.gen = pd.DataFrame(
            columns=[
                "id",
                "sub",
                "bus",
                "sub_bus",
                "p_pu",
                "min_p_pu",
                "max_p_pu",
                "cost_pu",
            ]
        )
        self.load = pd.DataFrame(columns=["id", "sub", "bus", "sub_bus", "p_pu"])

        # External grid
        self.ext_grid = pd.DataFrame(
            columns=["id", "sub", "bus", "sub_bus", "p_pu", "min_p_pu", "max_p_pu"]
        )

        # Transformer
        self.trafo = pd.DataFrame(
            columns=[
                "id",
                "sub_or",
                "sub_ex",
                "bus_or",
                "bus_ex",
                "sub_bus_or",
                "sub_bus_ex",
                "b_pu",
                "p_pu",
                "max_p_pu",
                "status",
                "trafo",
            ]
        )

        self.slack_bus = None
        self.delta_max = None
        self.fixed_elements = None

        self.build_grid()

    def __str__(self):
        output = "Grid p. u.\n"
        output = (
            output + f"\t - Substations {self.sub.shape} {list(self.sub.columns)}\n"
        )
        output = output + f"\t - Buses {self.bus.shape} {list(self.bus.columns)}\n"
        output = (
            output
            + f"\t - Power lines {self.line[~self.line.trafo].shape} {list(self.line[~self.line.trafo].columns)}\n"
        )
        output = output + f"\t - Generators {self.gen.shape} {list(self.gen.columns)}\n"
        output = output + f"\t - Loads {self.load.shape} {list(self.load.columns)}\n"
        output = (
            output
            + f"\t - External grids {self.ext_grid.shape} {list(self.ext_grid.columns)}\n"
        )
        output = (
            output + f"\t - Transformers {self.trafo.shape} {list(self.trafo.columns)}"
        )
        return output

    def build_grid(self):
        """
            Buses.
        """
        self.bus["id"] = self.case.grid_backend.bus.index
        self.bus["sub"] = bus_names_to_sub_ids(self.case.grid_backend.bus["name"])
        self.bus["v_pu"] = self.convert_kv_to_per_unit(
            self.case.grid_backend.bus["vn_kv"]
        )

        """
            Substations.
        """
        self.sub["id"] = sorted(self.bus["sub"].unique())

        """
            Power lines.
        """
        self.line["id"] = self.case.grid_backend.line.index

        # Inverse line reactance
        # Equation given: https://pandapower.readthedocs.io/en/v2.2.2/elements/line.html.

        x_pu = self.convert_ohm_to_per_unit(
            self.case.grid_backend.line["x_ohm_per_km"]
            * self.case.grid_backend.line["length_km"]
            / self.case.grid_backend.line["parallel"]
        )
        self.line["b_pu"] = 1 / x_pu

        # Power line flow thermal limit
        # P_l_max = I_l_max * V_l
        line_max_i_pu = self.convert_ka_to_per_unit(
            self.case.grid_backend.line["max_i_ka"]
        )
        self.line["max_p_pu"] = (
            np.sqrt(3)
            * line_max_i_pu
            * self.bus["v_pu"].values[self.case.grid_backend.line["from_bus"].values]
        )

        self.line["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.res_line["p_from_mw"]
        )

        # Line status
        self.line["status"] = self.case.grid_backend.line["in_service"]

        """
            Generators.
        """
        self.gen["id"] = self.case.grid_backend.gen.index
        self.gen["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.gen["p_mw"]
        )
        self.gen["max_p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.gen["max_p_mw"]
        )
        self.gen["min_p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.gen["min_p_mw"]
        )
        self.gen["min_p_pu"] = np.maximum(0.0, self.gen["min_p_pu"].values)
        self.gen["cost_pu"] = 1.0

        """
            Loads.
        """
        self.load["id"] = self.case.grid_backend.load.index
        self.load["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.load["p_mw"]
        )

        """
            External grids.
        """
        self.ext_grid["id"] = self.case.grid_backend.ext_grid.index
        self.ext_grid["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.res_ext_grid["p_mw"]
        )
        if "min_p_mw" in self.case.grid_backend.ext_grid.columns:
            self.ext_grid["min_p_pu"] = self.convert_mw_to_per_unit(
                self.case.grid_backend.ext_grid["min_p_mw"]
            )

        if "max_p_mw" in self.case.grid_backend.ext_grid.columns:
            self.ext_grid["max_p_pu"] = self.convert_mw_to_per_unit(
                self.case.grid_backend.ext_grid["max_p_mw"]
            )

        """
            Transformers.
            "High voltage bus is the origin (or) bus."
            Follows definitions from https://pandapower.readthedocs.io/en/v2.2.2/elements/trafo.html.
        """

        self.trafo["id"] = self.case.grid_backend.trafo.index
        if "b_pu" in self.case.grid_backend.trafo.columns:
            self.trafo["b_pu"] = self.case.grid_backend.trafo["b_pu"]
        else:
            self.trafo["b_pu"] = (
                1
                / (self.case.grid_backend.trafo["vk_percent"] / 100.0)
                * self.case.grid_backend.trafo["sn_mva"]
            )

        self.trafo["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid_backend.res_trafo["p_hv_mw"]
        )
        self.trafo["p_pu"].fillna(0, inplace=True)

        if "max_p_pu" in self.case.grid_backend.trafo.columns:
            self.trafo["max_p_pu"] = self.case.grid_backend.trafo["max_p_pu"]
        else:
            self.trafo["max_p_pu"] = self.convert_mw_to_per_unit(
                self.case.grid_backend.trafo["sn_mva"]
            )

        self.trafo["status"] = self.case.grid_backend.trafo["in_service"]

        # Reindex
        self.sub.set_index("id", inplace=True)
        self.bus.set_index("id", inplace=True)
        self.line.set_index("id", inplace=True)
        self.gen.set_index("id", inplace=True)
        self.load.set_index("id", inplace=True)
        self.ext_grid.set_index("id", inplace=True)
        self.trafo.set_index("id", inplace=True)

        """
            Topology.
        """
        # Generators
        self.gen["bus"] = self.case.grid_backend.gen["bus"]
        self.gen["sub"] = self.sub.index.values[self.gen["bus"]]

        # Loads
        self.load["bus"] = self.case.grid_backend.load["bus"]
        self.load["sub"] = self.sub.index.values[self.load["bus"]]

        # Power lines
        self.line["bus_or"] = self.case.grid_backend.line["from_bus"]
        self.line["bus_ex"] = self.case.grid_backend.line["to_bus"]
        self.line["sub_or"] = self.sub.index.values[self.line["bus_or"]]
        self.line["sub_ex"] = self.sub.index.values[self.line["bus_ex"]]

        # External grids
        self.ext_grid["bus"] = self.case.grid_backend.ext_grid["bus"]
        self.ext_grid["sub"] = self.sub.index.values[self.ext_grid["bus"]]

        # Transformers
        self.trafo["bus_or"] = self.case.grid_backend.trafo["hv_bus"]
        self.trafo["bus_ex"] = self.case.grid_backend.trafo["lv_bus"]
        self.trafo["sub_or"] = self.sub.index.values[self.trafo["bus_or"]]
        self.trafo["sub_ex"] = self.sub.index.values[self.trafo["bus_ex"]]

        # Merge power lines and transformers
        self.line["trafo"] = False
        self.trafo["trafo"] = True
        self.line = self.line.append(self.trafo, ignore_index=True)

        sub_bus = np.empty_like(self.bus.index)
        for sub_id in self.sub.index:
            bus_mask = self.bus["sub"] == sub_id
            gen_mask = self.gen["sub"] == sub_id
            load_mask = self.load["sub"] == sub_id
            line_or_mask = self.line["sub_or"] == sub_id
            line_ex_mask = self.line["sub_ex"] == sub_id
            ext_grid_mask = self.ext_grid["sub"] == sub_id

            sub_bus[bus_mask] = np.arange(1, np.sum(bus_mask) + 1)

            self.sub["bus"][sub_id] = tuple(np.flatnonzero(bus_mask))
            self.sub["gen"][sub_id] = tuple(np.flatnonzero(gen_mask))
            self.sub["load"][sub_id] = tuple(np.flatnonzero(load_mask))
            self.sub["line_or"][sub_id] = tuple(np.flatnonzero(line_or_mask))
            self.sub["line_ex"][sub_id] = tuple(np.flatnonzero(line_ex_mask))
            self.sub["ext_grid"][sub_id] = tuple(np.flatnonzero(ext_grid_mask))

        # Bus within a substation of each grid element
        self.bus["sub_bus"] = sub_bus
        self.gen["sub_bus"] = self.bus["sub_bus"].values[self.gen["bus"].values]
        self.load["sub_bus"] = self.bus["sub_bus"].values[self.load["bus"].values]
        self.line["sub_bus_or"] = self.bus["sub_bus"].values[self.line["bus_or"].values]
        self.line["sub_bus_ex"] = self.bus["sub_bus"].values[self.line["bus_ex"].values]
        self.ext_grid["sub_bus"] = self.bus["sub_bus"].values[
            self.ext_grid["bus"].values
        ]
        self.trafo["sub_bus_or"] = self.bus["sub_bus"].values[
            self.trafo["bus_or"].values
        ]
        self.trafo["sub_bus_ex"] = self.bus["sub_bus"].values[
            self.trafo["bus_ex"].values
        ]

        for bus_id in self.bus.index:
            gen_mask = self.gen["bus"] == bus_id
            load_mask = self.load["bus"] == bus_id
            line_or_mask = self.line["bus_or"] == bus_id
            line_ex_mask = self.line["bus_ex"] == bus_id
            ext_grid_mask = self.ext_grid["bus"] == bus_id

            self.bus["gen"][bus_id] = tuple(np.flatnonzero(gen_mask))
            self.bus["load"][bus_id] = tuple(np.flatnonzero(load_mask))
            self.bus["line_or"][bus_id] = tuple(np.flatnonzero(line_or_mask))
            self.bus["line_ex"][bus_id] = tuple(np.flatnonzero(line_ex_mask))
            self.bus["ext_grid"][bus_id] = tuple(np.flatnonzero(ext_grid_mask))

        # Number of elements per substation (without external grids)
        if self.case.env:
            self.sub["n_elements"] = self.case.env.sub_info
        else:
            self.sub["n_elements"] = [
                len(self.sub.line_or[sub_id])
                + len(self.sub.line_ex[sub_id])
                + len(self.sub.gen[sub_id])
                + len(self.sub.load[sub_id])
                for sub_id in self.sub.index
            ]

        # Fill with 0 if no value
        self.line["p_pu"] = self.line["p_pu"].fillna(0)
        self.gen["p_pu"] = self.gen["p_pu"].fillna(0)
        self.ext_grid["p_pu"] = self.ext_grid["p_pu"].fillna(0)

        # Grid and computation parameters
        if not self.case.grid_backend.gen["slack"].any():
            slack_bus = 0
            if len(self.ext_grid.index):
                slack_bus = self.case.grid_backend.ext_grid["bus"][0]
            self.slack_bus = slack_bus
        else:
            self.slack_bus = self.gen.bus[
                np.flatnonzero(self.case.grid_backend.gen["slack"])[0]
            ]

        self.delta_max = np.pi / 2

        # Substation topological symmetry
        self.fixed_elements = self.get_fixed_elements()

    def get_fixed_elements(self, verbose=False):
        """
        Get id of a power line end at each substation. Used for eliminating substation topological symmetry.
        """
        fixed_elements = dict()

        for sub_id in self.sub.index:
            fixed_elements[sub_id] = dict()

            if self.case.env:
                # Grid element ids
                line_or_ids = hot_to_indices(
                    self.case.env.action_space.line_or_to_subid == sub_id
                )
                line_ex_ids = hot_to_indices(
                    self.case.env.action_space.line_ex_to_subid == sub_id
                )

                # Grid element positions within substation
                lines_or_pos = self.case.env.action_space.line_or_to_sub_pos[
                    line_or_ids
                ]
                lines_ex_pos = self.case.env.action_space.line_ex_to_sub_pos[
                    line_ex_ids
                ]

                fixed_elements[sub_id]["line_or"] = line_or_ids[
                    np.flatnonzero(lines_or_pos == 0)
                ].tolist()
                fixed_elements[sub_id]["line_ex"] = line_ex_ids[
                    np.flatnonzero(lines_ex_pos == 0)
                ].tolist()

                if verbose:
                    describe_substation(sub_id, self.case.env)
            else:
                line_or_ids = self.sub["line_or"][sub_id]
                line_ex_ids = self.sub["line_ex"][sub_id]
                if len(line_or_ids):
                    fixed_elements[sub_id]["line_or"] = [line_or_ids[0]]
                    fixed_elements[sub_id]["line_ex"] = []
                elif len(line_ex_ids):
                    fixed_elements[sub_id]["line_or"] = []
                    fixed_elements[sub_id]["line_ex"] = [line_ex_ids[0]]

            # Check if each substation has exactly one power line end at position 0
            assert (
                len(fixed_elements[sub_id]["line_or"])
                + len(fixed_elements[sub_id]["line_ex"])
                == 1
            )

            if verbose:
                print(fixed_elements[sub_id])

        fixed_elements = pd.DataFrame(
            [fixed_elements[sub_id] for sub_id in fixed_elements]
        )
        return fixed_elements

    def print_grid(self):
        print("\nGRID\n")
        print("SUB\n" + self.sub.to_string())
        print("BUS\n" + self.bus.to_string())
        print("LINE\n" + self.line[~self.line["trafo"]].to_string())
        print("GEN\n" + self.gen.to_string())
        print("LOAD\n" + self.load.to_string())
        print("EXT GRID\n" + self.ext_grid.to_string())
        print("TRAFO\n" + self.trafo.to_string())
        print(f"SLACK BUS: {self.slack_bus}")
        print("FIXED ELEMENTS\n" + self.fixed_elements.to_string())
