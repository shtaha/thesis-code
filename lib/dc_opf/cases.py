import grid2op
import numpy as np
import pandapower as pp
import pandas as pd

from lib.data_utils import update_backend
from lib.dc_opf.models import UnitConverter


def bus_names_to_sub_ids(bus_names):
    sub_ids = [int(bus_name.split("-")[-1]) for bus_name in bus_names]
    return sub_ids


class GridDCOPF(UnitConverter):
    def __init__(self, case, base_unit_v, base_unit_p=1e6):
        UnitConverter.__init__(self, base_unit_v=base_unit_v, base_unit_p=base_unit_p)
        self.case = case

        # Initialize grid elements
        self.sub = pd.DataFrame(
            columns=["id", "bus", "line_or", "line_ex", "gen", "load", "ext_grid"]
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
                "b_pu",
                "p_pu",
                "max_p_pu",
                "status",
                "trafo",
            ]
        )
        self.gen = pd.DataFrame(
            columns=["id", "sub", "bus", "p_pu", "min_p_pu", "max_p_pu", "cost_pu"]
        )
        self.load = pd.DataFrame(columns=["id", "sub", "bus", "p_pu"])

        # External grid
        self.ext_grid = pd.DataFrame(
            columns=["id", "sub", "bus", "p_pu", "min_p_pu", "max_p_pu"]
        )

        # Transformer
        self.trafo = pd.DataFrame(
            columns=[
                "id",
                "sub_or",
                "sub_ex",
                "bus_or",
                "bus_ex",
                "b_pu",
                "p_pu",
                "max_p_pu",
                "status",
                "trafo",
            ]
        )

        self.slack_bus = None
        self.delta_max = None

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
        self.bus["id"] = self.case.grid.bus.index
        self.bus["sub"] = bus_names_to_sub_ids(self.case.grid.bus["name"])
        self.bus["v_pu"] = self.convert_kv_to_per_unit(self.case.grid.bus["vn_kv"])

        """
            Substations.
        """
        self.sub["id"] = sorted(self.bus["sub"].unique())

        """
            Power lines.
        """
        self.line["id"] = self.case.grid.line.index

        # Inverse line reactance
        x_pu = self.convert_ohm_to_per_unit(
            self.case.grid.line["x_ohm_per_km"]
            * self.case.grid.line["length_km"]
            / self.case.grid.line["parallel"]
        )
        self.line["b_pu"] = 1 / x_pu

        # Power line flow thermal limit
        # P_l_max = I_l_max * V_l
        line_max_i_pu = self.convert_ka_to_per_unit(self.case.grid.line["max_i_ka"])
        self.line["max_p_pu"] = (
            np.sqrt(3)
            * line_max_i_pu
            * self.bus["v_pu"].values[self.case.grid.line["from_bus"].values]
        )

        self.line["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid.res_line["p_from_mw"]
        )

        # Line status
        self.line["status"] = self.case.grid.line["in_service"]

        """
            Generators.
        """
        self.gen["id"] = self.case.grid.gen.index
        self.gen["p_pu"] = self.convert_mw_to_per_unit(self.case.grid.gen["p_mw"])
        self.gen["max_p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid.gen["max_p_mw"]
        )
        self.gen["min_p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid.gen["min_p_mw"]
        )
        self.gen["min_p_pu"] = np.maximum(0.0, self.gen["min_p_pu"].values)
        self.gen["cost_pu"] = 1.0

        self.gen["p_pu"] = self.convert_mw_to_per_unit(self.case.grid.gen["p_mw"])

        """
            Loads.
        """
        self.load["id"] = self.case.grid.load.index
        self.load["p_pu"] = self.convert_mw_to_per_unit(self.case.grid.load["p_mw"])

        """
            External grids.
        """
        self.ext_grid["id"] = self.case.grid.ext_grid.index
        self.ext_grid["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid.res_ext_grid["p_mw"]
        )
        if "min_p_mw" in self.case.grid.ext_grid.columns:
            self.ext_grid["min_p_pu"] = self.convert_mw_to_per_unit(
                self.case.grid.ext_grid["min_p_mw"]
            )

        if "max_p_mw" in self.case.grid.ext_grid.columns:
            self.ext_grid["max_p_pu"] = self.convert_mw_to_per_unit(
                self.case.grid.ext_grid["max_p_mw"]
            )

        """
            Transformers.
            "High voltage bus is the origin (or) bus."
            Follows definitions from https://pandapower.readthedocs.io/en/v2.2.2/elements/trafo.html.
        """

        self.trafo["id"] = self.case.grid.trafo.index
        if "b_pu" in self.case.grid.trafo.columns:
            self.trafo["b_pu"] = self.case.grid.trafo["b_pu"]

        self.trafo["p_pu"] = self.convert_mw_to_per_unit(
            self.case.grid.res_trafo["p_hv_mw"]
        )
        if "max_p_pu" in self.case.grid.trafo.columns:
            self.trafo["max_p_pu"] = self.case.grid.trafo["max_p_pu"]
        self.trafo["status"] = self.case.grid.trafo["in_service"]

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
        self.gen["bus"] = self.case.grid.gen["bus"]
        self.gen["sub"] = self.sub.index.values[self.gen["bus"]]

        # Loads
        self.load["bus"] = self.case.grid.load["bus"]
        self.load["sub"] = self.sub.index.values[self.load["bus"]]

        # Power lines
        self.line["bus_or"] = self.case.grid.line["from_bus"]
        self.line["bus_ex"] = self.case.grid.line["to_bus"]
        self.line["sub_or"] = self.sub.index.values[self.line["bus_or"]]
        self.line["sub_ex"] = self.sub.index.values[self.line["bus_ex"]]

        # External grids
        self.ext_grid["bus"] = self.case.grid.ext_grid["bus"]
        self.ext_grid["sub"] = self.sub.index.values[self.ext_grid["bus"]]

        # Transformers
        self.trafo["bus_or"] = self.case.grid.trafo["hv_bus"]
        self.trafo["bus_ex"] = self.case.grid.trafo["lv_bus"]
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

        self.bus["sub_bus"] = sub_bus

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

        # Fill with 0 if no value
        self.line["p_pu"] = self.line["p_pu"].fillna(0)
        self.gen["p_pu"] = self.gen["p_pu"].fillna(0)
        self.ext_grid["p_pu"] = self.ext_grid["p_pu"].fillna(0)

        # Grid and computation parameters
        self.slack_bus = self.gen.bus[np.flatnonzero(self.case.grid.gen["slack"])[0]]
        self.delta_max = np.pi / 2

    def print_grid(self):
        print("\nGRID\n")
        print("BUS\n" + self.bus.to_string())
        print("LINE\n" + self.line[~self.line["trafo"]].to_string())
        print("GEN\n" + self.gen.to_string())
        print("LOAD\n" + self.load.to_string())
        print("EXT GRID\n" + self.ext_grid.to_string())
        print("TRAFO\n" + self.trafo.to_string())
        print(f"SLACK BUS: {self.slack_bus}")


class OPFCase3(UnitConverter):
    """
    Test case for power flow computation.
    Found in http://research.iaun.ac.ir/pd/bahador.fani/pdfs/UploadFile_6990.pdf.
    """

    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=110000.0)

        self.name = "Case 3"

        self.grid = self.build_case3_grid()
        self.grid_backend = self.grid

    def build_case3_grid(self):
        grid = pp.create_empty_network()

        # Substation bus 1
        bus0 = pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-0-0")
        bus1 = pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-1-1")
        bus2 = pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-2-2")

        # Substation bus 2
        pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-3-0")
        pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-4-1")
        pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-5-2")

        pp.create_line_from_parameters(
            grid,
            bus0,
            bus1,
            length_km=1.0,
            r_ohm_per_km=0.01 * self.base_unit_z,
            x_ohm_per_km=1.0 / 3.0 * self.base_unit_z,
            c_nf_per_km=0.0,
            max_i_ka=self.convert_per_unit_to_ka(1.0),
            name="line-0",
            type="ol",
            max_loading_percent=100.0,
        )

        pp.create_line_from_parameters(
            grid,
            bus0,
            bus2,
            length_km=1.0,
            r_ohm_per_km=0.01 * self.base_unit_z,
            x_ohm_per_km=1.0 / 2.0 * self.base_unit_z,
            c_nf_per_km=0.0,
            max_i_ka=self.convert_per_unit_to_ka(1.0),
            name="line-1",
            type="ol",
            max_loading_percent=100.0,
        )

        pp.create_line_from_parameters(
            grid,
            bus1,
            bus2,
            length_km=1.0,
            r_ohm_per_km=0.01 * self.base_unit_z,
            x_ohm_per_km=1.0 / 2.0 * self.base_unit_z,
            c_nf_per_km=0.0,
            max_i_ka=self.convert_per_unit_to_ka(1.0),
            name="line-2",
            type="ol",
            max_loading_percent=100.0,
        )

        pp.create_load(
            grid,
            bus1,
            p_mw=self.convert_per_unit_to_mw(0.5),
            name="load-0",
            controllable=False,
        )
        pp.create_load(
            grid,
            bus2,
            p_mw=self.convert_per_unit_to_mw(1.0),
            name="load-1",
            controllable=False,
        )
        pp.create_gen(
            grid,
            bus0,
            p_mw=self.convert_per_unit_to_mw(1.5),
            min_p_mw=self.convert_per_unit_to_mw(0.0),
            max_p_mw=self.convert_per_unit_to_mw(2.0),
            slack=True,
            name="gen-0",
        )

        return grid


class OPFCase6(UnitConverter):
    """
    Test case for power flow computation.
    Found in http://research.iaun.ac.ir/pd/bahador.fani/pdfs/UploadFile_6990.pdf.
    """

    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=110000.0)

        self.name = "Case 6"

        self.grid = self.build_case6_grid()
        self.grid_backend = self.grid

    def build_case6_grid(self):
        grid = pp.create_empty_network()

        # Buses
        # Substation bus 1
        bus0 = pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-0-0")
        bus1 = pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-1-1")
        bus2 = pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-2-2")
        bus3 = pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-3-3")
        bus4 = pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-4-4")
        bus5 = pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-5-5")

        # Substation bus 2
        pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-6-0")
        pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-7-1")
        pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-8-2")
        pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-9-3")
        pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-10-4")
        pp.create_bus(grid, vn_kv=self.base_unit_v / 1000, name="bus-11-5")

        # Lines
        pp.create_line_from_parameters(
            grid,
            bus0,
            bus1,
            length_km=1.0,
            r_ohm_per_km=1e-3 * self.base_unit_z,  # Dummy
            x_ohm_per_km=1.0 / 4.0 * self.base_unit_z,
            c_nf_per_km=0.0,  # Dummy
            max_i_ka=self.convert_per_unit_to_ka(2.0),
            name="line-0",
            type="ol",
            max_loading_percent=100.0,
        )
        pp.create_line_from_parameters(
            grid,
            bus0,
            bus3,
            length_km=1.0,
            r_ohm_per_km=1e-3 * self.base_unit_z,  # Dummy
            x_ohm_per_km=1.0 / 4.706 * self.base_unit_z,
            c_nf_per_km=0.0,  # Dummy
            max_i_ka=self.convert_per_unit_to_ka(2.0),
            name="line-1",
            type="ol",
            max_loading_percent=100.0,
        )
        pp.create_line_from_parameters(
            grid,
            bus0,
            bus4,
            length_km=1.0,
            r_ohm_per_km=1e-3 * self.base_unit_z,  # Dummy
            x_ohm_per_km=1.0 / 3.102 * self.base_unit_z,
            c_nf_per_km=0.0,  # Dummy
            max_i_ka=self.convert_per_unit_to_ka(2.0),
            name="line-2",
            type="ol",
            max_loading_percent=100.0,
        )
        pp.create_line_from_parameters(
            grid,
            bus1,
            bus2,
            length_km=1.0,
            r_ohm_per_km=1e-3 * self.base_unit_z,  # Dummy
            x_ohm_per_km=1.0 / 3.846 * self.base_unit_z,
            c_nf_per_km=0.0,  # Dummy
            max_i_ka=self.convert_per_unit_to_ka(2.0),
            name="line-3",
            type="ol",
            max_loading_percent=100.0,
        )
        pp.create_line_from_parameters(
            grid,
            bus1,
            bus3,
            length_km=1.0,
            r_ohm_per_km=1e-3 * self.base_unit_z,  # Dummy
            x_ohm_per_km=1.0 / 8.001 * self.base_unit_z,
            c_nf_per_km=0.0,  # Dummy
            max_i_ka=self.convert_per_unit_to_ka(2.0),
            name="line-4",
            type="ol",
            max_loading_percent=100.0,
        )
        pp.create_line_from_parameters(
            grid,
            bus1,
            bus4,
            length_km=1.0,
            r_ohm_per_km=1e-3 * self.base_unit_z,  # Dummy
            x_ohm_per_km=1.0 / 3.0 * self.base_unit_z,
            c_nf_per_km=0.0,  # Dummy
            max_i_ka=self.convert_per_unit_to_ka(2.0),
            name="line-5",
            type="ol",
            max_loading_percent=100.0,
        )
        pp.create_line_from_parameters(
            grid,
            bus1,
            bus5,
            length_km=1.0,
            r_ohm_per_km=1e-3 * self.base_unit_z,  # Dummy
            x_ohm_per_km=1.0 / 1.454 * self.base_unit_z,
            c_nf_per_km=0.0,  # Dummy
            max_i_ka=self.convert_per_unit_to_ka(2.0),
            name="line-6",
            type="ol",
            max_loading_percent=100.0,
        )
        pp.create_line_from_parameters(
            grid,
            bus2,
            bus4,
            length_km=1.0,
            r_ohm_per_km=1e-3 * self.base_unit_z,  # Dummy
            x_ohm_per_km=1.0 / 3.175 * self.base_unit_z,
            c_nf_per_km=0.0,  # Dummy
            max_i_ka=self.convert_per_unit_to_ka(2.0),
            name="line-7",
            type="ol",
            max_loading_percent=100.0,
        )
        pp.create_line_from_parameters(
            grid,
            bus2,
            bus5,
            length_km=1.0,
            r_ohm_per_km=1e-3 * self.base_unit_z,  # Dummy
            x_ohm_per_km=1.0 / 9.6157 * self.base_unit_z,
            c_nf_per_km=0.0,  # Dummy
            max_i_ka=self.convert_per_unit_to_ka(2.0),
            name="line-8",
            type="ol",
            max_loading_percent=100.0,
        )
        pp.create_line_from_parameters(
            grid,
            bus3,
            bus4,
            length_km=1.0,
            r_ohm_per_km=1e-3 * self.base_unit_z,  # Dummy
            x_ohm_per_km=1.0 / 2.0 * self.base_unit_z,
            c_nf_per_km=0.0,  # Dummy
            max_i_ka=self.convert_per_unit_to_ka(2.0),
            name="line-9",
            type="ol",
            max_loading_percent=100.0,
        )
        pp.create_line_from_parameters(
            grid,
            bus4,
            bus5,
            length_km=1.0,
            r_ohm_per_km=1e-3 * self.base_unit_z,  # Dummy
            x_ohm_per_km=1.0 / 3.0 * self.base_unit_z,
            c_nf_per_km=0.0,  # Dummy
            max_i_ka=self.convert_per_unit_to_ka(2.0),
            name="line-10",
            type="ol",
            max_loading_percent=100.0,
        )

        # Loads
        pp.create_load(
            grid,
            bus3,
            p_mw=self.convert_per_unit_to_mw(0.9),
            name="load-0",
            controllable=False,
        )
        pp.create_load(
            grid,
            bus4,
            p_mw=self.convert_per_unit_to_mw(1.0),
            name="load-1",
            controllable=False,
        )
        pp.create_load(
            grid,
            bus5,
            p_mw=self.convert_per_unit_to_mw(0.9),
            name="load-2",
            controllable=False,
        )

        # Generators
        pp.create_gen(
            grid,
            bus0,
            p_mw=self.convert_per_unit_to_mw(1.0),
            min_p_mw=self.convert_per_unit_to_mw(0.5),
            max_p_mw=self.convert_per_unit_to_mw(1.5),
            slack=True,
            name="gen-0",
        )
        pp.create_gen(
            grid,
            bus1,
            p_mw=self.convert_per_unit_to_mw(0.9),
            min_p_mw=self.convert_per_unit_to_mw(0.5),
            max_p_mw=self.convert_per_unit_to_mw(2.0),
            name="gen-1",
        )
        pp.create_gen(
            grid,
            bus2,
            p_mw=self.convert_per_unit_to_mw(0.9),
            min_p_mw=self.convert_per_unit_to_mw(0.3),
            max_p_mw=self.convert_per_unit_to_mw(1.0),
            name="gen-2",
        )

        return grid


class OPFCase4(UnitConverter):
    """
    Test case for power flow computation, including an external grid and a transformer.
    Found in https://github.com/e2nIEE/pandapower/blob/develop/tutorials/opf_curtail.ipynb.
    """

    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=110000.0)

        self.name = "Case 4"

        self.grid = self.build_case4_grid()
        self.grid_backend = self.grid

    def build_case4_grid(self):
        grid = pp.create_empty_network()

        # Substation buses 1
        bus0 = pp.create_bus(
            grid,
            vn_kv=2 * self.base_unit_v / 1000,
            min_vm_pu=1.0,
            max_vm_pu=1.02,
            name="bus-0-0",
        )
        bus1 = pp.create_bus(
            grid,
            vn_kv=self.base_unit_v / 1000,
            min_vm_pu=1.0,
            max_vm_pu=1.02,
            name="bus-1-1",
        )
        bus2 = pp.create_bus(
            grid,
            vn_kv=self.base_unit_v / 1000,
            min_vm_pu=1.0,
            max_vm_pu=1.02,
            name="bus-2-2",
        )
        bus3 = pp.create_bus(
            grid,
            vn_kv=self.base_unit_v / 1000,
            min_vm_pu=1.0,
            max_vm_pu=1.02,
            name="bus-3-3",
        )

        # Substation buses 2
        pp.create_bus(
            grid,
            vn_kv=2 * self.base_unit_v / 1000,
            min_vm_pu=1.0,
            max_vm_pu=1.02,
            name="bus-4-0",
        )
        pp.create_bus(
            grid,
            vn_kv=self.base_unit_v / 1000,
            min_vm_pu=1.0,
            max_vm_pu=1.02,
            name="bus-5-1",
        )
        pp.create_bus(
            grid,
            vn_kv=self.base_unit_v / 1000,
            min_vm_pu=1.0,
            max_vm_pu=1.02,
            name="bus-6-2",
        )
        pp.create_bus(
            grid,
            vn_kv=self.base_unit_v / 1000,
            min_vm_pu=1.0,
            max_vm_pu=1.02,
            name="bus-7-3",
        )

        # Transformer
        pp.create_transformer_from_parameters(
            grid,
            hv_bus=bus0,
            lv_bus=bus1,
            name="trafo-0",
            sn_mva=3.5,
            vn_hv_kv=2 * self.base_unit_v / 1000,
            vn_lv_kv=self.base_unit_v / 1000,
            vk_percent=12.5,
            vkr_percent=0.0,
            pfe_kw=0.0,
            i0_percent=0.0,
            shift_degree=0.0,
            max_loading_percent=100.0,
        )

        pp.create_line_from_parameters(
            grid,
            bus1,
            bus2,
            length_km=1.0,
            r_ohm_per_km=0.0,
            x_ohm_per_km=1.0 / 4.0 * self.base_unit_z,
            c_nf_per_km=0.0,
            max_i_ka=self.convert_per_unit_to_ka(5.0),
            name="line-0",
            type="ol",
            max_loading_percent=100.0,
        )

        pp.create_line_from_parameters(
            grid,
            bus2,
            bus3,
            length_km=1.0,
            r_ohm_per_km=0.0,
            x_ohm_per_km=1.0 / 6.0 * self.base_unit_z,
            c_nf_per_km=0.0,
            max_i_ka=self.convert_per_unit_to_ka(5.0),
            name="line-1",
            type="ol",
            max_loading_percent=100.0,
        )

        pp.create_line_from_parameters(
            grid,
            bus3,
            bus1,
            length_km=1.0,
            r_ohm_per_km=0.0,
            x_ohm_per_km=1.0 / 5.0 * self.base_unit_z,
            c_nf_per_km=0.0,
            max_i_ka=self.convert_per_unit_to_ka(4.0),
            name="line-2",
            type="ol",
            max_loading_percent=100.0,
        )

        # Loads
        pp.create_load(grid, bus1, p_mw=2, name="load-0", controllable=False)
        pp.create_load(grid, bus2, p_mw=3, name="load-1", controllable=False)
        pp.create_load(grid, bus3, p_mw=6, name="load-2", controllable=False)

        # Generators
        pp.create_gen(
            grid,
            bus0,
            p_mw=0,
            min_p_mw=0,
            max_p_mw=1,
            vm_pu=1.01,
            name="gen-0",
            controllable=True,
            slack=True,
        )
        pp.create_gen(
            grid,
            bus2,
            p_mw=0,
            min_p_mw=0,
            max_p_mw=5,
            vm_pu=1.01,
            name="gen-1",
            controllable=True,
        )
        pp.create_gen(
            grid,
            bus3,
            p_mw=0,
            min_p_mw=0,
            max_p_mw=8,
            vm_pu=1.01,
            name="gen-2",
            controllable=True,
        )

        # External grids
        pp.create_ext_grid(
            grid,
            bus0,
            va_degree=0.0,
            name="ext-grid-0",
            max_p_mw=3.0,
            min_p_mw=0.0,
            max_loading_percent=100.0,
        )

        grid.trafo["b_pu"] = 28.0  # Empirically: x = vk_percent / 100 * 1 / sn_mva
        grid.trafo["max_p_pu"] = 3.5  # sn_mva

        return grid


class OPFRTECase5(UnitConverter):
    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=1e5)

        self.name = "Case RTE 5"

        self.env = grid2op.make(dataset="rte_case5_example")

        self.grid = update_backend(self.env)
        self.grid_backend = self.env.backend._grid


class OPFL2RPN2019(UnitConverter):
    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=1e5)

        self.name = "Case L2RPN 2019"

        self.env = grid2op.make(dataset="l2rpn_2019")

        self.grid = update_backend(self.env)
        self.grid_backend = self.env.backend._grid


class OPFL2RPN2020(UnitConverter):
    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=138000)

        self.name = "Case L2RPN 2020 WCCI"

        self.env = grid2op.make(dataset="l2rpn_wcci_2020")

        self.grid = update_backend(self.env)
        self.grid_backend = self.env.backend._grid
