import pandapower as pp
import pandas as pd
import numpy as np
import grid2op

from lib.dc_opf.models import UnitConverter
from lib.data_utils import bus_names_to_sub_ids, update_backend


class SubstationMixin:
    @staticmethod
    def update_grid_with_substation_info(grid):
        grid.bus["sub_id"] = bus_names_to_sub_ids(grid.bus["name"])
        sub_ids = sorted(grid.bus["sub_id"].unique())
        grid.bus["sub_bus_id"] = np.nan

        bus_to_sub_ids = np.empty_like(grid.bus.index.values)
        for sub_id in sub_ids:
            sub_id_mask = grid.bus["sub_id"] == sub_id
            bus_to_sub_ids[sub_id_mask] = np.arange(1, np.sum(sub_id_mask) + 1)

        grid.bus["sub_bus_id"] = bus_to_sub_ids

        # Substations
        grid.line["from_sub"] = grid.bus["sub_id"].values[
            grid.line["from_bus"].values.astype(int)
        ]
        grid.line["to_sub"] = grid.bus["sub_id"].values[
            grid.line["to_bus"].values.astype(int)
        ]
        grid.gen["sub"] = grid.bus["sub_id"].values[grid.gen["bus"].values.astype(int)]
        grid.load["sub"] = grid.bus["sub_id"].values[
            grid.load["bus"].values.astype(int)
        ]

        grid.sub = pd.DataFrame(index=sub_ids)
        grid.sub["bus"] = [
            tuple(grid.bus.index.values[grid.bus["sub_id"] == sub_id])
            for sub_id in sub_ids
        ]
        grid.sub["sub_bus"] = [
            tuple(grid.bus["sub_bus_id"][grid.bus["sub_id"] == sub_id])
            for sub_id in sub_ids
        ]
        grid.sub["gen"] = [
            tuple(grid.gen.index.values[grid.gen["sub"] == sub_id])
            for sub_id in sub_ids
        ]
        grid.sub["load"] = [
            tuple(grid.load.index.values[grid.load["sub"] == sub_id])
            for sub_id in sub_ids
        ]
        grid.sub["line_or"] = [
            tuple(grid.line.index.values[grid.line["from_sub"] == sub_id])
            for sub_id in sub_ids
        ]
        grid.sub["line_ex"] = [
            tuple(grid.line.index.values[grid.line["to_sub"] == sub_id])
            for sub_id in sub_ids
        ]


class OPFCase3(UnitConverter, SubstationMixin):
    """
    Test case for power flow computation.
    Found in http://research.iaun.ac.ir/pd/bahador.fani/pdfs/UploadFile_6990.pdf.
    """

    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=110000.0)

        self.name = "Case 3"

        self.grid = self.build_case3_grid()

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
            c_nf_per_km=0.0001,
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
            c_nf_per_km=0.0001,
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
            c_nf_per_km=0.0001,
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

        # Substations
        self.update_grid_with_substation_info(grid)

        return grid


class OPFCase6(UnitConverter, SubstationMixin):
    """
    Test case for power flow computation.
    Found in http://research.iaun.ac.ir/pd/bahador.fani/pdfs/UploadFile_6990.pdf.
    """

    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=110000.0)

        self.name = "Case 6"

        self.grid = self.build_case6_grid()

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
            c_nf_per_km=1e-9,  # Dummy
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
            c_nf_per_km=1e-9,  # Dummy
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
            c_nf_per_km=1e-9,  # Dummy
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
            c_nf_per_km=1e-9,  # Dummy
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
            c_nf_per_km=1e-9,  # Dummy
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
            c_nf_per_km=1e-9,  # Dummy
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
            c_nf_per_km=1e-9,  # Dummy
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
            c_nf_per_km=1e-9,  # Dummy
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
            c_nf_per_km=1e-9,  # Dummy
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
            c_nf_per_km=1e-9,  # Dummy
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
            c_nf_per_km=1e-9,  # Dummy
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

        # Substations
        self.update_grid_with_substation_info(grid)

        return grid


class OPFRTECase5(UnitConverter, SubstationMixin):
    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=1e5)

        self.name = "Case RTE 5"

        env = grid2op.make(dataset="rte_case5_example")
        update_backend(env)

        self.grid = env.backend._grid

        # Substations
        self.update_grid_with_substation_info(self.grid)


class OPFL2RPN2019(UnitConverter, SubstationMixin):
    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=1e5)

        self.name = "Case L2RPN 2019"

        env = grid2op.make(dataset="l2rpn_2019")
        update_backend(env)

        self.grid = env.backend._grid

        # Substations
        self.update_grid_with_substation_info(self.grid)
