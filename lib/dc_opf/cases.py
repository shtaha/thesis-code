from abc import ABC, abstractmethod

import grid2op
import numpy as np
import pandapower as pp
import pandas as pd

from .unit_converter import UnitConverter
from lib.visualizer import describe_environment, print_parameters


def load_case(case_name, env=None, env_dc=False, verbose=False):
    if env_dc:
        env.env_dc = env_dc
        env.parameters.ENV_DC = env_dc
        env.parameters.FORECAST_DC = env_dc

    if case_name == "case3":
        case = OPFCase3()
    elif case_name == "case6":
        case = OPFCase6()
    elif case_name == "case4":
        case = OPFCase4()
    elif case_name in ["rte_case5", "rte_case5_example"]:
        case = OPFRTECase5(env=env)
    elif case_name in ["l2rpn2019", "l2rpn_2019"]:
        case = OPFL2RPN2019(env=env)
    elif case_name in ["l2rpn2020", "l2rpn_wcci_2020", "l2rpn_2020"]:
        case = OPFL2RPN2020(env=env)
    else:
        raise ValueError(f"Invalid case name. Case {case_name} does not exist.")

    if verbose and case.env:
        describe_environment(env)
        print_parameters(env)

    return case


class OPFAbstractCase(ABC):
    @abstractmethod
    def build_case_grid(self):
        pass


class OPFCaseMixin:
    @staticmethod
    def _update_backend(env, grid):
        n_line = len(grid.line.index)

        # Grid element names from environment names
        grid.line["name"] = env.name_line[0:n_line]
        grid.gen["name"] = env.name_gen
        grid.load["name"] = env.name_load
        grid.trafo["name"] = env.name_line[n_line:]

        # Update thermal limits with environment thermal limits
        grid.line["max_i_ka"] = env.get_thermal_limit()[0:n_line] / 1000.0

        # Environment and backend inconsistency
        grid.gen["min_p_mw"] = env.gen_pmin
        grid.gen["max_p_mw"] = env.gen_pmax
        grid.gen["type"] = env.gen_type
        grid.gen["gen_redispatchable"] = env.gen_redispatchable
        grid.gen["gen_max_ramp_up"] = env.gen_max_ramp_up
        grid.gen["gen_max_ramp_down"] = env.gen_max_ramp_down
        grid.gen["gen_min_uptime"] = env.gen_min_uptime
        grid.gen["gen_min_downtime"] = env.gen_min_downtime


class OPFCase3(OPFAbstractCase, UnitConverter):
    """
    Test case for power flow computation.
    Found in http://research.iaun.ac.ir/pd/bahador.fani/pdfs/UploadFile_6990.pdf.
    """

    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=110000.0)

        self.name = "Case 3"

        self.env = None
        self.grid_org = self.build_case_grid()
        self.grid_backend = self.grid_org.deepcopy()

    def build_case_grid(self):
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


class OPFCase6(OPFAbstractCase, UnitConverter):
    """
    Test case for power flow computation.
    Found in http://research.iaun.ac.ir/pd/bahador.fani/pdfs/UploadFile_6990.pdf.
    """

    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=110000.0)

        self.name = "Case 6"

        self.env = None
        self.grid_org = self.build_case_grid()
        self.grid_backend = self.grid_org.deepcopy()

    def build_case_grid(self):
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


class OPFCase4(OPFAbstractCase, UnitConverter):
    """
    Test case for power flow computation, including an external grid and a transformer.
    Found in https://github.com/e2nIEE/pandapower/blob/develop/tutorials/opf_curtail.ipynb.
    """

    def __init__(self):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=110000.0)

        self.name = "Case 4"

        self.env = None
        self.grid_org = self.build_case_grid()
        self.grid_backend = self.grid_org.deepcopy()

    def build_case_grid(self):
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


class OPFRTECase5(OPFAbstractCase, UnitConverter, OPFCaseMixin):
    def __init__(self, env=None):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=1e5)

        self.name = "Case RTE 5"

        if not env:
            self.env = grid2op.make(dataset="rte_case5_example")
        else:
            self.env = env

        self.grid_org = self.build_case_grid()
        self.grid_backend = self.update_backend(self.env)
        self.env.backend._grid = self.grid_backend

    def build_case_grid(self):
        return self.env.backend._grid

    def update_backend(self, env):
        """
        Update backend grid with missing data.
        """
        grid = env.backend._grid.deepcopy()

        # Check if even number of buses
        assert len(grid.bus.index) % 2 == 0

        # Bus names
        bus_names = []
        for bus_id, bus_name in zip(grid.bus.index, grid.bus["name"]):
            sub_id = bus_name.split("_")[-1]
            bus_names.append(f"bus-{bus_id}-{sub_id}")
        grid.bus["name"] = bus_names

        # Controllable injections
        grid.load["controllable"] = False
        grid.gen["controllable"] = True

        self._update_backend(env, grid)

        return grid


class OPFL2RPN2019(OPFAbstractCase, UnitConverter, OPFCaseMixin):
    def __init__(self, env=None):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=1e5)

        self.name = "Case L2RPN 2019 - IEEE 14"

        if not env:
            self.env = grid2op.make(dataset="l2rpn_2019")
        else:
            self.env = env

        self.grid_org = self.build_case_grid()
        self.grid_backend = self.update_backend(self.env)
        self.env.backend._grid = self.grid_backend

    def build_case_grid(self):
        return self.env.backend._grid

    def update_backend(self, env):
        """
        Update backend grid with missing data.
        """
        grid = env.backend._grid.deepcopy()

        # Check if even number of buses
        assert len(grid.bus.index) % 2 == 0

        # Bus names
        bus_names = [
            f"bus-{bus_id}-{sub_id}"
            for bus_id, sub_id in zip(grid.bus.index, grid.bus["name"])
        ]
        grid.bus["name"] = bus_names

        self._update_backend(env, grid)

        return grid


class OPFL2RPN2020(OPFAbstractCase, UnitConverter, OPFCaseMixin):
    def __init__(self, env=None):
        UnitConverter.__init__(self, base_unit_p=1e6, base_unit_v=138000.0)

        self.name = "Case L2RPN 2020 WCCI - IEEE 118"

        if not env:
            self.env = grid2op.make(dataset="l2rpn_wcci_2020")
        else:
            self.env = env

        self.grid_org = self.build_case_grid()
        self.grid_backend = self.update_backend(self.env)
        self.env.backend._grid = self.grid_backend

    def build_case_grid(self):
        return self.env.backend._grid

    def update_backend(self, env):
        """
        Update backend grid with missing data.
        """
        grid = env.backend._grid.deepcopy()

        # Bus names
        n_sub = len(grid.bus.index) // 2
        bus_to_sub_ids = np.concatenate((np.arange(0, n_sub), np.arange(0, n_sub)))
        bus_names = [
            f"bus-{bus_id}-{sub_id}"
            for bus_id, sub_id in zip(grid.bus.index, bus_to_sub_ids)
        ]
        grid.bus["name"] = bus_names

        self._update_backend(env, grid)

        # Manually set
        trafo_params = {
            "id": {"0": 0, "1": 1, "2": 2, "3": 3,},
            "b_pu": {
                "0": 2852.04991087,
                "1": 2698.61830743,
                "2": 3788.16577013,
                "3": 2890.59112589,
            },
            "max_p_pu": {"0": 9900.0, "1": 9900.0, "2": 9900.0, "3": 9900.0},
        }

        trafo_params = pd.DataFrame.from_dict(trafo_params)
        trafo_params.set_index("id", inplace=True)
        grid.trafo["b_pu"] = trafo_params["b_pu"]
        grid.trafo["max_p_pu"] = trafo_params["max_p_pu"]

        return grid
