import time
import unittest

import numpy as np
import pandapower as pp


class TestDCOPF(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        print("testing")

    def test_dc_opf_pp(self):
        """
        Test case from http://research.iaun.ac.ir/pd/bahador.fani/pdfs/UploadFile_6990.pdf.
        """
        grid = pp.create_empty_network()

        # create buses
        v_n = 110000.0  # Nominal voltage unit in V
        s_n = 1e6  # Nominal power unit in W
        z_n = v_n ** 2 / s_n  # Nominal impedance unit in Ohms
        i_n = s_n / v_n  # Nominal current unit in Amperes
        print(f"s_n = {s_n} W")
        print(f"v_n = {v_n} V")
        print(f"i_n = {i_n} A")
        print(f"z_n = {z_n} Ohm")

        bus1 = pp.create_bus(grid, vn_kv=v_n / 1000, name="bus-0")
        bus2 = pp.create_bus(grid, vn_kv=v_n / 1000, name="bus-1")
        bus3 = pp.create_bus(grid, vn_kv=v_n / 1000, name="bus-2")

        pp.create_line_from_parameters(grid, bus1, bus2,
                                       length_km=1.0,
                                       r_ohm_per_km=0.01 * z_n,
                                       x_ohm_per_km=1.0 / 3.0 * z_n,
                                       c_nf_per_km=0.0001,
                                       max_i_ka=i_n / 1000,
                                       name="line-0",
                                       type="ol",
                                       max_loading_percent=100.0)

        pp.create_line_from_parameters(grid, bus1, bus3,
                                       length_km=1.0,
                                       r_ohm_per_km=0.01 * z_n,
                                       x_ohm_per_km=1.0 / 2.0 * z_n,
                                       c_nf_per_km=0.0001,
                                       max_i_ka=i_n / 1000,
                                       name="line-1",
                                       type="ol",
                                       max_loading_percent=100.0)

        pp.create_line_from_parameters(grid, bus2, bus3,
                                       length_km=1.0,
                                       r_ohm_per_km=0.01 * z_n,
                                       x_ohm_per_km=1.0 / 2.0 * z_n,
                                       c_nf_per_km=0.0001,
                                       max_i_ka=i_n / 1000,
                                       name="line-2",
                                       type="ol",
                                       max_loading_percent=100.0)

        pp.create_load(grid, bus2, p_mw=0.5, name="load-0", controllable=False)
        pp.create_load(grid, bus3, p_mw=1.0, name="load-1", controllable=False)
        pp.create_gen(grid, bus1, p_mw=1.5, min_p_mw=0, max_p_mw=2.0, slack=True, name="gen-0")

        pp.rundcpp(grid, verbose=True)

        # Per unit conversions
        # Buses
        grid.bus["vn_pu"] = grid.bus["vn_kv"] * 1000 / v_n

        # Power lines
        grid.line["x_pu"] = grid.line["x_ohm_per_km"] * grid.line["length_km"] / z_n / grid.line["parallel"]
        grid.line["r_pu"] = grid.line["r_ohm_per_km"] * grid.line["length_km"] / z_n / grid.line["parallel"]
        grid.line["b_pu"] = 1 / grid.line["x_pu"]
        grid.line["max_i_pu"] = grid.line["max_i_ka"] * 1000 / i_n
        grid.line["max_p_pu"] = grid.line["max_i_pu"] * grid.bus["vn_pu"][grid.line["from_bus"].values].values

        # Generators
        grid.gen["p_pu"] = grid.gen["p_mw"]
        grid.gen["max_p_pu"] = grid.gen["max_p_mw"]
        grid.gen["min_p_pu"] = grid.gen["min_p_mw"]

        # Loads
        grid.load["p_pu"] = grid.load["p_mw"]

        # Results
        grid.res_bus["va_pu"] = grid.res_bus["va_degree"] * np.pi / 180.0
        grid.res_line["p_pu"] = grid.res_line["p_from_mw"]
        grid.res_line["i_pu"] = grid.res_line["i_from_ka"] / i_n
        grid.res_gen["p_pu"] = grid.res_gen["p_mw"]

        print("BUS\n" + grid.bus[["name", "vn_pu"]].to_string())
        print("LINE\n" + grid.line[
            ["name", "from_bus", "to_bus", "b_pu", "max_i_pu", "max_p_pu", "max_loading_percent"]].to_string())
        print("GEN\n" + grid.gen[["name", "bus", "p_pu", "min_p_pu", "max_p_pu"]].to_string())
        print("LOAD\n" + grid.load[["name", "bus", "p_pu"]].to_string())
        print("RES BUS\n" + grid.res_bus[["va_pu"]].to_string())
        print("RES LINE\n" + grid.res_line[["p_pu", "i_pu", "loading_percent"]].to_string())
        print("RES GEN\n" + grid.res_gen[["p_pu"]].to_string())

        time.sleep(0.1)
        # Test DC Power Flow
        self.assertTrue(np.equal(grid.res_bus["va_pu"].values, np.array([0.0, -0.250, -0.375])).all())
