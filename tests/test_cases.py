import sys
import time
import unittest

import numpy as np
import pandas as pd

from lib.dc_opf import (
    StandardDCOPF,
    GridDCOPF,
    load_case,
)


class TestCasesDCOPF(unittest.TestCase):
    """
    Test per unit DC-OPF grid parameters.
    """

    @classmethod
    def setUpClass(cls):
        print("\nTest per unit DC-OPF grid parameters.\n")

    def runner(self, model, eps=1e-4, verbose=False):
        np.random.seed(0)
        model.grid.gen["cost_pu"] = np.random.uniform(0.5, 5.0, model.grid.gen.shape[0])

        model.build_model()
        model.solve_backend()

        """
            Power lines - Susceptances and thermal limits
        """
        bus_or = model.grid_backend.line["from_bus"].values
        bus_ex = model.grid_backend.line["to_bus"].values

        delta_or = model.convert_degree_to_rad(
            model.grid_backend.res_bus["va_degree"][bus_or]
        ).values
        delta_ex = model.convert_degree_to_rad(
            model.grid_backend.res_bus["va_degree"][bus_ex]
        ).values
        p_pu = model.convert_mw_to_per_unit(
            model.grid_backend.res_line["p_from_mw"]
        ).values
        max_p_pu = np.abs(p_pu) / (
            model.grid_backend.res_line["loading_percent"].values / 100.0
        )

        diff_line = pd.DataFrame()
        diff_line["b_pu"] = model.grid.line["b_pu"][~model.grid.line.trafo]
        diff_line["b_b_pu"] = p_pu / (delta_or - delta_ex)
        diff_line["diff_b_pu"] = np.abs(diff_line["b_pu"] - diff_line["b_b_pu"])
        diff_line["max_p_pu"] = model.grid.line["max_p_pu"][~model.grid.line.trafo]
        diff_line["b_max_p_pu"] = max_p_pu
        diff_line["diff_max_p_pu"] = np.abs(
            diff_line["max_p_pu"] - diff_line["b_max_p_pu"]
        )

        """
            Transformer - Susceptances and thermal limits.
        """
        bus_or = model.grid_backend.trafo["hv_bus"].values
        bus_ex = model.grid_backend.trafo["lv_bus"].values

        delta_or = model.convert_degree_to_rad(
            model.grid_backend.res_bus["va_degree"][bus_or]
        ).values
        delta_ex = model.convert_degree_to_rad(
            model.grid_backend.res_bus["va_degree"][bus_ex]
        ).values
        p_pu = model.convert_mw_to_per_unit(
            model.grid_backend.res_trafo["p_hv_mw"]
        ).values
        max_p_pu = np.abs(p_pu) / np.abs(
            model.grid_backend.res_trafo["loading_percent"].values / 100.0
        )

        diff_trafo = pd.DataFrame()
        diff_trafo["b_pu"] = model.grid.trafo["b_pu"]
        diff_trafo["b_b_pu"] = p_pu / (delta_or - delta_ex)
        diff_trafo["diff_b_pu"] = np.abs(diff_trafo["b_pu"] - diff_trafo["b_b_pu"])
        diff_trafo["max_p_pu"] = model.grid.trafo["max_p_pu"]
        diff_trafo["b_max_p_pu"] = max_p_pu
        diff_trafo["diff_max_p_pu"] = np.abs(
            diff_trafo["max_p_pu"] - diff_trafo["b_max_p_pu"]
        )

        if verbose:
            print(diff_line.to_string())
            if len(diff_trafo.index):
                print(diff_trafo.to_string())

        conds_lines = np.logical_and(
            diff_line["diff_b_pu"] < eps, diff_line["diff_max_p_pu"] < eps
        )
        conds_trafo = np.logical_and(
            diff_trafo["diff_b_pu"] < eps, diff_trafo["diff_max_p_pu"] < eps
        )

        time.sleep(0.1)
        self.assertTrue(conds_lines.all() and conds_trafo.all())

    def test_case3(self):
        case = load_case("case3")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
        )

        self.runner(model, verbose=False)

    def test_case4(self):
        case = load_case("case4")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
        )

        self.runner(model, verbose=False)

    def test_case6(self):
        case = load_case("case3")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
        )

        self.runner(model, verbose=False)

    def test_rte_case5(self):
        case = load_case("rte_case5")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
        )

        self.runner(model, eps=5e-3, verbose=True)

    def test_l2rpn2019(self):
        case = load_case("l2rpn2019")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
        )

        self.runner(model, eps=1e-2, verbose=False)

    def test_l2rpn2020(self):
        if sys.platform != "win32":
            print("L2RPN 2020 not available.")
            self.assertTrue(True)
            return

        case = load_case("l2rpn2020")
        grid = GridDCOPF(
            case, base_unit_v=case.base_unit_v, base_unit_p=case.base_unit_p
        )

        model = StandardDCOPF(
            f"{case.name} Standard DC OPF",
            grid=grid,
            grid_backend=case.grid_backend,
            base_unit_p=case.base_unit_p,
            base_unit_v=case.base_unit_v,
        )

        self.runner(model, eps=1e-3, verbose=False)
