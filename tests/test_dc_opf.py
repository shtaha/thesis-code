import time
import unittest
import itertools
import grid2op
import numpy as np
import pandas as pd

from lib.data_utils import update_backend, indices_to_hot, hot_to_indices
from lib.dc_opf.models import StandardDCOPF, LineSwitchingDCOPF
from lib.dc_opf.cases import OPFCase3, OPFCase6


class TestDCOPF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        print("DC-OPF Tests.\n\n")

    def runner_opf(self, model, n_tests=20, eps=1e-4, verbose=False):
        conditions = list()
        for i in range(n_tests):
            np.random.seed(i)
            gen_cost = np.random.uniform(1.0, 5.0, (model.grid.gen.shape[0],))
            model.set_gen_cost(gen_cost)
            model.build_model()
            result = model.solve_and_compare(verbose=verbose)

            conditions.append(
                {
                    "cost": np.less_equal(result["res_cost"]["diff"], eps).all(),
                    "bus": np.less_equal(result["res_bus"]["diff"], eps).all(),
                    "line": np.less_equal(result["res_line"]["diff"], eps).all(),
                    "gen": np.less_equal(result["res_gen"]["diff"], eps).all(),
                }
            )

        conditions = pd.DataFrame(conditions)
        conditions["passed"] = np.all(conditions.values, axis=-1)

        print(f"\n\n{model.name}\n")
        print(conditions.to_string())

        time.sleep(0.1)
        # Test DC Power Flow
        self.assertTrue(conditions["passed"].values.all())

    """
    Test standard DC-OPF implementation.
    """

    def test_case6(self):
        case6 = OPFCase6()
        model_opf = StandardDCOPF(
            "CASE 6",
            case6.grid,
            base_unit_p=case6.base_unit_p,
            base_unit_v=case6.base_unit_v,
        )

        self.runner_opf(model_opf)

    def test_case3(self):
        case3 = OPFCase3()
        model_opf = StandardDCOPF(
            "CASE 3",
            case3.grid,
            base_unit_p=case3.base_unit_p,
            base_unit_v=case3.base_unit_v,
        )

        self.runner_opf(model_opf)

    def test_case3_by_value(self):
        """
        Test for power flow computation.
        """

        case3 = OPFCase3()
        model_opf = StandardDCOPF(
            "CASE 3 BY VALUE",
            case3.grid,
            base_unit_p=case3.base_unit_p,
            base_unit_v=case3.base_unit_v,
        )

        model_opf.set_gen_cost(np.array([1.0]))
        model_opf.build_model()

        result = model_opf.solve()
        model_opf.print_results()

        time.sleep(0.1)
        # Test DC Power Flow
        self.assertTrue(
            np.equal(
                result["res_bus"]["delta_pu"].values, np.array([0.0, -0.250, -0.375])
            ).all()
        )

    def test_rte_case5(self):
        env = grid2op.make(dataset="rte_case5_example")
        update_backend(env, True)
        model_opf = StandardDCOPF(
            "RTE CASE 5", env.backend._grid, base_unit_p=1e6, base_unit_v=1e5
        )

        self.runner_opf(model_opf, n_tests=5)

    """
        Test DC-OPF with line status switching implementation.
    """

    def runner_opf_line_switching(
        self, model, grid, n_line_status_changes, verbose=False
    ):
        np.random.seed(0)
        gen_cost = np.random.uniform(1.0, 5.0, (model.grid.gen.shape[0],))
        model.set_gen_cost(gen_cost)
        model.build_model()

        if verbose:
            model.print_model()

        # Construct all possible configurations
        line_statuses = list()
        for i in range(
            n_line_status_changes + 1
        ):  # Number of line disconnection 0, 1, ..., n
            line_statuses.extend(
                [
                    ~indices_to_hot(
                        list(line_status), length=grid.line.shape[0], dtype=np.bool,
                    )
                    for line_status in itertools.combinations(grid.line.index, i)
                ]
            )

        results_backend = pd.DataFrame(
            columns=["line_status", "objective", "loads_p", "generators_p", "valid"]
        )
        for line_status in line_statuses:
            model.grid.line["in_service"] = line_status
            result_backend = model.solve_backend()

            objective = (
                result_backend["res_cost"]
                + np.square(
                    result_backend["res_line"]["p_pu"] / model.line["max_p_pu"]
                ).sum()
            )
            loads_p = grid.load["p_pu"].sum()
            generators_p = result_backend["res_gen"]["p_pu"].sum()
            valid = generators_p > loads_p - 1e-6 and result_backend["valid"]

            results_backend = results_backend.append(
                {
                    "line_status": tuple(line_status),
                    "objective": objective,
                    "loads_p": loads_p,
                    "generators_p": generators_p,
                    "valid": valid,
                },
                ignore_index=True,
            )

        # Solve for optimal line status configuration
        result = model.solve(verbose=verbose)
        result_status = result["res_x"]
        result_objective = result["res_cost"]
        result_gap = result["res_gap"]  # Gap for finding the optimal configuration

        if verbose:
            model.print_results()

        # Check with brute force solution
        objective_brute = results_backend["objective"][results_backend["valid"]].min()
        hot_brute = (
            np.abs(results_backend["objective"].values - objective_brute) < result_gap
        )
        indices_brute = hot_to_indices(hot_brute)
        status_brute = results_backend["line_status"][indices_brute]

        match_idx = [
            idx
            for idx, line_status in zip(indices_brute, status_brute)
            if np.equal(line_status, result_status).all()
        ]

        # Compare
        results_backend["candidates"] = hot_brute
        results_backend["result_objective"] = np.nan
        results_backend["result_objective"][match_idx] = result_objective

        results_backend["line_status"] = [
            " ".join(np.array(line_status).astype(int).astype(str))
            for line_status in results_backend["line_status"]
        ]

        print(f"\n{model.name}\n")
        print(results_backend.to_string())

        time.sleep(0.1)
        self.assertTrue(bool(match_idx))

    def test_case3_line_switching(self):
        n_line_status_changes = 2

        case3 = OPFCase3()
        model_opf = LineSwitchingDCOPF(
            "CASE 3 Line Switching",
            case3.grid,
            n_line_status_changes=n_line_status_changes,
            base_unit_p=case3.base_unit_p,
            base_unit_v=case3.base_unit_v,
        )

        self.runner_opf_line_switching(
            model_opf, case3.grid, n_line_status_changes, verbose=True
        )

    def test_case6_line_switching(self):
        n_line_status_changes = 2

        case6 = OPFCase6()
        model_opf = LineSwitchingDCOPF(
            "CASE 6 Line Switching",
            case6.grid,
            n_line_status_changes=n_line_status_changes,
            base_unit_p=case6.base_unit_p,
            base_unit_v=case6.base_unit_v,
        )

        self.runner_opf_line_switching(
            model_opf, case6.grid, n_line_status_changes, verbose=True
        )

    def test_rte_case5_line_switching(self):
        n_line_status_changes = 3

        env = grid2op.make(dataset="rte_case5_example")
        update_backend(env)
        grid = env.backend._grid

        model_opf = LineSwitchingDCOPF(
            "RTE CASE 5 Line Switching",
            grid,
            n_line_status_changes=n_line_status_changes,
            base_unit_p=1e6,
            base_unit_v=1e5,
        )

        self.runner_opf_line_switching(
            model_opf, grid, n_line_status_changes, verbose=True
        )

    """
    Infeasible problem.
    """

    # def test_l2rpn2019_line_switching(self):
    #     n_line_status_changes = 2
    #
    #     env = grid2op.make(dataset="l2rpn_2019")
    #     update_backend(env)
    #     grid = env.backend._grid
    #
    #     model_opf = LineSwitchingDCOPF(
    #         "L2RPN 2019 Line Switching",
    #         grid,
    #         n_line_status_changes=n_line_status_changes,
    #         base_unit_p=1e6,
    #         base_unit_v=1e5,
    #     )
    #
    #     self.runner_opf_line_switching(
    #         model_opf, grid, n_line_status_changes, verbose=True
    #     )
    #
    # # TODO: Resolve.
    # def test_l2rpn2019(self):
    #     env = grid2op.make(dataset="l2rpn_2019")
    #     update_backend(env, True)
    #     model_opf = StandardDCOPF(
    #         "L2RPN 2019", env.backend._grid, base_unit_p=1e6, base_unit_v=1e5
    #     )
    #
    #     self.runner_opf(model_opf, n_tests=5, verbose=True)
