import time
import unittest

import numpy as np
import grid2op
import pandas as pd

from lib.data_utils import update_backend
from lib.dc_opf import OPFCase3, OPFCase6, StandardDCOPF


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

            conditions.append({
                "cost": np.less_equal(result["res_cost"]["diff"], eps).all(),
                "bus": np.less_equal(result["res_bus"]["diff"], eps).all(),
                "line": np.less_equal(result["res_line"]["diff"], eps).all(),
                "gen": np.less_equal(result["res_gen"]["diff"], eps).all()
            })

        conditions = pd.DataFrame(conditions)
        conditions["passed"] = np.all(conditions.values, axis=-1)

        print(f"\n\n{model.name}\n")
        print(conditions.to_string())

        time.sleep(0.1)
        # Test DC Power Flow
        self.assertTrue(conditions["passed"].values.all())

    def test_case6(self):
        case6 = OPFCase6()
        model_opf = StandardDCOPF(
            "CASE 6", case6.grid, base_unit_p=case6.base_unit_p, base_unit_v=case6.base_unit_v
        )

        self.runner_opf(model_opf)

    def test_case3(self):
        case3 = OPFCase3()
        model_opf = StandardDCOPF(
            "CASE 3", case3.grid, base_unit_p=case3.base_unit_p, base_unit_v=case3.base_unit_v
        )

        self.runner_opf(model_opf)

    def test_case3_by_value(self):
        """
        Test for power flow computation.
        """

        case3 = OPFCase3()
        model_opf = StandardDCOPF(
            "CASE 3 BY VALUE", case3.grid, base_unit_p=case3.base_unit_p, base_unit_v=case3.base_unit_v
        )

        model_opf.set_gen_cost(np.array([1.0]))
        model_opf.build_model()

        result = model_opf.solve()
        model_opf.print_results()

        time.sleep(0.1)
        # Test DC Power Flow
        self.assertTrue(np.equal(result["res_bus"]["delta_pu"].values, np.array([0.0, -0.250, -0.375])).all())

    def test_rte_case5(self):
        env = grid2op.make(dataset="rte_case5_example")
        update_backend(env)
        model_opf = StandardDCOPF("RTE CASE 5", env.backend._grid, base_unit_p=1e6, base_unit_v=1e5)

        self.runner_opf(model_opf, n_tests=5)
