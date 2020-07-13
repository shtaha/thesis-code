import time
import unittest
import sys

import grid2op

from lib.visualizer import describe_environment, print_grid
from lib.dc_opf import OPFRTECase5, OPFL2RPN2019, OPFL2RPN2020


class TestGrid2op(unittest.TestCase):
    def test_rte_case5(self):
        try:
            env = grid2op.make(dataset="rte_case5_example")
            case = OPFRTECase5(env=env)

            grid = case.update_backend(env)

            describe_environment(env)
            print_grid(grid)

            time.sleep(0.1)
            self.assertTrue(True)
        except ModuleNotFoundError as e:
            print(e)
            time.sleep(0.1)
            self.assertTrue(False)

    def test_l2rpn2019(self):
        try:
            env = grid2op.make(dataset="l2rpn_2019")
            case = OPFL2RPN2019(env=env)

            grid = case.update_backend(env)

            describe_environment(env)
            print_grid(grid)

            time.sleep(0.1)
            self.assertTrue(True)
        except ModuleNotFoundError as e:
            print(e)
            time.sleep(0.1)
            self.assertTrue(False)

    def test_l2rpn2020(self):
        try:
            if sys.platform != "win32":
                print("L2RPN 2020 not available.")
                self.assertTrue(True)
                return

            env = grid2op.make(dataset="l2rpn_wcci_2020")
            case = OPFL2RPN2020(env=env)

            grid = case.update_backend(env)

            describe_environment(env)
            print_grid(grid)

            time.sleep(0.1)
            self.assertTrue(True)
        except ModuleNotFoundError as e:
            print(e)
            time.sleep(0.1)
            self.assertTrue(False)
