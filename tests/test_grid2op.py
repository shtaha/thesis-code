import sys
import time
import unittest

import grid2op

from lib.dc_opf import OPFRTECase5, OPFL2RPN2019, OPFL2RPN2020
from lib.visualizer import describe_environment, print_grid


class TestGrid2op(unittest.TestCase):
    def test_rte_case5(self):
        try:
            case = OPFRTECase5()
            env = case.env

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
            case = OPFL2RPN2019()
            env = case.env

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

            case = OPFL2RPN2020()
            env = case.env

            grid = case.update_backend(env)

            describe_environment(env)
            print_grid(grid)

            time.sleep(0.1)
            self.assertTrue(True)
        except ModuleNotFoundError as e:
            print(e)
            time.sleep(0.1)
            self.assertTrue(False)
