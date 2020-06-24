import time
import unittest

import grid2op

from lib.data_utils import update_backend
from lib.visualizer import describe_environment


class TestGrid2op(unittest.TestCase):
    def test_rte_case5(self):
        try:
            env = grid2op.make(dataset="rte_case5_example")
            update_backend(env)
            describe_environment(env)

            time.sleep(0.1)
            self.assertTrue(True)
        except ModuleNotFoundError as e:
            print(e)
            time.sleep(0.1)
            self.assertTrue(False)
