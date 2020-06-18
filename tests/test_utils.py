import time
import unittest

import numpy as np

from lib.data_utils import indices_to_hot, hot_to_indices, bus_names_to_sub_ids


class TestDataUtils(unittest.TestCase):
    def test_indexing_1D(self):
        length = 10

        indices = np.array([3, 6, 8])
        print("input", indices)
        print("hot output", indices_to_hot(indices, length))
        print("indices output", hot_to_indices(indices_to_hot(indices, length)))

        time.sleep(0.1)
        self.assertTrue(
            np.equal(indices, hot_to_indices(indices_to_hot(indices, length))).all()
        )

    def test_bus_names_to_sub_ids(self):
        bus_names = ["bus-0-0", "bus-1-1", "bus-2-2", "bus-2-3", "bus-2-1"]
        sub_ids = [0, 1, 2, 3, 1]

        print(bus_names)
        print(sub_ids)

        self.assertTrue(np.equal(sub_ids, bus_names_to_sub_ids(bus_names)).all())

    # def test_indexing_2D(self):
    #     length = 10
    #
    #     indices = np.array([[3, 6, 8], [2, 4, 5]])
    #     print("input", indices)
    #     print("hot output", indices_to_hot(indices, length))
    #     print("indices output", hot_to_indices(indices_to_hot(indices, length)))
    #
    #     self.assertTrue(np.equal(indices, hot_to_indices(indices_to_hot(indices, length))).all())
