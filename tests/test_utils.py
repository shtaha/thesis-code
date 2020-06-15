import unittest

import numpy as np

from lib.data_utils import indices_to_hot, hot_to_indices


class TestDCOPF(unittest.TestCase):

    def test_indexing_1D(self):
        length = 10

        indices = np.array([3, 6, 8])
        print("input", indices)
        print("hot output", indices_to_hot(indices, length))
        print("indices output", hot_to_indices(indices_to_hot(indices, length)))

        self.assertTrue(np.equal(indices, hot_to_indices(indices_to_hot(indices, length))).all())

    # def test_indexing_2D(self):
    #     length = 10
    #
    #     indices = np.array([[3, 6, 8], [2, 4, 5]])
    #     print("input", indices)
    #     print("hot output", indices_to_hot(indices, length))
    #     print("indices output", hot_to_indices(indices_to_hot(indices, length)))
    #
    #     self.assertTrue(np.equal(indices, hot_to_indices(indices_to_hot(indices, length))).all())
