import unittest

from utilities.arrays import elem_rank


class TestRank(unittest.TestCase):
    def test_urank(self):
        arr = [0.0, 1.0, 2.0]
        self.assertEqual(elem_rank(arr, -0.5), -1)
        for i, val in enumerate(arr):
            self.assertEqual(elem_rank(arr, val), i - 1)
            self.assertEqual(elem_rank(arr, val + 0.5), i)

    def test_lrank(self):
        arr = [0.0, 1.0, 2.0]
        self.assertEqual(elem_rank(arr, -0.5), -1)
        for i, val in enumerate(arr):
            self.assertEqual(elem_rank(arr, val), i)
            self.assertEqual(elem_rank(arr, val + 0.5), i)
