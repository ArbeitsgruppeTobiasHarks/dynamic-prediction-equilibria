from utilities.arrays import elem_lrank, elem_rank


def test_urank():
    arr = [0.0, 1.0, 2.0]
    assert elem_rank(arr, -0.5) == -1
    for i, val in enumerate(arr):
        assert elem_rank(arr, val) == i - 1
        assert elem_rank(arr, val + 0.5) == i


def test_lrank():
    arr = [0.0, 1.0, 2.0]
    assert elem_lrank(arr, -0.5) == -1
    for i, val in enumerate(arr):
        assert elem_lrank(arr, val) == i
        assert elem_lrank(arr, val + 0.5) == i
