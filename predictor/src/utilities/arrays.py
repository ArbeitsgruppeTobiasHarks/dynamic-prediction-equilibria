from typing import List, Iterable


def arg_min(list: Iterable, key=lambda x: x):
    minimum = None
    min_item = None
    for item in list:
        item_key = key(item)
        if minimum is None or item_key < minimum:
            minimum = item_key
            min_item = item
    return min_item


def elem_rank(arr: List[float], x: float):
    """
    Assume arr to be strictly increasing.
    Returns the rank of the element x in arr:
    The rank is the minimal number i in -1, ..., len(arr)-1,
    such that arr[i] < x <= arr[i+1] (with the interpretation arr[-1] = -inf and arr[len(arr)] = inf)
    """
    if x <= arr[0]:
        return -1
    low = 0
    high = len(arr)
    while high > low:
        mid = (high + low) // 2
        if x <= arr[mid]:
            high = mid
        else:
            low = mid + 1
    return high - 1
