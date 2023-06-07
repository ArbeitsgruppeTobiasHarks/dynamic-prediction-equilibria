from typing import List, Iterable

from core.machine_precision import eps


def arg_min(list: Iterable, key=lambda x: x):
    minimum = None
    min_item = None
    for item in list:
        item_key = key(item)
        if minimum is None or item_key < minimum:
            minimum = item_key
            min_item = item
    return min_item


def elem_rank(arr: List[float], x: float) -> int:
    """
    Assume arr is increasing.
    Returns the rank of the element x in arr:
    The rank is the minimal number i in -1, ..., len(arr)-1,
    such that arr[i] < x <= arr[i+1] (with the interpretation arr[-1] = -inf and arr[len(arr)] = inf)
    """
    if x <= arr[0]:
        return -1
    low = 0
    high = len(arr)
    # Invariant: low - 1 <= rnk < high
    while high > low:
        mid = (high + low) // 2
        if x <= arr[mid]:
            high = mid
        else:
            low = mid + 1
    return high - 1


def elem_lrank(arr: List[float], x: float) -> int:
    """
    Assume arr to be strictly increasing.
    Returns the lower rank of the element x in arr:
    The lower rank is the minimal number i in -1, ..., len(arr)-1,
    such that arr[i] <= x < arr[i+1] (with the interpretation arr[-1] = -inf and arr[len(arr)] = inf)
    """
    if x < arr[0]:
        return -1
    low = 0
    high = len(arr)
    while high > low:
        mid = (high + low) // 2
        if x < arr[mid]:
            high = mid
        else:  # arr[mid] <= x
            low = mid + 1
    return high - 1


def merge_sorted_many(arrays: List[List[float]]) -> List[float]:
    """
    Merge multiple sorted arrays into a sorted array without duplicates (up to eps)
    """
    num_arrays = len(arrays)
    merged = []
    indices = [0 for _ in arrays]
    while True:  # any(indices[i] < len(arrays[i]) for i in range(num_arrays))
        i = min(
            (j for j in range(num_arrays) if indices[j] < len(arrays[j])),
            key=lambda j: arrays[j][indices[j]],
            default=None,
        )
        if i is None:
            break
        value_i = arrays[i][indices[i]]
        if len(merged) == 0 or merged[-1] < value_i - eps:
            merged.append(value_i)
        elif len(merged) > 0 and value_i - eps <= merged[-1] < value_i:
            merged[-1] = value_i
        indices[i] += 1
    return merged


def merge_sorted(arr1: List[float], arr2: List[float]) -> List[float]:
    """
    Merge two sorted arrays into a sorted array without duplicates (up to eps)
    """
    merged = []

    ind1 = 0
    ind2 = 0
    while ind1 < len(arr1) and ind2 < len(arr2):
        if arr1[ind1] < arr2[ind2]:
            if len(merged) == 0 or merged[-1] < arr1[ind1] - eps:
                merged.append(arr1[ind1])
            elif len(merged) > 0 and arr1[ind1] - eps <= merged[-1] < arr1[ind1]:
                merged[-1] = arr1[ind1]
            ind1 += 1
        else:
            if len(merged) == 0 or merged[-1] < arr2[ind2] - eps:
                merged.append(arr2[ind2])
            elif len(merged) > 0 and arr2[ind2] - eps <= merged[-1] < arr2[ind2]:
                merged[-1] = arr2[ind2]
            ind2 += 1

    while ind1 < len(arr1):
        if len(merged) == 0 or merged[-1] < arr1[ind1] - eps:
            merged.append(arr1[ind1])
        elif len(merged) > 0 and arr1[ind1] - eps <= merged[-1] < arr1[ind1]:
            merged[-1] = arr1[ind1]
        ind1 += 1
    while ind2 < len(arr2):
        if len(merged) == 0 or merged[-1] < arr2[ind2] - eps:
            merged.append(arr2[ind2])
        elif len(merged) > 0 and arr2[ind2] - eps <= merged[-1] < arr2[ind2]:
            merged[-1] = arr2[ind2]
        ind2 += 1

    return merged
