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


def elem_rank(arr: List[float], x: float) -> int:
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


def merge_sorted(arr1: List[float], arr2: List[float]) -> List[float]:
    """
    Merge two sorted arrays into a sorted array without duplicates
    """
    merged = []

    ind1 = 0
    ind2 = 0
    while ind1 < len(arr1) and ind2 < len(arr2):
        if arr1[ind1] < arr2[ind2]:
            if len(merged) == 0 or merged[-1] < arr1[ind1]:
                merged.append(arr1[ind1])
            ind1 += 1
        else:
            if len(merged) == 0 or merged[-1] < arr2[ind2]:
                merged.append(arr2[ind2])
            ind2 += 1

    while ind1 < len(arr1):
        if len(merged) == 0 or merged[-1] < arr1[ind1]:
            merged.append(arr1[ind1])
        ind1 += 1
    while ind2 < len(arr2):
        if len(merged) == 0 or merged[-1] < arr2[ind2]:
            merged.append(arr2[ind2])
        ind2 += 1

    return merged

