
from typing import Iterable, List
import cython


from cython.cimports.cpython import array
import array

cdef double eps = 1e-10

def arg_min(list: Iterable, key=lambda x: x):
    minimum = None
    min_item = None
    for item in list:
        item_key = key(item)
        if minimum is None or item_key < minimum:
            minimum = item_key
            min_item = item
    return min_item

cdef Py_ssize_t elem_rank(array.array arr, double x):
    """
    Assume arr is increasing.
    Returns the rank of the element x in arr:
    The rank is the minimal number i in -1, ..., len(arr)-1,
    such that arr[i] < x <= arr[i+1] (with the interpretation arr[-1] = -inf and arr[len(arr)] = inf)
    """
    size: cython.Py_ssize_t = len(arr)
    if size == 0 or x <= arr.data.as_doubles[0]:
        return -1
    low: cython.Py_ssize_t = 0
    high: cython.Py_ssize_t = size
    # Invariant: low - 1 <= rnk < high
    while high > low:
        mid: cython.Py_ssize_t = (high + low) // 2
        if x <= arr.data.as_doubles[mid]:
            high = mid
        else:
            low = mid + 1
    return high - 1

cdef Py_ssize_t elem_lrank(array.array arr, double x):
    """
    Assume arr to be strictly increasing.
    Returns the lower rank of the element x in arr:
    The lower rank is the minimal number i in -1, ..., len(arr)-1,
    such that arr[i] <= x < arr[i+1] (with the interpretation arr[-1] = -inf and arr[len(arr)] = inf)
    """
    size: cython.Py_ssize_t = len(arr)
    if size == 0 or x < arr.data.as_doubles[0]:
        return -1
    low: cython.Py_ssize_t = 0
    high: cython.Py_ssize_t = size
    while high > low:
        mid: cython.Py_ssize_t = (high + low) // 2
        if x < arr.data.as_doubles[mid]:
            high = mid
        else:  # arr[mid] <= x
            low = mid + 1
    return high - 1


cdef array.array merge_sorted_many(list arrays):
    """
    Merge multiple sorted arrays into a sorted array without duplicates (up to eps)
    """
    num_arrays: cython.Py_ssize_t = len(arrays)

    merged: array.array = array.clone(arrays[0], length=sum(len(arr) for arr in arrays), zero=False)
    merged_size: cython.Py_ssize_t = 0

    indices_template: array.array = array.array('l', [])
    indices: array.array = array.clone(indices_template, num_arrays, zero=True)

    while True:  # any(indices[i] < len(arrays[i]) for i in range(num_arrays))
        i: cython.Py_ssize_t = min(
            (j for j in range(num_arrays) if indices[j] < len(arrays[j])),
            key=lambda j: arrays[j][indices[j]],
            default=-1,
        )
        if i == -1:
            break
        cur_array: array.array = arrays[i]
        value_i: cython.double = cur_array.data.as_doubles[indices.data.as_longs[i]]
        if merged_size == 0 or merged.data.as_doubles[merged_size - 1] < value_i - eps:
            merged.data.as_doubles[merged_size] = value_i
            merged_size += 1
        elif merged_size > 0 and value_i - eps <= merged.data.as_doubles[merged_size - 1] < value_i:
            merged.data.as_doubles[merged_size - 1] = value_i
        indices.data.as_longs[i] += 1
    
    array.resize_smart(merged, merged_size)
    return merged

cdef array.array merge_sorted(array.array arr1, array.array arr2):
    """
    Merge two sorted arrays into a sorted array without duplicates (up to eps)
    """
    merged = array.clone(arr1, length=arr1.ob_size + arr2.ob_size, zero=False)
    merged_size: cython.Py_ssize_t = 0

    ind1: cython.Py_ssize_t = 0
    ind2: cython.Py_ssize_t = 0

    cdef double* arr1_p = arr1.data.as_doubles 
    cdef double* arr2_p = arr2.data.as_doubles
    cdef double* merged_p = merged.data.as_doubles
    while ind1 < arr1.ob_size and ind2 < arr2.ob_size:
        if arr1_p[ind1] < arr2_p[ind2]:
            if merged_size == 0 or merged_p[merged_size - 1] < arr1_p[ind1] - eps:
                merged_p[merged_size] = arr1_p[ind1]
                merged_size += 1
            elif merged_size > 0 and arr1_p[ind1] - eps <= merged_p[merged_size-1] < arr1_p[ind1]:
                merged_p[merged_size - 1] = arr1_p[ind1]
            ind1 += 1
        else:
            if merged_size == 0 or merged_p[merged_size - 1] < arr2_p[ind2] - eps:
                merged_p[merged_size] = arr2_p[ind2]
                merged_size += 1
            elif merged_size > 0 and arr2_p[ind2] - eps <= merged_p[merged_size-1] < arr2_p[ind2]:
                merged_p[merged_size-1] = arr2_p[ind2]
            ind2 += 1

    while ind1 < arr1.ob_size:
        if merged_size == 0 or merged_p[merged_size-1] < arr1_p[ind1] - eps:
            merged_p[merged_size] = arr1_p[ind1]
            merged_size += 1
        elif merged_size > 0 and arr1_p[ind1] - eps <= merged_p[merged_size - 1] < arr1_p[ind1]:
            merged_p[merged_size - 1] = arr1_p[ind1]
        ind1 += 1
    while ind2 < arr2.ob_size:
        if merged_size == 0 or merged_p[merged_size - 1] < arr2_p[ind2] - eps:
            merged_p[merged_size] = arr2_p[ind2]
            merged_size += 1
        elif merged_size > 0 and arr2_p[ind2] - eps <= merged_p[merged_size - 1] < arr2_p[ind2]:
            merged_p[merged_size - 1] = arr2_p[ind2]
        ind2 += 1

    array.resize_smart(merged, merged_size) 
    
    return merged
