
from cython.cimports.cpython import array

cdef Py_ssize_t elem_rank(array.array arr, double x)

cdef Py_ssize_t elem_lrank(array.array arr, double x)

cdef array.array merge_sorted_many(list arrays)

cdef array.array merge_sorted(array.array arr1, array.array arr2)

