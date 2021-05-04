from libcpp.vector cimport vector

cdef cppclass Edge:
    cdef int node_from, node_to, id

    cdef __cinit__(self, int node_from, int node_to, int id):
        self.node_from = node_from
        self.node_to = node_to
        self.id = id

cdef class Node:
    cdef public int id
    cdef public vector[Edge] incoming_edges
    cdef public vector[Edge] outgoing_edges

cdef class Network:
    cdef public map[int, Edge] edges
    cdef public map[int, Node] nodes

    def __init__(self):
        pass

    def add_edge(self):
        pass
