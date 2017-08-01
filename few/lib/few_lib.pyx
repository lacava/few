# distutils: language=c++
from eigency.core cimport *
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "epsilon_lexicase.h":
     cdef void _epsilon_lexicase "epsilon_lexicase"(Map[ArrayXXd] & F, int n,
                                                    int d, int num_selections,
                                                    Map[ArrayXi] & locs, bool lex_size, Map[ArrayXi] &sizes)

# This will be exposed to Python
def ep_lex(np.ndarray F, int n, int d, int num_selections, np.ndarray locs, bool lex_size, np.ndarray sizes):
    return _epsilon_lexicase(Map[ArrayXXd](F), n, d, num_selections,
                               Map[ArrayXi](locs), lex_size, Map[ArrayXi](sizes))
# WIP
# cdef extern from "evaluation.h":
#     cdef void _evaluate "evaluate"(node n, Map[ArrayXXd] & features,
#                                    vector[Map[ArrayXd]]] stack_float,
#                                    vector[Map[ArrayXb]] stack_bool)

# def evaluate(node n, np.ndarray features, vector[np.ndarray] stack_float,
#              vector[np.ndarray] stack_bool):
#     return _evaluate(node n, Map[ArrayXXd](features),
#                      vector[Map[ArrayXd]]](stack_float),
#                      vector[Map[ArrayXb]](stack_bool))
