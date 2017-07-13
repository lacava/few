# distutils: language=c++
from eigency.core cimport *
cimport numpy as np
from libcpp.vector cimport vector

cdef extern from "epsilon_lexicase.h":
     cdef void _epsilon_lexicase "epsilon_lexicase"(Map[ArrayXXd] & F, int n,
                                                    int d, int num_selections,
                                                    Map[ArrayXi] & locs)

# This will be exposed to Python
def ep_lex(np.ndarray F, int n, int d, int num_selections, np.ndarray locs):
    return _epsilon_lexicase(Map[ArrayXXd](F), n, d, num_selections,
                               Map[ArrayXi](locs))
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
