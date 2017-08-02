# distutils: language=c++
from eigency.core cimport *
cimport numpy as np
from libcpp.vector cimport vector
from libcpp cimport bool

cdef extern from "epsilon_lexicase.h":
     cdef void _epsilon_lexicase "epsilon_lexicase"(Map[ArrayXXd] & F, int n,
                                                    int d, int num_selections,
                                                    Map[ArrayXi] & locs, bool lex_size,
                                                    Map[ArrayXi] &sizes)

# This will be exposed to Python
def ep_lex(np.ndarray F, int n, int d, int num_selections, np.ndarray locs, bool lex_size,
           np.ndarray sizes):
    return _epsilon_lexicase(Map[ArrayXXd](F), n, d, num_selections,
                               Map[ArrayXi](locs), lex_size, Map[ArrayXi](sizes))
