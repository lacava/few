# distutils: language = c++
from eigency.core cimport *
cimport numpy as np

cdef extern from "epsilon_lexicase.h":
     cdef void _epsilon_lexicase "epsilon_lexicase"(Map[ArrayXXd] & F, int n,
                                                    int d, int num_selections,
                                                    Map[ArrayXi] & locs)

# This will be exposed to Python
def ep_lex(np.ndarray F, int n, int d, int num_selections, np.ndarray locs):
    return _epsilon_lexicase(Map[ArrayXXd](F), n, d, num_selections,
                               Map[ArrayXi](locs))
