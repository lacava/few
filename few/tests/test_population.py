"""
Copyright 2016 William La Cava

This file is part of the FEW library.

The FEW library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The FEW library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
the FEW library. If not, see http://www.gnu.org/licenses/.

"""
import numpy as np

from few.population import *
from itertools import accumulate
# unit tests for population methods.
def test_pop_shape():
    """test_population.py: population class returns correct sizes """
    # pop = Pop(0)
    # assert len(pop) == 0
    pop = Pop(10)
    assert len(pop.individuals) == 10
    print("pop.X.shape",pop.X.shape)
    assert pop.X.shape == (1,10)
    assert pop.E.shape == (0,)


    pop = Pop(73)
    assert len(pop.individuals) == 73
    assert pop.X.shape == (1,73)
    assert pop.E.shape == (0,)

    pop = Pop(73,5)
    assert len(pop.individuals) == 73
    assert pop.X.shape == (5,73)
    assert pop.E.shape == (0,)

def test_pop_init():
    """test_population.py: population initialization makes valid trees """
    # define function set
    # function set
    func_set = [('+',2),('-',2),('*',2),('/',2),('sin',1),('cos',1),('exp',1),('log',1)]
    # terminal set
    term_set = []
    n_features = 3
    # numbers represent column indices of features
    for i in np.arange(n_features):
        term_set.append(('x',0,i)) # features
        # term_set.append(('erc',0,np.random.rand())) # ephemeral random constants

    pop = init(10,500,func_set,term_set,1,5)
    for I in pop.individuals:
        assert is_valid_program(I.stack)

def is_valid_program(p):
    """ checks that the accumulated program length is always greater than the
    accumulated arities, indicating that the appropriate number of arguments is
    alway present for functions. It then checks that the sum of arties +1
    exactly equals the length of the stack, indicating that there are no
    missing arguments. """
    # print("p:",p)
    arities = list(a[1] for a in p)
    accu_arities = list(accumulate(arities))
    accu_len = list(np.arange(len(p))+1)
    check = list(a < b for a,b in zip(accu_arities,accu_len))
    # print("accu_arities:",accu_arities)
    # print("accu_len:",accu_len)
    # print("accu_arities < accu_len:",accu_arities<accu_len)
    return all(check) and sum(a[1] for a in p) +1 == len(p) and len(p)>0
