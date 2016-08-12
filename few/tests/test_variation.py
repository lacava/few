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
# unit tests for variation methods.
import numpy as np

from few.variation import cross, mutate

def test_cross_makes_valid_program():
    # np.random.seed(65)
    # I = (a+b)*x
    I = [('a',0,1),('b',0,2),('+',2),('x',0,3),('*',2)]
    # J = (x/z)-(n*b)
    J = [('x',0,1),('z',0,2),('/',2),('n',0,3),('b',0,4),('*',2),('-',2)]

    for i in np.arange(1000):
        cross(I,J)
        assert sum(a[1] for a in I) +1 == len(I)
        assert sum(a[1] for a in J) +1 == len(J)

def test_mutate_makes_valid_program():

    func_set = [('+',2),('-',2),('*',2),('/',2),('sin',1),('cos',1),('exp',1),('log',1)]
    # terminal set
    term_set = [('x',0),('z',0),('n',0),('b',0)]
    # numbers represent column indices of features
    for i in np.arange(10):
        term_set.append(('n',0,i)) # features
        term_set.append(('erc',0,np.random.rand())) # ephemeral random constants

    I = [('n',0,5),('n',0,6),('/',2),('n',0,7),('n',0,8),('*',2),('-',2)]
    for i in np.arange(1000):
        mutate(I,func_set,term_set)
        assert sum(a[1] for a in I) +1 == len(I)
