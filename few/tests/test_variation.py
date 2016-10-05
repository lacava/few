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

from few.tests.test_population import is_valid_program
from few.variation import cross, mutate

def test_cross_makes_valid_program():
    """test_variation.py: crossover makes valid programs """
    # np.random.seed(65)
    # I = (a+b)*x
    p1 = [('x',0,1),('x',0,2),('+',2),('x',0,3),('*',2)]
    # J = (x/z)-(n*b)
    p2 = [('x',0,1),('x',0,2),('/',2),('k',0,3.7),('x',0,4),('*',2),('-',2)]

    for i in np.arange(1000):
        cross(p1,p2)
        assert is_valid_program(p1) and is_valid_program(p2)

def test_mutate_makes_valid_program():
    """test_variation.py: mutation makes valid programs """
    func_set = [('+',2),('-',2),('*',2),('/',2),('sin',1),('cos',1),('exp',1),('log',1)]
    # terminal set
    term_set = []
    # numbers represent column indices of features
    for i in np.arange(10):
        term_set.append(('x',0,i)) # features
        term_set.append(('k',0,np.random.rand())) # ephemeral random constants

    p = [('k',0,5),('x',0,6),('/',2),('k',0,7),('x',0,8),('*',2),('-',2)]
    for i in np.arange(1000):
        mutate(p,func_set,term_set)
        assert is_valid_program(p)
