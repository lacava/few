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
from few import FEW
from few.tests.test_population import is_valid_program,node

def test_cross_makes_valid_program():
    """test_variation.py: crossover makes valid programs """
    # np.random.seed(65)
    # I = (a+b)*x
    p1 = [node('x',loc=1),node('x',loc=2),node('+'),node('x',loc=3),node('*')]
    # J = (x/z)-(n*b)
    p2 = [node('x',loc=1), node('x',loc=2), node('/'), node('k',value=3.7), node('x',loc=4), node('*'), node('-')]
    # test 1000 crossover events
    few = FEW()
    for i in np.arange(1000):
        few.cross(p1,p2)
        assert is_valid_program(p1) and is_valid_program(p2)

def test_mutate_makes_valid_program():
    """test_variation.py: mutation makes valid programs """
    func_set = [node('+'), node('-'), node('*'), node('/'), node('sin'),
                 node('cos'), node('exp'),node('log'), node('^2'),
                 node('^3'), node('sqrt')]
    # terminal set
    term_set = []
    # numbers represent column indices of features
    term_set = [node('x',loc=i) for i in np.arange(10)]
    term_set += [node('k',value=np.random.rand()) for i in np.arange(10)]
    # program
    p = [node('k',value=5),node('x',loc=6),node('/'),node('k',value=7),node('x',loc=8),node('*'),node('-')]
    # test 1000 mutation events
    few = FEW()
    for i in np.arange(1000):
        few.mutate(p,func_set,term_set)
        assert is_valid_program(p)
