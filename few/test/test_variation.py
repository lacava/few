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
import sys

from ..src import variation as vary

def test_cross_makes_valid_program():
    np.random.seed(65)
    # I = (a+b)*x
    I = [('a',0),('b',0),('+',2),('x',0),('*',2)]
    # J = (x/z)-(n*b)
    J = [('x',0),('z',0),('/',2),('n',0),('b',0),('*',2),('-',2)]

    for i in np.arange(100):
        vary.cross(I,J)
        assert valid_program(I)
        assert valid_program(J)

def valid_program(I):
    """ checks that program forms a correct tree. """
    return sum(a[1] for a in I) +1 == len(I)
