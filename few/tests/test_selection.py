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
from few.population import ind, Pop, init
from few.selection import tournament,lexicase,epsilon_lexicase
import numpy as np
# unit tests for selection methods.
def test_tournament_shapes():
    """test_selection.py: tournament selection returns correct shape """
    pop = Pop(257)
    offspring = tournament(pop,2)
    assert len(offspring) == 257

    offspring = tournament(pop,5)
    assert len(offspring) == 257

    # smaller popsize than tournament size
    pop = Pop(2)
    offspring = tournament(pop,5)
    assert len(offspring) == 2;

def test_lexicase_shapes():
    """test_selection.py: lexicase selection returns correct shape """
    pop = Pop(257)
    offspring = lexicase(pop)
    assert len(offspring) == 257

    # smaller popsize than tournament size
    pop = Pop(2)
    offspring = lexicase(pop)
    assert len(offspring) == 2;

def test_epsilon_lexicase_shapes():
    """test_selection.py: epsilon lexicase selection returns correct shape """
    """test_selection.py: lexicase selection returns correct shape """
    pop = Pop(257,fit = 1)
    offspring = epsilon_lexicase(pop)
    assert len(offspring) == 257

    # smaller popsize than tournament size
    pop = Pop(2,fit = 0)
    offspring = epsilon_lexicase(pop)
    assert len(offspring) == 2;
