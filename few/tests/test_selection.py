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
from few.population import *
from few.selection import *
import numpy as np
# unit tests for selection methods.
def test_tournament_shapes():
    """test_selection.py: tournament selection returns correct shape"""
    pop = Pop(257)
    offspring,locs = tournament(pop.individuals,2)
    assert len(offspring) == 257

    offspring,locs = tournament(pop.individuals,5)
    assert len(offspring) == 257

    # smaller popsize than tournament size
    pop = Pop(2)
    offspring,locs = tournament(pop.individuals,5)
    assert len(offspring) == 2;

def test_lexicase_shapes():
    """test_selection.py: lexicase selection returns correct shape"""
    pop = Pop(257)
    offspring,locs = lexicase(pop.individuals)
    assert len(offspring) == 257

    # smaller popsize than tournament size
    pop = Pop(2)
    offspring,locs = lexicase(pop.individuals)
    assert len(offspring) == 2;

def test_epsilon_lexicase_shapes():
    """test_selection.py: epsilon lexicase selection returns correct shape"""

    pop = Pop(257,fit = 1)
    offspring,locs = lexicase(pop.individuals, epsilon=True)
    assert len(offspring) == 257

    # smaller popsize than tournament size
    pop = Pop(2,fit = 0)
    offspring,locs = lexicase(pop.individuals,epsilon=True)
    assert len(offspring) == 2;
    assert len(locs) == 2;

def test_lexicase_survival_shapes():
    """test_selection.py: lexicase survival returns correct shape"""
    func_set = [('+',2),('-',2),('*',2),('/',2),('sin',1),('cos',1),('exp',1),('log',1)]
    # terminal set
    term_set = []
    n_features = 3
    # numbers represent column indices of features
    for i in np.arange(n_features):
        term_set.append(('x',0,i)) # features
        # term_set.append(('erc',0,np.random.rand())) # ephemeral random constants

    pop = init(257,10,func_set,term_set,1,5)
    for i in pop.individuals:
        i.fitness_vec = list(np.random.rand(10,1))

    offspring,locs = lexicase(pop.individuals,num_selections=100,survival=True)
    assert len(offspring) == 100

    # smaller popsize than tournament size
    pop = init(2,10,func_set,term_set,1,5)
    for i in pop.individuals:
        i.fitness_vec = np.random.rand(10,1)
    offspring,locs = lexicase(pop.individuals,num_selections=1,survival=True)
    assert len(offspring) == 1;
