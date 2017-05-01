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
# from few.selection import *
from few import FEW
import numpy as np
# unit tests for selection methods.
def test_tournament_shapes():
    """test_selection.py: tournament selection returns correct shape"""

    few = FEW(seed_with_ml=False,population_size=257)
    few.term_set = [node('x',loc=0)]
    pop = few.init_pop()
    offspring,locs = few.tournament(pop.individuals,2)
    assert len(offspring) == 257

    offspring,locs = few.tournament(pop.individuals,5)
    assert len(offspring) == 257

    # smaller popsize than tournament size
    few = FEW(seed_with_ml=False,population_size=2)
    few.term_set = [node('x',loc=0)]
    pop = few.init_pop()
    offspring,locs = few.tournament(pop.individuals,5)
    assert len(offspring) == 2;

def test_lexicase_shapes():
    """test_selection.py: lexicase selection returns correct shape"""
    few = FEW(seed_with_ml=False,population_size=257)
    few.term_set = [node('x',loc=0)]
    pop = few.init_pop()
    offspring,locs = few.lexicase(pop.individuals)
    assert len(offspring) == 257

    # smaller popsize than tournament size
    few = FEW(seed_with_ml=False,population_size=2)
    few.term_set = [node('x',loc=0)]
    pop = few.init_pop()
    offspring,locs = few.lexicase(pop.individuals)
    assert len(offspring) == 2;

def test_epsilon_lexicase_shapes():
    """test_selection.py: epsilon lexicase selection returns correct shape"""

    few = FEW(seed_with_ml=False,population_size=257)
    few.term_set = [node('x',loc=0)]
    pop = few.init_pop()
    offspring,locs = few.lexicase(pop.individuals, epsilon=True)
    assert len(offspring) == 257

    # smaller popsize than tournament size
    few = FEW(seed_with_ml=False,population_size=2)
    few.term_set = [node('x',loc=0)]
    pop = few.init_pop()
    offspring,locs = few.lexicase(pop.individuals,epsilon=True)
    assert len(offspring) == 2;
    assert len(locs) == 2;

def test_lexicase_survival_shapes():
    """test_selection.py: lexicase survival returns correct shape"""
    # func_set = [node('+'), node('-'), node('*'), node('/'), node('sin'),
    #                  node('cos'), node('exp'),node('log'), node('^2'),
    #                  node('^3'), node('sqrt')]
    # terminal set
    term_set = []
    n_features = 3
    # numbers represent column indices of features
    # for i in np.arange(n_features):
    #     term_set.append(node('x',loc=i)) # features
    term_set = [node('x',loc=i) for i in np.arange(n_features)]
        # term_set.append(('erc',0,np.random.rand())) # ephemeral random constants

    few = FEW(seed_with_ml=False,population_size=257)
    few.term_set = term_set
    pop = few.init_pop()

    for i in pop.individuals:
        i.fitness_vec = list(np.random.rand(10,1))

    offspring,locs = few.lexicase(pop.individuals,num_selections=100,survival=True)
    assert len(offspring) == 100

    # smaller popsize than tournament size
    ew = FEW(seed_with_ml=False,population_size=2)
    few.term_set = term_set
    pop = few.init_pop()
    for i in pop.individuals:
        i.fitness_vec = np.random.rand(10,1)
    offspring,locs = few.lexicase(pop.individuals,num_selections=1,survival=True)
    assert len(offspring) == 1;
