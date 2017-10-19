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
    np.random.seed(42)
    few = FEW(seed_with_ml=False,population_size=257)
    few.term_set = [node('x',loc=0)]
    pop = few.init_pop()
    offspring = few.lexicase(np.random.rand(257,100))
    assert len(offspring) == 257

    # smaller popsize than tournament size
    np.random.seed(42)
    few = FEW(seed_with_ml=False,population_size=2)
    few.term_set = [node('x',loc=0)]
    pop = few.init_pop()
    offspring = few.lexicase(np.random.rand(2,100))
    assert len(offspring) == 2;

def test_epsilon_lexicase_shapes():
    """test_selection.py: epsilon lexicase selection returns correct shape"""
    np.random.seed(42)
    few = FEW(seed_with_ml=False,population_size=257,lex_size=False)
    few.term_set = [node('x',loc=0)]
    pop = few.init_pop()
    offspring = few.epsilon_lexicase(np.random.rand(257,100),[])
    assert len(offspring) == 257

    # smaller popsize than tournament size
    few = FEW(seed_with_ml=False,population_size=2,lex_size=False)
    few.term_set = [node('x',loc=0)]
    pop = few.init_pop()
    offspring = few.epsilon_lexicase(np.random.rand(2,100),[])
    assert len(offspring) == 2;

def test_lex_size():
    """test_selection.py: lex_size flag on/off"""

    few = FEW(seed_with_ml=False,population_size=257, lex_size=True)

    Fitness_mat = np.random.rand(257,10)
    size_mat = np.random.randint(1,100,size=257)

    locs = few.epsilon_lexicase(Fitness_mat,size_mat,num_selections=100,
                                          survival=True)
    assert len(locs) == 100

    few = FEW(seed_with_ml=False,population_size=257, lex_size=False)

    Fitness_mat = np.random.rand(257,10)
    size_mat = np.random.rand(257,1)

    locs = few.epsilon_lexicase(Fitness_mat,size_mat,num_selections=100,
                                          survival=True)
    assert len(locs) == 100
