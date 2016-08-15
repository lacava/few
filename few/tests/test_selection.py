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
from few.selection import tournament
import numpy as np
# unit tests for selection methods.
def test_tournament_shapes():
    """ tournament selection returns correct shape """
    pop = Pop(257)
    offspring = tournament(pop,2)
    assert len(offspring) == 257

    offspring = tournament(pop,5)
    assert len(offspring) == 257

    # smaller popsize than tournament size
    pop = Pop(2)
    offspring = tournament(pop,5)
    assert len(offspring) == 2;
    # pop.individuals[0].fitness = 10;
    # pop.individuals[1].fitness = 2;
# 
# def test_tournament_winners_are_better():
#     """ test tournament win conditions """
#     # in a tournament with replacement, there aren't many guarantees regarding
#     # the fitness structure of the resulting population. but we can say that the
#     # average fitness of the population should be at least as good as the
#     # parents
#     pop = Pop(5)
#     pop.individuals[0].fitness = 0
#     pop.individuals[1].fitness = 1
#     pop.individuals[2].fitness = 1
#     pop.individuals[3].fitness = 1
#     pop.individuals[4].fitness = 1
#
#     offspring = tournament(pop,100)
#
#     assert all(a.fitness == 0 for a in offspring)
