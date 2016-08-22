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
import copy

def tournament(pop,tourn_size):
    """conducts tournament selection of size tourn_size, returning len(pop)
    individuals."""
    winners = []
    for i in np.arange(len(pop.individuals)):
        # sample pool with replacement
        pool_i = np.random.choice(len(pop.individuals),size=tourn_size)
        pool = []
        for i in pool_i:
            pool.append(np.mean(pop.individuals[i].fitness))

        winners.append(copy.deepcopy(pop.individuals[pool_i[np.argmin(pool)]]))
    # print("size winners:",len(winners))
    # for index,i in enumerate(winners):
    #     print("fitness "+str(index)+":",i.fitness)
    # print("winners:",pop.stacks_2_eqns())
    return winners

def lexicase(pop):
    """conducts lexicase selection for de-aggregated fitness vectors"""

    winners = []

    for i in np.arange(len(pop.individuals)):

        candidates = pop.individuals
        # print("pop.individuals[0].fitness",pop.individuals[0].fitness)
        cases = list(np.arange(len(pop.individuals[0].fitness_vec)))
        np.random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            # get elite fitness for case
            best_val_for_case = min(map(lambda x: x.fitness_vec[cases[0]], pop.individuals))
            # filter individuals without an elite fitness on this case
            candidates = list(filter(lambda x: x.fitness_vec[cases[0]] == best_val_for_case, pop.individuals))
            cases.pop(0)

        winners.append(copy.deepcopy(np.random.choice(candidates)))
    # print("winners:",pop.stacks_2_eqns())
    return winners

def epsilon_lexicase(pop, num_selections=None):
    """conducts epsilon lexicase selection for de-aggregated fitness vectors"""

    if num_selections is None:
        num_selections = len(pop.individuals)

    winners = []
    mad_for_case = []
    best_val_for_case = []
    # calculate epsilon thresholds based on median absolute deviation (MAD)
    for i in np.arange(len(pop.individuals[0].fitness_vec)):
        mad_for_case.append(mad(np.asarray(list(map(lambda x: x.fitness_vec[i], pop.individuals)))))
        best_val_for_case.append(min(map(lambda x: x.fitness_vec[i], pop.individuals)))

    for i in np.arange(num_selections):

        candidates = pop.individuals
        cases = list(range(len(pop.individuals[0].fitness_vec)))
        np.random.shuffle(cases)

        while len(cases) > 0 and len(candidates) > 1:
            if not np.isinf(best_val_for_case):
                # filter individuals without an elite+epsilon fitness on this case
                candidates = list(filter(lambda x: x.fitness_vec[cases[0]] <= best_val_for_case[cases[0]]+mad_for_case[cases[0]], pop.individuals))

            cases.pop(0)

        winners.append(copy.deepcopy(np.random.choice(candidates)))

    # print("winners:",pop.stacks_2_eqns())

    return winners

def mad(x, axis=None):
    """median absolute deviation statistics"""
    return np.median(np.abs(x - np.median(x, axis)), axis)
