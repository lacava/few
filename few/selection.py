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

def tournament(pop,tourn_size):
    """ conducts tournament selection of size tourn_size, returning len(pop)
    individuals. """
    winners = []
    for i in np.arange(len(pop.individuals)):
        # sample pool with replacement
        pool_i = np.random.choice(len(pop.individuals),size=tourn_size)
        pool = []
        for i in pool_i:
            pool.append(pop.individuals[i].fitness)

        winners.append(pop.individuals[pool_i[np.argmin(pool)]])
    # print("size winners:",len(winners))
    # for index,i in enumerate(winners):
    #     print("fitness "+str(index)+":",i.fitness)

    return winners
