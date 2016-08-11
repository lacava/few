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
import population

def cross(I,J):
    """ crossover between individuals I and J"""
    x_i = np.random.randint(0,len(I))
    # x_i_begin = x_i_end
    # while (I[x_i_end][1] - sum(I[x_i_begin:x_i_end][1]) != 0):
    #     --x_i_begin

    x_j_end = np.random.randint(len(J))
    x_j_begin = x_j_end
    while (J[x_j_end][1] - sum(arity[1] for arity in J[x_j_begin:x_j_end+1:]) != 0):
        x_j_begin -= 1

    return J[x_j_begin:x_j_end+1:] + I[x_i+1::]

def mutate(I,func_set,term_set):
    """ mutates individual I """
    x_i_end = np.random.randint(len(I))
    x_i_begin = x_i_end
    while (I[x_i_end][1] - sum(arity[1] for arity in I[x_i_begin:x_i_end+1:]) != 0):
        x_i_begin -= 1

    # swap mutation
    depth = 1
    newpiece = population.make_program(population.ind(),depth,func_set,term_set)
