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
from .population import make_program, ind

def cross(I,J):
    """subtree-like swap crossover between individuals I and J."""
    x_i_end = np.random.randint(0,len(I))

    x_i_begin = x_i_end
    arity_sum = I[x_i_end][1]
    print("x_i_end:",x_i_end)
    while (arity_sum > 0):
        print("arity_sum:",arity_sum,"x_i_begin:",x_i_begin,"x_i_end:",x_i_end)
        x_i_begin -= 1
        arity_sum += I[x_i_begin][1]-1

    x_j_end = np.random.randint(len(J))
    x_j_begin = x_j_end
    arity_sum = J[x_j_end][1]

    while (arity_sum > 0):
        print("arity_sum:",arity_sum,"x_j_begin:",x_j_begin,"x_j_end:",x_j_end)
        x_j_begin -= 1
        arity_sum += J[x_j_begin][1]-1

    print("J subtree:",J[x_j_begin:x_j_end+1:])
    print("I subtree:",I[x_i_begin:x_i_end+1:])
    # #print("Returned tree:",J[x_j_begin:x_j_end+1:] + I[x_i::])
    # X = del I[-1]
    #swap subtrees
    I[x_i_begin:x_i_end+1:],J[x_j_begin:x_j_end+1:] = J[x_j_begin:x_j_end+1:],I[x_i_begin:x_i_end+1:]

    print(I)
    print(J)
    # return I,J

def mutate(I,func_set,term_set):
    """ mutates individual I """
    x_i_end = np.random.randint(len(I))
    x_i_begin = x_i_end
    while (I[x_i_end][1] - sum(arity[1] for arity in I[x_i_begin:x_i_end+1:]) != 0):
        x_i_begin -= 1

    # swap mutation
    depth = 1
    newpiece = make_program([],func_set,term_set,depth)
