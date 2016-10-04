# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava

license: GNU/GPLv3

"""
import numpy as np
from .population import make_program, Ind, in_type, out_type
from itertools import accumulate
import pdb
# from few.tests.test_population import is_valid_program

def cross(p_i,p_j, max_depth = 3):
    """subtree-like swap crossover between programs p_i and p_j."""
    # only choose crossover points for out_types available in both programs
    # pdb.set_trace()
    # determine possible outttypes
    types_p_i = [t for t in [out_type[p[0]] for p in p_i]]
    types_p_j = [t for t in [out_type[p[0]] for p in p_j]]
    types = set(types_p_i).intersection(types_p_j)

    # grab subtree of p_i
    p_i_sub = [i for i,n in enumerate(p_i) if out_type[n[0]] in types]
    x_i_end = np.random.choice(p_i_sub)
    x_i_begin = x_i_end
    arity_sum = p_i[x_i_end][1]
    # print("x_i_end:",x_i_end)
    i = 0
    while (arity_sum > 0) and i < 1000:
        if x_i_begin == 0:
            print("arity_sum:",arity_sum,"x_i_begin:",x_i_begin,"x_i_end:",x_i_end)
        x_i_begin -= 1
        arity_sum += p_i[x_i_begin][1]-1
        i += 1
    if i == 1000:
        print("in variation")
        pdb.set_trace()

    # grab subtree of p_j with matching out_type to p_i[x_i_end]
    p_j_sub = [i for i,n in enumerate(p_j) if out_type[n[0]] == out_type[p_i[x_i_end][0]]]
    x_j_end = np.random.choice(p_j_sub)
    x_j_begin = x_j_end
    arity_sum = p_j[x_j_end][1]
    i = 0
    while (arity_sum > 0) and i < 1000:
        if x_j_begin == 0:
            print("arity_sum:",arity_sum,"x_j_begin:",x_j_begin,"x_j_end:",x_j_end)
            print("p_j:",p_j)
        x_j_begin -= 1
        arity_sum += p_j[x_j_begin][1]-1
        i += 1
    if i == 1000:
        print("in variation")
        pdb.set_trace()
    #swap subtrees
    tmpi = p_i[:]
    tmpj = p_j[:]
    tmpi[x_i_begin:x_i_end+1:],tmpj[x_j_begin:x_j_end+1:] = tmpj[x_j_begin:x_j_end+1:],tmpi[x_i_begin:x_i_end+1:]

    if not is_valid_program(p_i) or not is_valid_program(p_j):
        # pdb.set_trace()
        print("parent 1:",p_i,"x_i_begin:",x_i_begin,"x_i_end:",x_i_end)
        print("parent 2:",p_j,"x_j_begin:",x_j_begin,"x_j_end:",x_j_end)
        print("child 1:",tmpi)
        print("child 2:",tmpj)
        raise ValueError('Crossover produced an invalid program.')

    # size check, then assignment
    if len(tmpi) <= 2**max_depth-1:
        p_i[:] = tmpi
    if len(tmpj) <= 2**max_depth-1:
        p_j[:] = tmpj



def mutate(p_i,func_set,term_set):
    """point mutation on individual p_i"""
    # point mutation
    x = np.random.randint(len(p_i))
    arity = p_i[x][1]
    wholeset = func_set+term_set
    reps = [n for n in func_set+term_set
            if n[1]==arity and out_type[n[0]]==out_type[p_i[x][0]] and in_type[n[0]]==in_type[p_i[x][0]]]
    tmp = reps[np.random.randint(len(reps))]
    tmp_p = p_i[:]
    p_i[x] = tmp
    if not is_valid_program(p_i):
        print("old:",tmp_p)
        print("new:",p_i)
        raise ValueError('Mutation produced an invalid program.')


def is_valid_program(p):
    """checks whether program p makes a syntactically valid tree.

    checks that the accumulated program length is always greater than the
    accumulated arities, indicating that the appropriate number of arguments is
    alway present for functions. It then checks that the sum of arties +1
    exactly equals the length of the stack, indicating that there are no
    missing arguments.
    """
    # print("p:",p)
    arities = list(a[1] for a in p)
    accu_arities = list(accumulate(arities))
    accu_len = list(np.arange(len(p))+1)
    check = list(a < b for a,b in zip(accu_arities,accu_len))
    # print("accu_arities:",accu_arities)
    # print("accu_len:",accu_len)
    # print("accu_arities < accu_len:",accu_arities<accu_len)
    return all(check) and sum(a[1] for a in p) +1 == len(p) and len(p)>0
