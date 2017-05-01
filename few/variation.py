# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava

license: GNU/GPLv3

"""
import numpy as np
from .population import make_program
from itertools import accumulate
import pdb
import copy
import itertools as it
import uuid
from .population import Ind
# from few.tests.test_population import is_valid_program
class VariationMixin(object):
    """ Defines crossover and mutation operator methods."""
    def variation(self,parents):
        """performs variation operators on parents."""
        # downselect to features that are important
        if type(self.ml).__name__ != 'SVC' and type(self.ml).__name__ != 'SVR': # this is needed because svm has a bug that throws valueerror on attribute check
            if hasattr(self.ml,'coef_'):
                # for l1 regularization, filter individuals with 0 coefficients
                offspring = copy.deepcopy(list(x for i,x in zip(self.ml.coef_, self.valid(parents)) if  (i != 0).any()))
            elif hasattr(self.ml,'feature_importances_'):
                # for tree methods, filter our individuals with 0 feature importance
                offspring = copy.deepcopy(list(x for i,x in zip(self.ml.feature_importances_, self.valid(parents)) if  i != 0))
            else:
                offspring = copy.deepcopy(self.valid(parents))
        else:
            offspring = copy.deepcopy(self.valid(parents))

        if self.elitism: # keep a copy of the elite individual
            elite_index = np.argmin([x.fitness for x in parents])
            elite = copy.deepcopy(parents[elite_index])

        # Apply crossover and mutation on the offspring
        if self.verbosity > 2: print("variation...")
        for child1, child2 in it.zip_longest(offspring[::2], offspring[1::2],fillvalue=None):

            if np.random.rand() < self.crossover_rate and child2 != None:
            # crossover
                self.cross(child1.stack, child2.stack, self.max_depth)
                # update ids
                child1.parentid = [child1.id,child2.id]
                child1.id = uuid.uuid4()
                child2.parentid = [child1.id,child2.id]
                child2.id = uuid.uuid4()
                # set default fitness
                child1.fitness = -1
                child2.fitness = -1
            elif child2 == None:
            # single mutation
                self.mutate(child1.stack,self.func_set,self.term_set)
                # update ids
                child1.parentid = [child1.id]
                child1.id = uuid.uuid4()
                # set default fitness
                child1.fitness = -1
            else:
            #double mutation
                self.mutate(child1.stack,self.func_set,self.term_set)
                self.mutate(child2.stack,self.func_set,self.term_set)
                # update ids
                child1.parentid = [child1.id]
                child1.id = uuid.uuid4()
                child2.parentid = [child2.id]
                child2.id = uuid.uuid4()
                # set default fitness
                child1.fitness = -1
                child2.fitness = -1

        while len(offspring) < self.population_size:
            #make new offspring to replace the invalid ones
            offspring.append(Ind())
            make_program(offspring[-1].stack,self.func_set,self.term_set,np.random.randint(self.min_depth,self.max_depth+1),self.otype)
            offspring[-1].stack = list(reversed(offspring[-1].stack))

        return offspring,elite,elite_index

    def cross(self,p_i,p_j, max_depth = 2):
        """subtree-like swap crossover between programs p_i and p_j."""
        # only choose crossover points for out_types available in both programs
        # pdb.set_trace()
        # determine possible outttypes
        types_p_i = [t for t in [p.out_type for p in p_i]]
        types_p_j = [t for t in [p.out_type for p in p_j]]
        types = set(types_p_i).intersection(types_p_j)

        # grab subtree of p_i
        p_i_sub = [i for i,n in enumerate(p_i) if n.out_type in types]
        x_i_end = np.random.choice(p_i_sub)
        x_i_begin = x_i_end
        arity_sum = p_i[x_i_end].arity[p_i[x_i_end].in_type]
        # print("x_i_end:",x_i_end)
        # i = 0
        while (arity_sum > 0): #and i < 1000:
            if x_i_begin == 0:
                print("arity_sum:",arity_sum,"x_i_begin:",x_i_begin,"x_i_end:",x_i_end)
            x_i_begin -= 1
            arity_sum += p_i[x_i_begin].arity[p_i[x_i_begin].in_type]-1
        #     i += 1
        # if i == 1000:
        #     print("in variation")
        #     pdb.set_trace()

        # grab subtree of p_j with matching out_type to p_i[x_i_end]
        p_j_sub = [i for i,n in enumerate(p_j) if n.out_type == p_i[x_i_end].out_type]
        x_j_end = np.random.choice(p_j_sub)
        x_j_begin = x_j_end
        arity_sum = p_j[x_j_end].arity[p_j[x_j_end].in_type]
        # i = 0
        while (arity_sum > 0): #and i < 1000:
            if x_j_begin == 0:
                print("arity_sum:",arity_sum,"x_j_begin:",x_j_begin,"x_j_end:",x_j_end)
                print("p_j:",p_j)
            x_j_begin -= 1
            arity_sum += p_j[x_j_begin].arity[p_j[x_j_begin].in_type]-1
        #     i += 1
        # if i == 1000:
        #     print("in variation")
        #     pdb.set_trace()
        #swap subtrees
        tmpi = p_i[:]
        tmpj = p_j[:]
        tmpi[x_i_begin:x_i_end+1:],tmpj[x_j_begin:x_j_end+1:] = tmpj[x_j_begin:x_j_end+1:],tmpi[x_i_begin:x_i_end+1:]

        if not self.is_valid_program(p_i) or not self.is_valid_program(p_j):
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


    def mutate(self,p_i,func_set,term_set): #, max_depth=2
        """point mutation, addition, removal"""
        self.point_mutate(p_i,func_set,term_set)

    def point_mutate(self,p_i,func_set,term_set):
        """point mutation on individual p_i"""
        # point mutation
        x = np.random.randint(len(p_i))
        arity = p_i[x].arity[p_i[x].in_type]
        # find eligible replacements based on arity and type
        reps = [n for n in func_set+term_set
                if n.arity[n.in_type]==arity and n.out_type==p_i[x].out_type
                and n.in_type==p_i[x].in_type]

        tmp = reps[np.random.randint(len(reps))]
        tmp_p = p_i[:]
        p_i[x] = tmp
        if not self.is_valid_program(p_i):
            print("old:",tmp_p)
            print("new:",p_i)
            raise ValueError('Mutation produced an invalid program.')
    #
    # def add_mutate(p_i,func_set,term_set, max_depth=2):
    #     """ mutation that adds operation to program"""
    #     #choose node. move it down, pick an operator to put before it, with another leaf if
    #     #new operator arity requires it.
    #     #make sure size requirements are not invalidated (if they are, discard changes)
    #
    # def sub_mutate(p_i,func_set,term_set, max_depth=2):
    #     """ mutation that removes operation from program"""
    #     #choose a node with arity>0. replace it and its subtree with a node with lower arity.

    def is_valid_program(self,p):
        """checks whether program p makes a syntactically valid tree.

        checks that the accumulated program length is always greater than the
        accumulated arities, indicating that the appropriate number of arguments is
        alway present for functions. It then checks that the sum of arties +1
        exactly equals the length of the stack, indicating that there are no
        missing arguments.
        """
        # print("p:",p)
        arities = list(a.arity[a.in_type] for a in p)
        accu_arities = list(accumulate(arities))
        accu_len = list(np.arange(len(p))+1)
        check = list(a < b for a,b in zip(accu_arities,accu_len))
        # print("accu_arities:",accu_arities)
        # print("accu_len:",accu_len)
        # print("accu_arities < accu_len:",accu_arities<accu_len)
        return all(check) and sum(a.arity[a.in_type] for a in p) +1 == len(p) and len(p)>0
