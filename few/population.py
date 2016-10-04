# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava

license: GNU/GPLv3

"""
import numpy as np
import copy
import pdb
eqn_dict = {
    '+': lambda n,stack_eqn: '(' + stack_eqn.pop() + '+' + stack_eqn.pop() + ')',
    '-': lambda n,stack_eqn: '(' + stack_eqn.pop() + '-' + stack_eqn.pop()+ ')',
    '*': lambda n,stack_eqn: '(' + stack_eqn.pop() + '*' + stack_eqn.pop()+ ')',
    '/': lambda n,stack_eqn: '(' + stack_eqn.pop() + '/' + stack_eqn.pop()+ ')',
    'sin': lambda n,stack_eqn: 'sin(' + stack_eqn.pop() + ')',
    'cos': lambda n,stack_eqn: 'cos(' + stack_eqn.pop() + ')',
    'exp': lambda n,stack_eqn: 'exp(' + stack_eqn.pop() + ')',
    'log': lambda n,stack_eqn: 'log(' + stack_eqn.pop() + ')',
    '^2': lambda n,stack_eqn: '(' + stack_eqn.pop() + '^2)',
    '^3': lambda n,stack_eqn: '(' + stack_eqn.pop() + '^3)',
    'sqrt': lambda n,stack_eqn: 'sqrt(|' + stack_eqn.pop() + '|)',
    # 'rbf': lambda n,stack_eqn: 'exp(-||' + stack_eqn.pop()-stack_eqn.pop() '||^2/2)',
    'x':  lambda n,stack_eqn: 'x_' + str(n[2]),
    'k': lambda n,stack_eqn: str(n[2])
}

class Ind(object):
    """class for features, represented as GP stacks."""

    def __init__(self,fitness = -1.0,stack = None):
        """initializes empty individual with invalid fitness."""
        self.fitness = fitness
        self.fitness_vec = []
        self.fitness_bool = []

        if stack is None:
            self.stack = []
        else:
            self.stack = copy.deepcopy(stack)

class Pop(object):
    """class representing population"""
    def __init__(self,pop_size=100,n_samples=1, fit = None):
        """initializes population of inds of size pop_size"""

        self.individuals = []
        # initialize empty output matrix
        self.X = np.empty([n_samples,pop_size],dtype=float,order='F')
        # initialize empty error matrix
        self.E = np.empty(0) #np.empty([n_samples,pop_size],dtype=float)
        # initialize population programs
        for i in np.arange(pop_size):
            if fit is None:
                self.individuals.append(Ind())
            else:
                self.individuals.append(Ind(fitness = fit))

def stacks_2_eqns(stacks):
    """returns equation strings from stacks"""
    if stacks:
        return list(map(lambda p: stack_2_eqn(p), stacks))
    else:
        return []

def stack_2_eqn(p):
    """returns equation string for program stack"""
    stack_eqn = []
    if p: # if stack is not empty
        for n in p.stack:
            eval_eqn(n,stack_eqn)
        return stack_eqn[-1]
    return []

def eval_eqn(n,stack_eqn):
    if len(stack_eqn) >= n[1]:
        stack_eqn.append(eqn_dict[n[0]](n,stack_eqn))
        # if any(np.isnan(stack_eqn[-1])) or any(np.isinf(stack_eqn[-1])):
        #     print("problem operator:",n)

def make_program(stack,func_set,term_set,max_d,ntype):
    """makes a program stack"""
    # print("stack:",stack,"max d:",max_d)
    if max_d == 0: #or np.random.rand() < float(len(term_set))/(len(term_set)+len(func_set)):
        ts = [t for t in term_set if out_type[t[0]]==ntype]

        # if not ts:
        #     pdb.set_trace()
        #     fs = [f for f in func_set if out_type[f[0]]==ntype and in_type[f[0]]==ntype]
        #     stack.append(fs[np.random.choice(len(fs))])
        #     for i in np.arange(stack[-1][1]):
        #         make_program(stack,func_set,term_set,max_d-1,in_type[stack[-1][0]])

        stack.append(ts[np.random.choice(len(ts))])
    else:
        fs = [f for f in func_set if (out_type[f[0]]==ntype and (in_type[f[0]]=='f' or max_d>1))]
        # if not fs:
        #     pdb.set_trace()
        stack.append(fs[np.random.choice(len(fs))])
        for i in np.arange(stack[-1][1]):
            make_program(stack,func_set,term_set,max_d-1,in_type[stack[-1][0]])
    # return stack
    # print("current stack:",stack)

def init(population_size,n_samples,func_set,term_set,min_depth,max_depth):
    """initializes population of features as GP stacks"""
    pop = Pop(population_size,n_samples)

    for I in pop.individuals:
        depth = np.random.randint(min_depth,max_depth+1)
        # print("hex(id(I)):",hex(id(I)))
        # depth = 2;
        # print("initial I.stack:",I.stack)
        make_program(I.stack,func_set,term_set,depth,'f')
        # print(I.stack)
        I.stack = list(reversed(I.stack))

    # print(I.stack)

    return pop

in_type = {
# float operations
    '+':'f', '-':'f', '*':'f', '/':'f', 'sin':'f', 'cos':'f', 'exp': 'f',
    'log':'f', 'x':'f', 'k':'f', '^2':'f', '^3':'f', 'sqrt': 'f',
    # 'rbf': ,
# bool operations
    '!':'b', '&':'b', '|':'b', '==':'b', '>_f':'f', '<_f':'f', '>=_f':'f',
    '<=_f':'f', '>_b':'b', '<_b':'b', '>=_b':'b', '<=_b':'b',
}
out_type = {
# float operations
    '+': 'f','-': 'f','*': 'f','/': 'f','sin': 'f','cos': 'f','exp': 'f',
    'log': 'f','x':  'f','k': 'f','^2': 'f','^3': 'f','sqrt': 'f',
    # 'rbf': ,
# bool operations
    '!': 'b', '&': 'b','|': 'b','==': 'b','>_f': 'b','<_f': 'b','>=_f': 'b',
    '<=_f': 'b','>_b': 'b','<_b': 'b','>=_b': 'b','<=_b': 'b',
}
