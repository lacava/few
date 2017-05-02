# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava

license: GNU/GPLv3

"""
import numpy as np
import copy
import pdb
import uuid
from mdr import MDR
from collections import defaultdict

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
    'x':  lambda n,stack_eqn: 'x_' + str(n.loc),
    'k': lambda n,stack_eqn: str(n.value),
    'mdr2': lambda n,stack_eqn: 'MDR(' + stack_eqn.pop() + ',' + stack_eqn.pop() + ')',
# bool operations
    '!': lambda n,stack_eqn: '(!' + stack_eqn.pop() + ')',
    '&': lambda n,stack_eqn: '(' + stack_eqn.pop() + '&' + stack_eqn.pop() + ')',
    '|': lambda n,stack_eqn: '(' + stack_eqn.pop() + '|' + stack_eqn.pop() + ')',
    '==': lambda n,stack_eqn: '(' + stack_eqn.pop() + '==' + stack_eqn.pop() + ')',
    '>_f': lambda n,stack_eqn: '(' + stack_eqn.pop() + '>' + stack_eqn.pop() + ')',
    '<_f': lambda n,stack_eqn: '(' + stack_eqn.pop() + '<' + stack_eqn.pop() + ')',
    '>=_f': lambda n,stack_eqn: '(' + stack_eqn.pop() + '>=' + stack_eqn.pop() + ')',
    '<=_f': lambda n,stack_eqn: '(' + stack_eqn.pop() + '<=' + stack_eqn.pop() + ')',
    '>_b': lambda n,stack_eqn: '(' + stack_eqn.pop() + '>' + stack_eqn.pop() + ')',
    '<_b': lambda n,stack_eqn: '(' + stack_eqn.pop() + '<' + stack_eqn.pop() + ')',
    '>=_b': lambda n,stack_eqn: '(' + stack_eqn.pop() + '>=' + stack_eqn.pop() + ')',
    '<=_b': lambda n,stack_eqn: '(' + stack_eqn.pop() + '<=' + stack_eqn.pop() + ')',
    'xor_b': lambda n,stack_eqn: '(' + stack_eqn.pop() + ' XOR ' + stack_eqn.pop() + ')',
    'xor_f': lambda n,stack_eqn: '(' + stack_eqn.pop() + ' XOR ' + stack_eqn.pop() + ')',

}
def run_MDR(n,stack_float,labels=None):
    """run utility function for MDR nodes."""
    # need to check that tmp is categorical

    x1 = stack_float.pop()
    x2 = stack_float.pop()
    # check data is categorical
    if len(np.unique(x1))<=3 and len(np.unique(x2))<=3:
        tmp = np.vstack((x1,x2)).transpose()

        if labels is None: # prediction
            return n.model.transform(tmp)[:,0]
        else: # training
            out =  n.model.fit_transform(tmp,labels)[:,0]

            return out
    else:
        return np.zeros(x1.shape[0])

class node(object):
    """node in program"""
    def __init__(self,name,loc=None,value=None):
        """defines properties of a node given its name"""
        self.name = name
        self.arity = {None:0}
        self.arity['f'] = defaultdict(lambda: 0, {
                           'sin':1,'cos':1,'exp':1,'log':1,'^2':1,'^3':1,
                           'sqrt':1,'if':1,
                           'ife':2,'+':2,'-':2,'*':2,'/':2,'>_f':2,'<_f':2,
                           '>=_f':2,'<=_f':2,'xor_f':2,'mdr2':2})[name]

        self.arity['b'] = defaultdict(lambda: 0, {
                            '!':1,'if':1,'ife':1,
                            '&':2,'|':2,'==':2,'>_b':2,'<_b':2,'>=_b':2,
                            '<=_b':2,'xor_b':2})[name]
        self.in_type = {
        # float operations
            '+':'f', '-':'f', '*':'f', '/':'f', 'sin':'f', 'cos':'f', 'exp': 'f',
            'log':'f', 'x':None, 'k':None, '^2':'f', '^3':'f', 'sqrt': 'f',
            # 'rbf': ,
        # bool operations
            '!':'b', '&':'b', '|':'b', '==':'b', '>_f':'f', '<_f':'f', '>=_f':'f',
            '<=_f':'f', '>_b':'b', '<_b':'b', '>=_b':'b', '<=_b':'b','xor_b':'b',
            'xor_f':'f',
        # mixed
            'mdr2':'f','if':('f','b'),'ife':('f','b')
        }[name]
        self.out_type = {
        # float operations
            '+': 'f','-': 'f','*': 'f','/': 'f','sin': 'f','cos': 'f','exp': 'f',
            'log': 'f','x':'f','k': 'f','^2': 'f','^3': 'f','sqrt': 'f',
            # 'rbf': ,
        # bool operations
            '!': 'b', '&': 'b','|': 'b','==': 'b','>_f': 'b','<_f': 'b','>=_f': 'b',
            '<=_f': 'b','>_b': 'b','<_b': 'b','>=_b': 'b','<=_b': 'b','xor_f':'b',
            'xor_b':'b',
        # mixed
            'mdr2':'b','if':'f','ife':'f'
        }[name]

        if 'mdr' in self.name:
            self.model = MDR()
            self.evaluate = run_MDR

        self.loc = loc
        self.value = value


class Ind(object):
    """class for features, represented as GP stacks."""

    def __init__(self,fitness = -1.0,stack = None):
        """initializes empty individual with invalid fitness."""
        self.fitness = fitness
        self.fitness_vec = []
        self.fitness_bool = []
        self.parentid = []
        self.id = uuid.uuid4()
        if stack is None:
            self.stack = []
        else:
            self.stack = copy.deepcopy(stack)

class Pop(object):
    """class representing population"""
    def __init__(self,pop_size=100,n_samples=1, fit = None):
        """initializes population of inds of size pop_size"""

        self.individuals = []
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
    if len(stack_eqn) >= n.arity['f']+n.arity['b']:
        stack_eqn.append(eqn_dict[n.name](n,stack_eqn))
        # if any(np.isnan(stack_eqn[-1])) or any(np.isinf(stack_eqn[-1])):
        #     print("problem operator:",n)

def make_program(stack,func_set,term_set,max_d,ntype):
    """makes a program stack"""
    # print("stack:",stack,"max d:",max_d)
    if max_d == 0: #or np.random.rand() < float(len(term_set))/(len(term_set)+len(func_set)):
        ts = [t for t in term_set if t.out_type==ntype]

        if not ts:
            raise ValueError('no ts. ntype:'+ntype+'. term_set out_types:'+','.join([t.out_type for t in term_set]))

        stack.append(ts[np.random.choice(len(ts))])
    else:
        fs = [f for f in func_set if (f.out_type==ntype and (f.in_type=='f' or max_d>1))]
        if len(fs)==0:
            print('ntype:',ntype,'\nfunc_set:',[f.name for f in func_set])
        stack.append(fs[np.random.choice(len(fs))])
        tmp = copy.copy(stack[-1])

        for i in np.arange(tmp.arity['f']):
            make_program(stack,func_set,term_set,max_d-1,'f')
        for i in np.arange(tmp.arity['b']):
            make_program(stack,func_set,term_set,max_d-1,'b')
