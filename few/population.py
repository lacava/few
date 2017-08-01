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
import itertools as it

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
    def __init__(self,pop_size=100, fit = None):
        """initializes population of inds of size pop_size"""

        self.individuals = []
        # initialize population programs
        for i in np.arange(pop_size):
            if fit is None:
                self.individuals.append(Ind())
            else:
                self.individuals.append(Ind(fitness = fit))

class PopMixin(object):
    """methods for constructing features."""
    ######################################################## printing equations
    def eval_eqn(self,n,stack_eqn):
        if len(stack_eqn) >= n.arity['f']+n.arity['b']:
            stack_eqn.append(eqn_dict[n.name](n,stack_eqn))

    def stack_2_eqn(self,p):
        """returns equation string for program stack"""
        stack_eqn = []
        if p: # if stack is not empty
            for n in p.stack:
                self.eval_eqn(n,stack_eqn)
            return stack_eqn[-1]
        return []

    def stacks_2_eqns(self,stacks):
        """returns equation strings from stacks"""
        if stacks:
            return list(map(lambda p: self.stack_2_eqn(p), stacks))
        else:
            return []

    ########################################################### making programs
    def make_program(self,stack,func_set,term_set,max_d,ntype):
        """makes a program stack"""
        # print("stack:",stack,"max d:",max_d)
        if max_d == 0:
            ts = [t for t in term_set if t.out_type==ntype]

            if not ts:
                raise ValueError('no ts. ntype:'+ntype+'. term_set out_types:'+
                                 ','.join([t.out_type for t in term_set]))

            stack.append(ts[np.random.choice(len(ts))])
        else:
            fs = [f for f in func_set if (f.out_type==ntype
                                          and (f.in_type=='f' or max_d>1))]
            if len(fs)==0:
                print('ntype:',ntype,'\nfunc_set:',[f.name for f in func_set])
            stack.append(fs[np.random.choice(len(fs))])
            tmp = copy.copy(stack[-1])

            for i in np.arange(tmp.arity['f']):
                self.make_program(stack,func_set,term_set,max_d-1,'f')
            for i in np.arange(tmp.arity['b']):
                self.make_program(stack,func_set,term_set,max_d-1,'b')

    def init_pop(self):
        """initializes population of features as GP stacks."""
        pop = Pop(self.population_size)
        seed_with_raw_features = False
        # make programs
        if self.seed_with_ml:
            # initial population is the components of the default ml model
            if (self.ml_type == 'SVC' or self.ml_type == 'SVR'):
                # this is needed because svm has a bug that throws valueerror
                #on attribute check
                seed_with_raw_features=True
            elif (hasattr(self.ml.named_steps['ml'],'coef_') or
                  hasattr(self.ml.named_steps['ml'],'feature_importances_')):
                # add model components with non-zero coefficients to initial
                # population, in order of coefficient size
                coef = (self.ml.named_steps['ml'].coef_ if
                        hasattr(self.ml.named_steps['ml'],'coef_') else
                        self.ml.named_steps['ml'].feature_importances_)
                # compress multiple coefficients for each feature into single
                # numbers (occurs with multiclass classification)
                if len(coef.shape)>1:
                    coef = [np.mean(abs(c)) for c in coef.transpose()]

                # remove zeros
                coef = [c for c in coef if c!=0]
                # sort feature locations based on importance/coefficient
                locs = np.arange(len(coef))
                locs = locs[np.argsort(np.abs(coef))[::-1]]
                for i,p in enumerate(pop.individuals):
                    if i < len(locs):
                        p.stack = [node('x',loc=locs[i])]
                    else:
                        # make program if pop is bigger than n_features
                        self.make_program(p.stack,self.func_set,self.term_set,
                                     self.random_state.randint(self.min_depth,
                                                       self.max_depth+1),
                                     self.otype)
                        p.stack = list(reversed(p.stack))
            else:
                seed_with_raw_features = True
            # seed with random features if no importance info available
            if seed_with_raw_features:
                for i,p in enumerate(pop.individuals):
                    if i < self.n_features:
                        p.stack = [node('x',
                                        loc=np.random.randint(self.n_features))]
                    else:
                        # make program if pop is bigger than n_features
                        self.make_program(p.stack,self.func_set,self.term_set,
                                     self.random_state.randint(self.min_depth,
                                                       self.max_depth+1),
                                     self.otype)
                        p.stack = list(reversed(p.stack))

            # print initial population
            if self.verbosity > 2:
                print("seeded initial population:",
                      self.stacks_2_eqns(pop.individuals))

        else: # don't seed with ML
            for I in pop.individuals:
                depth = self.random_state.randint(self.min_depth,
                                                  self.max_depth+1)
                self.make_program(I.stack,self.func_set,self.term_set,depth,
                             self.otype)
                # print(I.stack)
                I.stack = list(reversed(I.stack))

        return pop
