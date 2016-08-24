# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava

license: GNU/GPLv3

"""
import numpy as np
import pdb
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import itertools as it
import math
# import pdb

# evaluation functions. these can be sped up using a GPU!
eval_dict = {
    '+': lambda n,features,stack_float: stack_float.pop() + stack_float.pop(),
    '-': lambda n,features,stack_float: stack_float.pop() - stack_float.pop(),
    '*': lambda n,features,stack_float: stack_float.pop() * stack_float.pop(),
    '/': lambda n,features,stack_float: stack_float.pop() / stack_float.pop(),
    'sin': lambda n,features,stack_float: np.sin(stack_float.pop()),
    'cos': lambda n,features,stack_float: np.cos(stack_float.pop()),
    'exp': lambda n,features,stack_float: np.exp(stack_float.pop()),
    'log': lambda n,features,stack_float: np.log(stack_float.pop()),
    'x':  lambda n,features,stack_float: features[:,n[2]],
    'k': lambda n,features,stack_float: np.ones(features.shape[0])*n[2]
}
def safe(x):
    """removes nans and infs from outputs."""
    x[np.isinf(x)] = 0
    x[np.isnan(x)] = 0
    return x

def eval(n, features, stack_float):

    if len(stack_float) >= n[1]:
        stack_float.append(safe(eval_dict[n[0]](n,features,stack_float)))
        if any(np.isnan(stack_float[-1])) or any(np.isinf(stack_float[-1])):
            print("problem operator:",n)

def out(I,features,labels=None):
    """computes the output for individual I"""
    stack_float = []
    # print("stack:",I.stack)
    # evaulate stack over rows of features,labels
    for n in I.stack:
        eval(n,features,stack_float)
        # print("stack_float:",stack_float)

    return stack_float[-1]

def calc_fitness(pop,labels,fit_choice):
    """computes fitness of individual output yhat.
    yhat: output of a program.
    labels: correct outputs
    fit_choice: choice of fitness function
    """
    f = { # available fitness metrics
    'mse': lambda y,yhat: mean_squared_error(y,yhat),
    'mae': lambda y,yhat: mean_absolute_error(y,yhat),
    'mdae': lambda y,yhat: median_absolute_error(y,yhat),
    'r2':  lambda y,yhat: 1-r2_score(y,yhat),
    'vaf': lambda y,yhat: 1-explained_variance_score(y,yhat),
    # non-aggregated fitness calculations
    'mse_vec': lambda y,yhat: (y - yhat) ** 2, #mean_squared_error(y,yhat,multioutput = 'raw_values'),
    'mae_vec': lambda y,yhat: np.abs(y-yhat), #mean_absolute_error(y,yhat,multioutput = 'raw_values'),
    'mdae_vec': lambda y,yhat: median_absolute_error(y,yhat,multioutput = 'raw_values'),
    'r2_vec':  lambda y,yhat: 1-r2_score(y,yhat,multioutput = 'raw_values'),
    'vaf_vec': lambda y,yhat: 1-explained_variance_score(y,yhat,multioutput = 'raw_values')
    }

    if (fit_choice[-3::] == 'rel'): # relative fitness calculation
        # calculate fitness of pairwise-comparisons of individual residuals
        # a row of zeros is added to X to return the metric w.r.t. the labels
        pop.E = np.asarray(list(map(lambda yhat: labels - yhat,np.vstack((pop.X,np.zeros(labels.shape))))))
        fitness = []
        fitmap = np.asarray(list(map(lambda xe: f[fit_choice[0:-4]](xe[1],xe[0]),it.product(pop.X,pop.E))))
        num_fits = math.floor(fitmap.shape[0]/len(pop.individuals))
        # print("fitmap.shape[0]:",fitmap.shape[0],"len(pop.individuals):",len(pop.individuals),"num_fits:",num_fits)
        for i in np.arange(0,fitmap.shape[0],num_fits):
            fitness.append(fitmap[i:i+num_fits])

        return fitness
    # pdb.set_trace()
    # standard fitness calculation
    return list(map(lambda yhat: f[fit_choice](labels,yhat),pop.X))
