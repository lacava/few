# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava

license: GNU/GPLv3

"""
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import itertools as it
import math
import pdb
from numpy.linalg import norm
from .population import in_type, out_type
from sklearn.metrics import silhouette_samples, silhouette_score
# evaluation functions. these can be sped up using a GPU!
eval_dict = {
# float operations
    '+': lambda n,features,stack_float,stack_bool: stack_float.pop() + stack_float.pop(),
    '-': lambda n,features,stack_float,stack_bool: stack_float.pop() - stack_float.pop(),
    '*': lambda n,features,stack_float,stack_bool: stack_float.pop() * stack_float.pop(),
    '/': lambda n,features,stack_float,stack_bool: divs(stack_float.pop(),stack_float.pop()),
    'sin': lambda n,features,stack_float,stack_bool: np.sin(stack_float.pop()),
    'cos': lambda n,features,stack_float,stack_bool: np.cos(stack_float.pop()),
    'exp': lambda n,features,stack_float,stack_bool: np.exp(stack_float.pop()),
    'log': lambda n,features,stack_float,stack_bool: logs(stack_float.pop()),#np.log(np.abs(stack_float.pop())),
    'x':  lambda n,features,stack_float,stack_bool: features[:,n[2]],
    'k': lambda n,features,stack_float,stack_bool: np.ones(features.shape[0])*n[2],
    '^2': lambda n,features,stack_float,stack_bool: stack_float.pop()**2,
    '^3': lambda n,features,stack_float,stack_bool: stack_float.pop()**3,
    'sqrt': lambda n,features,stack_float,stack_bool: np.sqrt(np.abs(stack_float.pop())),
    # 'rbf': lambda n,features,stack_float,stack_bool: np.exp(-(np.norm(stack_float.pop()-stack_float.pop())**2)/2)
# bool operations
    '!': lambda n,features,stack_float,stack_bool: not stack_bool.pop(),
    '&': lambda n,features,stack_float,stack_bool: stack_bool.pop() and stack_bool.pop(),
    '|': lambda n,features,stack_float,stack_bool: stack_bool.pop() or stack_bool.pop(),
    '==': lambda n,features,stack_float,stack_bool: stack_bool.pop() == stack_bool.pop(),
    '>_f': lambda n,features,stack_float,stack_bool: stack_float.pop() > stack_float.pop(),
    '<_f': lambda n,features,stack_float,stack_bool: stack_float.pop() < stack_float.pop(),
    '>=_f': lambda n,features,stack_float,stack_bool: stack_float.pop() >= stack_float.pop(),
    '<=_f': lambda n,features,stack_float,stack_bool: stack_float.pop() <= stack_float.pop(),
    '>_b': lambda n,features,stack_float,stack_bool: stack_bool.pop() > stack_bool.pop(),
    '<_b': lambda n,features,stack_float,stack_bool: stack_bool.pop() < stack_bool.pop(),
    '>=_b': lambda n,features,stack_float,stack_bool: stack_bool.pop() >= stack_bool.pop(),
    '<=_b': lambda n,features,stack_float,stack_bool: stack_bool.pop() <= stack_bool.pop(),
}
f = { # available fitness metrics
'mse': lambda y,yhat: mean_squared_error(y,yhat),
'mae': lambda y,yhat: mean_absolute_error(y,yhat),
'mdae': lambda y,yhat: median_absolute_error(y,yhat),
'r2':  lambda y,yhat: 1-r2_score(y,yhat),
'vaf': lambda y,yhat: 1-explained_variance_score(y,yhat),
'silhouette': lambda y,yhat: 1 - silhouette_score(yhat.reshape(-1,1),y),
# non-aggregated fitness calculations
'mse_vec': lambda y,yhat: (y - yhat) ** 2, #mean_squared_error(y,yhat,multioutput = 'raw_values'),
'mae_vec': lambda y,yhat: np.abs(y-yhat), #mean_absolute_error(y,yhat,multioutput = 'raw_values'),
'mdae_vec': lambda y,yhat: median_absolute_error(y,yhat,multioutput = 'raw_values'),
'r2_vec':  lambda y,yhat: 1-r2_score_vec(y,yhat),
'vaf_vec': lambda y,yhat: 1-explained_variance_score(y,yhat,multioutput = 'raw_values'),
'silhouette_vec': lambda y,yhat: 1 - silhouette_samples(yhat.reshape(-1,1),y),
}

def safe(x):
    """removes nans and infs from outputs."""
    x[np.isinf(x)] = 1
    x[np.isnan(x)] = 1
    return x

def divs(x,y):
    """safe division"""
    tmp = np.ones(x.shape)
    nonzero_y = y != 0
    tmp[nonzero_y] = x[nonzero_y]/y[nonzero_y]
    return tmp

def logs(x):
    """safe log"""
    tmp = np.ones(x.shape)
    nonzero_x = x != 0
    tmp[nonzero_x] = np.log(np.abs(x[nonzero_x]))
    return tmp


def eval(n, features, stack_float, stack_bool):
    np.seterr(all='ignore')
    if (in_type[n[0]]=='f' and len(stack_float) >= n[1]) or (in_type[n[0]]=='b' and len(stack_bool) >= n[1]):
        stack_float.append(safe(eval_dict[n[0]](n,features,stack_float,stack_bool)))
        if any(np.isnan(stack_float[-1])) or any(np.isinf(stack_float[-1])):
            print("problem operator:",n)

def out(I,features,labels=None):
    """computes the output for individual I"""
    stack_float = []
    stack_bool = []
    # print("stack:",I.stack)
    # evaulate stack over rows of features,labels
    for n in I.stack:
        eval(n,features,stack_float,stack_bool)
        # print("stack_float:",stack_float)

    return stack_float[-1]

def calc_fitness(X,labels,fit_choice):
    """computes fitness of individual output yhat.
    yhat: output of a program.
    labels: correct outputs
    fit_choice: choice of fitness function
    """

    # pdb.set_trace()
    return list(map(lambda yhat: f[fit_choice](labels,yhat),X))

def r2_score_vec(y_true,y_pred):
    """ returns non-aggregate version of r2 score.

    based on r2_score() function from sklearn (http://sklearn.org)
    """

    numerator = (y_true - y_pred) ** 2
    denominator = (y_true - np.average(y_true)) ** 2

    nonzero_denominator = denominator != 0
    nonzero_numerator = numerator != 0
    valid_score = nonzero_denominator & nonzero_numerator
    output_scores = np.ones([y_true.shape[0]])
    output_scores[valid_score] = 1 - (numerator[valid_score] /
                                      denominator[valid_score])
    # arbitrary set to zero to avoid -inf scores, having a constant
    # y_true is not interesting for scoring a regression anyway
    output_scores[nonzero_numerator & ~nonzero_denominator] = 0.

    return output_scores
