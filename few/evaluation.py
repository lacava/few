# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava

license: GNU/GPLv3

"""
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import pdb
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score


# evaluation functions. these can be sped up using a GPU!

eval_dict = {
# float operations
    '+': lambda n,features,stack_float,stack_bool,labels: stack_float.pop() + stack_float.pop(),
    '-': lambda n,features,stack_float,stack_bool,labels: stack_float.pop() - stack_float.pop(),
    '*': lambda n,features,stack_float,stack_bool,labels: stack_float.pop() * stack_float.pop(),
    '/': lambda n,features,stack_float,stack_bool,labels: divs(stack_float.pop(),stack_float.pop()),
    'sin': lambda n,features,stack_float,stack_bool,labels: np.sin(stack_float.pop()),
    'cos': lambda n,features,stack_float,stack_bool,labels: np.cos(stack_float.pop()),
    'exp': lambda n,features,stack_float,stack_bool,labels: np.exp(stack_float.pop()),
    'log': lambda n,features,stack_float,stack_bool,labels: logs(stack_float.pop()),#np.log(np.abs(stack_float.pop())),
    'x':  lambda n,features,stack_float,stack_bool,labels: features[:,n.loc],
    'k': lambda n,features,stack_float,stack_bool,labels: np.ones(features.shape[0])*n.value,
    '^2': lambda n,features,stack_float,stack_bool,labels: stack_float.pop()**2,
    '^3': lambda n,features,stack_float,stack_bool,labels: stack_float.pop()**3,
    'sqrt': lambda n,features,stack_float,stack_bool,labels: np.sqrt(np.abs(stack_float.pop())),
    # 'rbf': lambda n,features,stack_float,stack_bool,labels: np.exp(-(np.norm(stack_float.pop()-stack_float.pop())**2)/2)
# bool operations
    '!': lambda n,features,stack_float,stack_bool,labels: np.logical_not(stack_bool.pop()),
    '&': lambda n,features,stack_float,stack_bool,labels: np.logical_and(stack_bool.pop(), stack_bool.pop()),
    '|': lambda n,features,stack_float,stack_bool,labels: np.logical_or(stack_bool.pop(), stack_bool.pop()),
    '==': lambda n,features,stack_float,stack_bool,labels: stack_bool.pop() == stack_bool.pop(),
    '>_f': lambda n,features,stack_float,stack_bool,labels: stack_float.pop() > stack_float.pop(),
    '<_f': lambda n,features,stack_float,stack_bool,labels: stack_float.pop() < stack_float.pop(),
    '>=_f': lambda n,features,stack_float,stack_bool,labels: stack_float.pop() >= stack_float.pop(),
    '<=_f': lambda n,features,stack_float,stack_bool,labels: stack_float.pop() <= stack_float.pop(),
    '>_b': lambda n,features,stack_float,stack_bool,labels: stack_bool.pop() > stack_bool.pop(),
    '<_b': lambda n,features,stack_float,stack_bool,labels: stack_bool.pop() < stack_bool.pop(),
    '>=_b': lambda n,features,stack_float,stack_bool,labels: stack_bool.pop() >= stack_bool.pop(),
    '<=_b': lambda n,features,stack_float,stack_bool,labels: stack_bool.pop() <= stack_bool.pop(),
    'xor_b': lambda n,features,stack_float,stack_bool,labels: np.logical_xor(stack_bool.pop(),stack_bool.pop()),
    'xor_f': lambda n,features,stack_float,stack_bool,labels: np.logical_xor(stack_float.pop().astype(bool), stack_float.pop().astype(bool)),
# MDR
    'mdr2': lambda n,features,stack_float,stack_bool,labels: n.evaluate(n,stack_float,labels),
# control flow:
    # 'if': lambda n,features,stack_float,stack_bool,labels: stack_float.pop() if stack_bool.pop(),
    # 'ife': lambda n,features,stack_float,stack_bool,labels: stack_float.pop() if stack_bool.pop() else stack_float.pop(),
    }

f = { # available fitness metrics
'mse': lambda y,yhat: mean_squared_error(y,yhat),
'mae': lambda y,yhat: mean_absolute_error(y,yhat),
'mdae': lambda y,yhat: median_absolute_error(y,yhat),
'r2':  lambda y,yhat: 1-r2_score(y,yhat),
'vaf': lambda y,yhat: 1-explained_variance_score(y,yhat),
'silhouette': lambda y,yhat: 1 - silhouette_score(yhat.reshape(-1,1),y),
'accuracy': lambda y,yhat: 1 - accuracy_score(yhat,y),
# non-aggregated fitness calculations
'mse_vec': lambda y,yhat: (y - yhat) ** 2, #mean_squared_error(y,yhat,multioutput = 'raw_values'),
'mae_vec': lambda y,yhat: np.abs(y-yhat), #mean_absolute_error(y,yhat,multioutput = 'raw_values'),
# 'mdae_vec': lambda y,yhat: median_absolute_error(y,yhat,multioutput = 'raw_values'),
'r2_vec':  lambda y,yhat: 1-r2_score_vec(y,yhat),
'vaf_vec': lambda y,yhat: 1-explained_variance_score(y,yhat,multioutput = 'raw_values'),
'silhouette_vec': lambda y,yhat: 1 - silhouette_samples(yhat.reshape(-1,1),y),
'accuracy_vec': lambda y,yhat: 1 - np.sum(yhat==y)/y.shape[0]
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


def evaluate(n, features, stack_float, stack_bool,labels=None):
    """evaluate node in program"""
    np.seterr(all='ignore')
    if len(stack_float) >= n.arity['f'] and len(stack_bool) >= n.arity['b']:
        if n.out_type == 'f':
            stack_float.append(safe(eval_dict[n.name](n,features,stack_float,stack_bool,labels)))
            if np.isnan(stack_float[-1]).any() or np.isinf(stack_float[-1]).any():
                print("problem operator:",n)
        else:
            stack_bool.append(safe(eval_dict[n.name](n,features,stack_float,stack_bool,labels)))
            if np.isnan(stack_bool[-1]).any() or np.isinf(stack_bool[-1]).any():
                print("problem operator:",n)

def all_finite(X):
    """returns true if X is finite, false, otherwise"""
    # Adapted from sklearn utils: _assert_all_finite(X)
    # First try an O(n) time, O(1) space solution for the common case that
    # everything is finite; fall back to O(n) space np.isfinite to prevent
    # false positives from overflow in sum method.
    # Note: this is basically here because sklearn tree.py uses float32 internally,
    # and float64's that are finite are not finite in float32.
    if (X.dtype.char in np.typecodes['AllFloat'] and not np.isfinite(np.asarray(X,dtype='float32').sum())
        and not np.isfinite(np.asarray(X,dtype='float32')).all()):
        return False
    return True

def out(I,features,labels=None,otype='f'):
    """computes the output for individual I"""
    stack_float = []
    stack_bool = []
    # print("stack:",I.stack)
    # evaulate stack over rows of features,labels
    # pdb.set_trace()
    for n in I.stack:
        evaluate(n,features,stack_float,stack_bool,labels)
        # print("stack_float:",stack_float)
    if otype=='f':
        return stack_float[-1] if all_finite(stack_float[-1]) else np.zeros(len(features))
    else:
        return stack_bool[-1] if all_finite(stack_bool[-1]) else np.zeros(len(features))

def calc_fitness(X,labels,fit_choice):
    """computes fitness of individual output yhat.
    yhat: output of a program.
    labels: correct outputs
    fit_choice: choice of fitness function
    """



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
