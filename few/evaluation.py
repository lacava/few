# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava

license: GNU/GPLv3

"""
import numpy as np
from sklearn.metrics import explained_variance_score, mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
import pdb
from sklearn.metrics import silhouette_samples, silhouette_score, accuracy_score, roc_auc_score
import itertools as it
import sys
from sklearn.metrics.pairwise import pairwise_distances
# from profilehooks import profile
from sklearn.externals.joblib import Parallel, delayed

# safe division
def divs(x,y):
    """safe division"""
    tmp = np.ones(x.shape)
    nonzero_y = y != 0
    tmp[nonzero_y] = x[nonzero_y]/y[nonzero_y]
    return tmp

    # safe log
def logs(x):
    """safe log"""
    tmp = np.ones(x.shape)
    nonzero_x = x != 0
    tmp[nonzero_x] = np.log(np.abs(x[nonzero_x]))
    return tmp

# vectorized r2 score
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

class EvaluationMixin(object):
    """methods for evaluation."""

    #evaluation functions
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
    'inertia': lambda y,yhat: inertia(yhat,y),
    'separation': lambda y,yhat: 1 - separation(yhat,y),
    'fisher': lambda y,yhat: 1 - fisher(yhat,y),
    'accuracy': lambda y,yhat: 1 - accuracy_score(yhat,y),
    'random': lambda y,yhat: np.random.rand(),
    'roc_auc': lambda y,yhat: 1 - roc_auc_score(y,yhat)
    # 'relief': lambda y,yhat: 1-ReliefF(n_jobs=-1).fit(yhat.reshape(-1,1),y).feature_importances_
    }
    #
    f_vec = {# non-aggregated fitness calculations
    'mse': lambda y,yhat: (y - yhat) ** 2, #mean_squared_error(y,yhat,multioutput = 'raw_values'),
    'mae': lambda y,yhat: np.abs(y-yhat), #mean_absolute_error(y,yhat,multioutput = 'raw_values'),
    # 'mdae_vec': lambda y,yhat: median_absolute_error(y,yhat,multioutput = 'raw_values'),
    'r2':  lambda y,yhat: 1-r2_score_vec(y,yhat),
    'vaf': lambda y,yhat: 1-explained_variance_score(y,yhat,multioutput = 'raw_values'),
    'silhouette': lambda y,yhat: 1 - silhouette_samples(yhat.reshape(-1,1),y),
    'inertia': lambda y,yhat: inertia(yhat,y,samples=True),
    'separation': lambda y,yhat: 1 - separation(yhat,y,samples=True),
    'fisher': lambda y,yhat: 1 - fisher(yhat,y,samples=True),
    'accuracy': lambda y,yhat: 1 - np.sum(yhat==y)/y.shape[0], # this looks wrong, CHECK
    'random': lambda y,yhat: np.random.rand(len(y)),
    # 'relief': lambda y,yhat: 1-ReliefF(n_jobs=-1,sample_scores=True).fit(yhat.reshape(-1,1),y).feature_importances_
    }

    # f_vec = {# non-aggregated fitness calculations
    # 'mse':  (y - yhat) ** 2, #mean_squared_error(y,yhat,multioutput = 'raw_values'),
    # 'mae':  np.abs(y-yhat), #mean_absolute_error(y,yhat,multioutput = 'raw_values'),
    # # 'mdae_vec':  median_absolute_error(y,yhat,multioutput = 'raw_values'),
    # 'r2':   1-r2_score_vec(y,yhat),
    # 'vaf':  1-explained_variance_score(y,yhat,multioutput = 'raw_values'),
    # 'silhouette':  1 - silhouette_samples(yhat.reshape(-1,1),y),
    # 'inertia':  inertia(yhat,y,samples=True),
    # 'separation':  1 - separation(yhat,y,samples=True),
    # 'fisher':  1 - fisher(yhat,y,samples=True),
    # 'accuracy':  1 - np.sum(yhat==y)/y.shape[0],
    # 'random':  np.random.rand(len(y)),
    # # 'relief':  1-ReliefF(n_jobs=-1,sample_scores=True).fit(yhat.reshape(-1,1),y).feature_importances_
    # }


    def proper(self,x):
        """cleans fitness vector"""
        x[x < 0] = self.max_fit
        x[np.isnan(x)] = self.max_fit
        x[np.isinf(x)] = self.max_fit
        return x
    def safe(self,x):
        """removes nans and infs from outputs."""
        x[np.isinf(x)] = 1
        x[np.isnan(x)] = 1
        return x

    def evaluate(self,n, features, stack_float, stack_bool,labels=None):
        """evaluate node in program"""
        np.seterr(all='ignore')
        if len(stack_float) >= n.arity['f'] and len(stack_bool) >= n.arity['b']:
            if n.out_type == 'f':
                stack_float.append(
                    self.safe(self.eval_dict[n.name](n,features,stack_float,
                                                     stack_bool,labels)))
                if (np.isnan(stack_float[-1]).any() or
                    np.isinf(stack_float[-1]).any()):
                    print("problem operator:",n)
            else:
                stack_bool.append(self.safe(self.eval_dict[n.name](n,features,
                                                                   stack_float,
                                                                   stack_bool,
                                                                   labels)))
                if np.isnan(stack_bool[-1]).any() or np.isinf(stack_bool[-1]).any():
                    print("problem operator:",n)

    def all_finite(self,X):
        """returns true if X is finite, false, otherwise"""
        # Adapted from sklearn utils: _assert_all_finite(X)
        # First try an O(n) time, O(1) space solution for the common case that
        # everything is finite; fall back to O(n) space np.isfinite to prevent
        # false positives from overflow in sum method.
        # Note: this is basically here because sklearn tree.py uses float32 internally,
        # and float64's that are finite are not finite in float32.
        if (X.dtype.char in np.typecodes['AllFloat']
            and not np.isfinite(np.asarray(X,dtype='float32').sum())
            and not np.isfinite(np.asarray(X,dtype='float32')).all()):
            return False
        return True

    def out(self,I,features,labels=None,otype='f'):
        """computes the output for individual I"""
        stack_float = []
        stack_bool = []
        # print("stack:",I.stack)
        # evaulate stack over rows of features,labels
        # pdb.set_trace()
        for n in I.stack:
            self.evaluate(n,features,stack_float,stack_bool,labels)
            # print("stack_float:",stack_float)
        if otype=='f':
            return (stack_float[-1] if self.all_finite(stack_float[-1])
                    else np.zeros(len(features)))
        else:
            return (stack_bool[-1].astype(float) if self.all_finite(stack_bool[-1])
                    else np.zeros(len(features)))

    def calc_fitness(self,X,labels,fit_choice,sel):
        """computes fitness of individual output yhat.
        yhat: output of a program.
        labels: correct outputs
        fit_choice: choice of fitness function
        """

        if 'lexicase' in sel:
            # return list(map(lambda yhat: self.f_vec[fit_choice](labels,yhat),X))
            return np.asarray(
                              [self.proper(self.f_vec[fit_choice](labels,
                                                        yhat)) for yhat in X],
                                                        order='F')
            # return list(Parallel(n_jobs=-1)(delayed(self.f_vec[fit_choice])(labels,yhat) for yhat in X))
        else:
            # return list(map(lambda yhat: self.f[fit_choice](labels,yhat),X))
            return np.asarray([self.f[fit_choice](labels,yhat) for yhat in X],
                            order='F').transpose()

            # return list(Parallel(n_jobs=-1)(delayed(self.f[fit_choice])(labels,yhat) for yhat in X))

    def inertia(self,X,y,samples=False):
        """ return the within-class squared distance from the centroid"""
        # pdb.set_trace()
        if samples:
            # return within-class distance for each sample
            inertia = np.zeros(y.shape)
            for label in np.unique(y):
                inertia[y==label] = (X[y==label] - np.mean(X[y==label])) ** 2

        else: # return aggregate score
            inertia = 0
            for i,label in enumerate(np.unique(y)):
                inertia += np.sum((X[y==label] - np.mean(X[y==label])) ** 2)/len(y[y==label])
            inertia = inertia/len(np.unique(y))

        return inertia

    def separation(self,X,y,samples=False):
        """ return the sum of the between-class squared distance"""
        # pdb.set_trace()
        num_classes = len(np.unique(y))
        total_dist = (X.max()-X.min())**2
        if samples:
            # return intra-class distance for each sample
            separation = np.zeros(y.shape)
            for label in np.unique(y):
                for outsider in np.unique(y[y!=label]):
                    separation[y==label] += (X[y==label] - np.mean(X[y==outsider])) ** 2

            #normalize between 0 and 1
            print('separation:',separation)
            print('num_classes:',num_classes)
            print('total_dist:',total_dist)
            separation = separation#/separation.max()

            print('separation after normalization:',separation)

        else:
            # return aggregate score
            separation = 0
            for i,label in enumerate(np.unique(y)):
                for outsider in np.unique(y[y!=label]):
                    separation += np.sum((X[y==label] - np.mean(X[y==outsider])) ** 2)/len(y[y==label])
            separation = separation/len(np.unique(y))

        return separation

    def pairwise(self,iterable):
        "s -> (s0,s1), (s1,s2), (s2, s3), ..."
        a, b = it.tee(iterable)
        next(b, None)
        return zip(a, b)


    def fisher(self,yhat,y,samples=False):
        """Fisher criterion"""
        classes = np.unique(y)
        mu = np.zeros(len(classes))
        v = np.zeros(len(classes))
        # pdb.set_trace()
        for c in classes.astype(int):
            mu[c] = np.mean(yhat[y==c])
            v[c] = np.var(yhat[y==c])

        if not samples:
            fisher = 0
            for c1,c2 in pairwise(classes.astype(int)):
                fisher += np.abs(mu[c1] - mu[c2])/np.sqrt(v[c1]+v[c2])
        else:
            # lexicase version
            fisher = np.zeros(len(yhat))
            # get closests classes to each class (min mu distance)
            mu_d = pairwise_distances(mu.reshape(-1,1))
            min_mu=np.zeros(len(classes),dtype=int)
            for i in np.arange(len(min_mu)):
                min_mu[i] = np.argsort(mu_d[i])[1]
            # for c1, pairwise(classes.astype(int)):
            #     min_mu[c1] = np.argmin()
            for i,l in enumerate(yhat.astype(int)):
                fisher[i] = np.abs(l - mu[min_mu[y[i]]])/np.sqrt(v[y[i]]+v[min_mu[y[i]]])

        # pdb.set_trace()
        return fisher
