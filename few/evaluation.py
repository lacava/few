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
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
import tensorflow as tf

import pdb

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

def divs_tf(x,y):
    """safe division"""
    return tf.where(tf.equal(y,tf.zeros(tf.shape(x),dtype=tf.float32)),
                   tf.ones(tf.shape(x),dtype=tf.float32),
                   tf.realdiv(x,y))
def logs_tf(x):
    """safe log"""
    return tf.where(tf.equal(x,tf.zeros(tf.shape(x),dtype=tf.float32)),
                   tf.ones(tf.shape(x),dtype=tf.float32),
                   tf.log(tf.abs(x)))

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

    def transform(self,x,inds=None,labels = None):
        """return a transformation of x using population outputs"""

        if inds:
            if self.tf:
                result = np.asarray(
                    [self.out_tf(I,x,labels,self.otype) for I in inds]).transpose()

                return result
                #construct feed_dict
                # feed_dict={}
                # for i,_ in enumerate(x.transpose()):
                    # pdb.set_trace()
                    # feed_dict['x'+str(i)+':0'] = x[:,i]
                # Initialize TensorFlow session
                # tf.reset_default_graph() # Reset TF internal state and cache
                # # tf.device('/cpu:0')
                # config = tf.ConfigProto(allow_soft_placement=False)
                # config.gpu_options.allow_growth = False
                # # pdb.set_trace()
                # programs = [self.build_tf_graph(I,x,labels) for I in inds]
                # # run graphs
                # with tf.Session(config=config) as sess:
                #     with tf.device('/gpu:1'):
                #
                #         # compile list of graphs
                #         result = sess.run(programs)
                #     # tf.train.write_graph(sess.graph,'tmp','graph.txt',as_text=True)
                # # pdb.set_trace()
                # return np.asarray(result).transpose()
                # #  if self.all_finite(result)
                #          else np.zeros(len(features)))
            else:
                return np.asarray(
                    [self.out(I,x,labels,self.otype) for I in inds]).transpose()
        else:
            # return np.asarray(Parallel(n_jobs=10)(delayed(self.out)(I,x,labels,self.otype) for I in self._best_inds)).transpose()
            return np.asarray(
                [self.out(I,x,labels,self.otype) for I in self._best_inds]).transpose()


    #evaluation functions
    eval_dict = {
    # float operations
        '+': lambda n,features,stack,labels: stack['f'].pop() + stack['f'].pop(),
        '-': lambda n,features,stack,labels: stack['f'].pop() - stack['f'].pop(),
        '*': lambda n,features,stack,labels: stack['f'].pop() * stack['f'].pop(),
        '/': lambda n,features,stack,labels: divs(stack['f'].pop(),stack['f'].pop()),
        'sin': lambda n,features,stack,labels: np.sin(stack['f'].pop()),
        'cos': lambda n,features,stack,labels: np.cos(stack['f'].pop()),
        'exp': lambda n,features,stack,labels: np.exp(stack['f'].pop()),
        'log': lambda n,features,stack,labels: logs(stack['f'].pop()),#np.log(np.abs(stack['f'].pop())),
        'x':  lambda n,features,stack,labels: features[:,n.loc],
        'k': lambda n,features,stack,labels: np.ones(features.shape[0])*n.value,
        '^2': lambda n,features,stack,labels: stack['f'].pop()**2,
        '^3': lambda n,features,stack,labels: stack['f'].pop()**3,
        'sqrt': lambda n,features,stack,labels: np.sqrt(np.abs(stack['f'].pop())),
        # 'rbf': lambda n,features,stack,labels: np.exp(-(np.norm(stack['f'].pop()-stack['f'].pop())**2)/2)
    # bool operations
        '!': lambda n,features,stack,labels: np.logical_not(stack['b'].pop()),
        '&': lambda n,features,stack,labels: np.logical_and(stack['b'].pop(), stack['b'].pop()),
        '|': lambda n,features,stack,labels: np.logical_or(stack['b'].pop(), stack['b'].pop()),
        '==': lambda n,features,stack,labels: stack['b'].pop() == stack['b'].pop(),
        '>_f': lambda n,features,stack,labels: stack['f'].pop() > stack['f'].pop(),
        '<_f': lambda n,features,stack,labels: stack['f'].pop() < stack['f'].pop(),
        '>=_f': lambda n,features,stack,labels: stack['f'].pop() >= stack['f'].pop(),
        '<=_f': lambda n,features,stack,labels: stack['f'].pop() <= stack['f'].pop(),
        '>_b': lambda n,features,stack,labels: stack['b'].pop() > stack['b'].pop(),
        '<_b': lambda n,features,stack,labels: stack['b'].pop() < stack['b'].pop(),
        '>=_b': lambda n,features,stack,labels: stack['b'].pop() >= stack['b'].pop(),
        '<=_b': lambda n,features,stack,labels: stack['b'].pop() <= stack['b'].pop(),
        'xor_b': lambda n,features,stack,labels: np.logical_xor(stack['b'].pop(),stack['b'].pop()),
        'xor_f': lambda n,features,stack,labels: np.logical_xor(stack['f'].pop().astype(bool), stack['f'].pop().astype(bool)),
    # MDR
        'mdr2': lambda n,features,stack,labels: n.evaluate(n,stack['f'],labels),
    # control flow:
        # 'if': lambda n,features,stack,labels: stack['f'].pop() if stack['b'].pop(),
        # 'ife': lambda n,features,stack,labels: stack['f'].pop() if stack['b'].pop() else stack['f'].pop(),
        }
    ##################################### evaluation functions using tensor flow
    tf_dict = {
    # float operations
        '+': lambda n,features,stack,labels: tf.add(stack['f'].pop(), stack['f'].pop()),
        '-': lambda n,features,stack,labels: tf.subtract(stack['f'].pop(),stack['f'].pop()),
        '*': lambda n,features,stack,labels: tf.multiply(stack['f'].pop(),stack['f'].pop()),
        '/': lambda n,features,stack,labels: divs_tf(stack['f'].pop(),stack['f'].pop()),
        'sin': lambda n,features,stack,labels: tf.sin(stack['f'].pop()),
        'cos': lambda n,features,stack,labels: tf.cos(stack['f'].pop()),
        'exp': lambda n,features,stack,labels: tf.exp(stack['f'].pop()),
        'log': lambda n,features,stack,labels: logs_tf(stack['f'].pop()),#np.log(np.abs(stack['f'].pop())),
        'x':  lambda n,features,stack,labels: tf.placeholder(tf.float32,(None,),name='x'+str(n.loc)),#tf.constant(features[:,n.loc],name='x'+str(n.loc)),
        'k': lambda n,features,stack,labels: tf.constant(n.value,dtype=tf.float32),#tf.constant(n.value,dtype=tf.float32,shape=[features.shape[0],],name=str(n.value)),
        '^2': lambda n,features,stack,labels: tf.square(stack['f'].pop()),
        '^3': lambda n,features,stack,labels: tf.pow(stack['f'].pop(),3),
        'sqrt': lambda n,features,stack,labels: tf.sqrt(np.abs(stack['f'].pop())),
        # 'rbf': lambda n,features,stack,labels: np.exp(-(np.norm(stack['f'].pop()-stack['f'].pop())**2)/2)
    # bool operations
        '!': lambda n,features,stack,labels: tf.logical_not(stack['b'].pop()),
        '&': lambda n,features,stack,labels: tf.logical_and(stack['b'].pop(), stack['b'].pop()),
        '|': lambda n,features,stack,labels: tf.logical_or(stack['b'].pop(), stack['b'].pop()),
        '==': lambda n,features,stack,labels: tf.equal(stack['b'].pop(),stack['b'].pop()),
        '>_f': lambda n,features,stack,labels: tf.greater(stack['f'].pop(), stack['f'].pop()),
        '<_f': lambda n,features,stack,labels: tf.less(stack['f'].pop(), stack['f'].pop()),
        '>=_f': lambda n,features,stack,labels: tf.greater_equal(stack['f'].pop(),stack['f'].pop()),
        '<=_f': lambda n,features,stack,labels: tf.less_equal(stack['f'].pop(), stack['f'].pop()),
        '>_b': lambda n,features,stack,labels: tf.greater(stack['b'].pop(),stack['b'].pop()),
        '<_b': lambda n,features,stack,labels: tf.less(stack['b'].pop(),stack['b'].pop()),
        '>=_b': lambda n,features,stack,labels: tf.greater_equal(stack['b'].pop(), stack['b'].pop()),
        '<=_b': lambda n,features,stack,labels: tf.less_equal(stack['b'].pop(), stack['b'].pop()),
        'xor_b': lambda n,features,stack,labels: tf.logical_xor(stack['b'].pop(),stack['b'].pop()),
        'xor_f': lambda n,features,stack,labels: tf.logical_xor(stack['f'].pop().astype(bool), stack['f'].pop().astype(bool)),
        }
    ######################################################## tensor flow methods
    def build_tf_graph(self, I, features, labels=None):
        """evaluate node in program"""
        np.seterr(all='ignore')
        stack={'f':[],'b':[]}
        feed_dict={}
        for n in I.stack:
            if (len(stack['f']) >= n.arity['f']
                and len(stack['b']) >= n.arity['b']):
                stack[n.out_type].append(
                                self.tf_dict[n.name](n,features,stack,labels))
            if n.name=='x':#add x to feed_dict
                if stack[n.out_type][-1] not in feed_dict.keys():
                    feed_dict[stack[n.out_type][-1]] = features[:,n.loc]

        return stack[self.otype][-1],feed_dict

    def out_tf(self,I,features,labels=None,otype='f'):
        """computes the output for individual I"""

        # Initialize TensorFlow session
        tf.reset_default_graph() # Reset TF internal state and cache
        config = tf.ConfigProto(log_device_placement=False,
                                allow_soft_placement=True)
        config.gpu_options.allow_growth = False
        # print("stack:",I.stack)
        # evaulate stack over rows of features,labels
        # pdb.set_trace()
        program,feed_dict = self.build_tf_graph(I,features,labels)

        with tf.Session(config=config) as sess:
            with sess.graph.device('/gpu:0'):
                result = np.array(sess.run(program,feed_dict=feed_dict))
            # tf.train.write_graph(sess.graph,'tmp','graph.txt',as_text=True)
        # pdb.set_trace()
        return (result if self.all_finite(result)
                 else np.zeros(len(features)))


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

    def evaluate(self,n, features, stack,labels=None):
        """evaluate node in program"""
        np.seterr(all='ignore')
        if len(stack['f']) >= n.arity['f'] and len(stack['b']) >= n.arity['b']:
            stack[n.out_type].append(
                    self.safe(self.eval_dict[n.name](n,features,stack,labels)))
            if (np.isnan(stack[n.out_type][-1]).any() or
                np.isinf(stack[n.out_type][-1]).any()):
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
        stack = {'f':[],'b':[]}
        # evaulate stack over rows of features,labels
        for n in I.stack:
            self.evaluate(n,features,stack,labels)
            # print("stack_float:",stack_float)

        return (stack[self.otype][-1] if self.all_finite(stack[self.otype][-1])
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
