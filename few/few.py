# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava

license: GNU/GPLv3

"""

import argparse
from ._version import __version__
from .evaluation import EvaluationMixin
from .population import *
from .variation import VariationMixin
from .selection import SurvivalMixin

from sklearn.base import BaseEstimator
from sklearn.linear_model import LassoLarsCV, LogisticRegression, SGDClassifier
from sklearn.svm import SVR, LinearSVR, SVC, LinearSVC
from sklearn.tree import DecisionTreeRegressor, DecisionTreeClassifier
from sklearn.tree import export_graphviz
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import Imputer
from DistanceClassifier import DistanceClassifier
import numpy as np
import pandas as pd
import warnings
import copy
import itertools as it
import pdb

# from update_checker import update_check
# from joblib import Parallel, delayed
#from sklearn.externals.joblib import Parallel, delayed
from tqdm import tqdm
import uuid

# from profilehooks import profile
# import multiprocessing as mp
# NUM_THREADS = mp.cpu_count()



class FEW(SurvivalMixin, VariationMixin, EvaluationMixin, BaseEstimator):
    """FEW uses GP to find a set of transformations from the original feature
    space that produces the best performance for a given machine learner.
    """
    update_checked = False

    def __init__(self, population_size=50, generations=100,
                 mutation_rate=0.5, crossover_rate=0.5,
                 ml = None, min_depth = 1, max_depth = 2, max_depth_init = 2,
                 sel = 'epsilon_lexicase', tourn_size = 2, fit_choice = None,
                 op_weight = False, max_stall=10, seed_with_ml = True, erc = False,
                 random_state=np.random.randint(9999999), verbosity=0,
                 scoring_function=None, disable_update_check=False,
                 elitism=True, boolean = False,classification=False,clean=False,
                 track_diversity=False,mdr=False,otype='f',c=True):
                # sets up GP.

        # Save params to be recalled later by get_params()
        self.params = locals()  # placed before any local variable definitions
        self.params.pop('self')

        # # Do not prompt the user to update during this session if they
        # ever disabled the update check
        # if disable_update_check:
        #     FEW.update_checked = True
        #
        # # Prompt the user if their version is out of date
        # if not disable_update_check and not FEW.update_checked:
        #     update_check('FEW', __version__)
        #     FEW.update_checked = True

        self._best_estimator = None
        self._training_features = None
        self._training_labels = None
        self._best_inds = None

        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_depth_init = max_depth_init
        self.sel = sel
        self.tourn_size = tourn_size
        self.fit_choice = fit_choice
        self.op_weight = op_weight
        self.max_stall = max_stall
        self.seed_with_ml = seed_with_ml
        self.erc = erc
        self.random_state = random_state
        self.verbosity = verbosity
        self.scoring_function = scoring_function
        self.gp_generation = 0
        self.elitism = elitism
        self.max_fit = 99999999.666
        self.boolean = boolean
        self.classification = classification
        self.clean = clean
        self.ml = ml
        self.ml_type = type(self.ml).__name__
        self.track_diversity = track_diversity
        self.mdr = mdr
        self.otype = otype

        # if otype is b, boolean functions must be turned on
        if self.otype=='b':
            self.boolean = True

        # instantiate sklearn estimator according to specified machine learner
        if self.ml is None:
            if self.classification:
                self.ml = LogisticRegression(solver='sag')
            else:
                self.ml = LassoLarsCV()
        if not self.scoring_function:
            if self.classification:
                self.scoring_function = accuracy_score
            else:
                self.scoring_function = r2_score

        # set default fitness metrics for various learners
        if not self.fit_choice:
            self.fit_choice =  {
                            #regression
                            type(LassoLarsCV()): 'mse',
                            type(SVR()): 'mae',
                            type(LinearSVR()): 'mae',
                            type(KNeighborsRegressor()): 'mse',
                            type(DecisionTreeRegressor()): 'mse',
                            type(RandomForestRegressor()): 'mse',
                            #classification
                            type(SGDClassifier()): 'r2',
                            type(LogisticRegression()): 'r2',
                            type(SVC()): 'r2',
                            type(LinearSVC()): 'r2',
                            type(RandomForestClassifier()): 'r2',
                            type(DecisionTreeClassifier()): 'r2',
                            type(DistanceClassifier()): 'silhouette',
                            type(KNeighborsClassifier()): 'r2',
            }[type(self.ml)]


        # Columns to always ignore when in an operator
        self.non_feature_columns = ['label', 'group', 'guess']

        # function set
        self.func_set = [node('+'), node('-'), node('*'), node('/'),
                         node('sin'), node('cos'), node('exp'), node('log'),
                         node('^2'), node('^3'), node('sqrt')]

        # terminal set
        self.term_set = []
        # diversity
        self.diversity = []
        # use cython
        self.c = c
    # @profile
    def fit(self, features, labels):
        """Fit model to data"""

        np.random.seed(self.random_state)
        # setup data
        # imputation
        if self.clean:
            features = self.impute_data(features)
        # Train-test split routine for internal validation
        ####
        train_val_data = pd.DataFrame(data=features)
        train_val_data['labels'] = labels
        # print("train val data:",train_val_data[::10])
        new_col_names = {}
        for column in train_val_data.columns.values:
            if type(column) != str:
                new_col_names[column] = str(column).zfill(10)
        train_val_data.rename(columns=new_col_names, inplace=True)
        # internal training/validation split
        train_i, val_i = train_test_split(train_val_data.index,
                                          stratify=None,
                                          train_size=0.75,
                                          test_size=0.25)

        x_t = train_val_data.loc[train_i].drop('labels',axis=1).values
        x_v = train_val_data.loc[val_i].drop('labels',axis=1).values
        y_t = train_val_data.loc[train_i, 'labels'].values
        y_v = train_val_data.loc[val_i, 'labels'].values

        # Store the training features and classes for later use
        self._training_features = x_t
        self._training_labels = y_t
        ####

        # set population size
        if type(self.population_size) is str:
            if 'x' in self.population_size: # set pop size prop to features
                self.population_size = int(
                        float(self.population_size[:-1])*features.shape[1])
            else:
                self.population_size = int(self.population_size)

        if self.verbosity >0: print("population size:",self.population_size)
        # print few settings
        if self.verbosity > 1:
            for arg in self.get_params():
                print('{}\t=\t{}'.format(arg, self.get_params()[arg]))
            print('')

        # initial model
        initial_estimator = copy.deepcopy(self.ml.fit(x_t,y_t))
        # self._best_estimator = copy.deepcopy(self.ml.fit(x_t,y_t))

        self._best_score = self.ml.score(x_v,y_v)
        initial_score = self._best_score
        if self.verbosity > 2:
            print("initial estimator size:",self.ml.coef_.shape)
        if self.verbosity > 0:
            print("initial ML CV: {:1.3f}".format(self._best_score))

        # create terminal set
        for i in np.arange(x_t.shape[1]):
            # dictionary of node name, arity, feature column index, output type
            # and input type
            self.term_set.append(node('x',loc=i)) # features
            # add ephemeral random constants if flag
            if self.erc: # ephemeral random constants
                self.term_set.append(node('k',value=np.random.rand()))

        # edit function set if boolean
        if self.boolean or self.otype=='b': # include boolean functions
            self.func_set += [node('!'), node('&'), node('|'), node('=='),
                        node('>_f'), node('<_f'), node('>=_f'), node('<=_f'),
                        node('>_b'), node('<_b'), node('>=_b'), node('<=_b'),
                        node('xor_b'), node('xor_f')]

        # add mdr if specified
        if self.mdr:
            self.func_set += [node('mdr2')]

        # Create initial population
        # for now, force seed_with_ml to be off if otype is 'b', since data
        # types are assumed to be float
        if self.otype=='b':
            self.seed_with_ml = False
        self.pop = self.init_pop(self._training_features.shape[0])
        # check that uuids are unique in population
        uuids = [p.id for p in self.pop.individuals]
        if len(uuids) != len(set(uuids)):
            pdb.set_trace()
        # Evaluate the entire population
        # X represents a matrix of the population outputs (number of samples x
        # population size)
        # single thread
        self.X = self.transform(x_t,self.pop.individuals,y_t).transpose()
        # pdb.set_trace()
        # parallel:
        # X = np.asarray(Parallel(n_jobs=-1)(
        # delayed(out)(I,x_t,self.otype,y_t) for I in self.pop.individuals),
        # order = 'F')

        # calculate fitness of individuals
        # fitnesses = list(map(lambda I: fitness(I,y_t,self.ml),X))
        self.F = self.calc_fitness(self.X,y_t,self.fit_choice,self.sel)

        #with Parallel(n_jobs=10) as parallel:
        ####################
        ### Main GP loop
        self.diversity=[]
        # progress bar
        pbar = tqdm(total=self.generations,disable = self.verbosity==0,
                    desc='Internal CV: {:1.3f}'.format(self._best_score))
        stall_count = 0
        # for each generation g
        for g in np.arange(self.generations):
            if stall_count == self.max_stall:
                if self.verbosity > 0: print('max stall count reached.')
                break

            if self.track_diversity:
                self.get_diversity(self.X)

            if self.verbosity > 1:
                print(".",end='')
            if self.verbosity > 1:
                print(str(g)+".)",end='')
            # if self.verbosity > 1:
            #   print("population:",stacks_2_eqns(self.pop.individuals))
            if self.verbosity > 2:
                print("pop fitnesses:", ["%0.2f" % np.mean(f) for f in self.F])
            if self.verbosity > 1:
                print("median fitness pop: %0.2f" % np.median(
                        [np.mean(f) for f in self.F]))
            if self.verbosity > 1:
                print("best fitness pop: %0.2f" % np.min(
                    [np.mean(f) for f in self.F]))
            if self.verbosity > 1 and self.track_diversity:
                print("feature diversity: %0.2f" % self.diversity[-1])
            if self.verbosity > 1: print("ml fitting...")
            # fit ml model
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    # if len(self.valid_loc(self.F)) > 0:
                    if self.valid_loc():
                        self.ml.fit(self.X[self.valid_loc(),:].transpose(),y_t)
                    # else:
                    #     self.ml.fit(X.transpose(),y_t)

                except ValueError as detail:
                    # pdb.set_trace()
                    print("warning: ValueError in ml fit. X.shape:",
                          self.X[:,self.valid_loc()].transpose().shape,
                          "y_t shape:",y_t.shape)
                    print("First ten entries X:",
                          self.X[self.valid_loc(),:].transpose()[:10])
                    print("First ten entries y_t:",y_t[:10])
                    print("equations:",stacks_2_eqns(self.pop.individuals))
                    print("FEW parameters:",self.get_params())
                    if self.verbosity > 1: print("---\ndetailed error message:",
                                                 detail)
                    raise(ValueError)

            # if self.verbosity > 1:
            #   print("number of non-zero regressors:",self.ml.coef_.shape[0])
            # keep best model
            tmp_score = 0
            try:
                # if len(self.valid_loc(F)) > 0:
                if self.valid_loc():
                    tmp_score = self.ml.score(self.transform(
                                    x_v,self.pop.individuals)[:,self.valid_loc()],
                                    y_v)
                # else:
                #     tmp_score = 0
                #     tmp = self.ml.score(self.transform(x_v,
                #                                         self.pop.individuals),
                #                         y_v)
            except Exception as detail:
                if self.verbosity > 1: print(detail)

            if self.verbosity > 1:
                print("current ml validation score:",tmp_score)


            if self.valid_loc() and tmp_score > self._best_score:
                self._best_estimator = copy.deepcopy(self.ml)
                self._best_score = tmp_score
                stall_count = 0;
                self._best_inds = copy.deepcopy(self.valid())
                if self.verbosity > 1:
                    print("updated best internal CV:",self._best_score)
            else:
                stall_count = stall_count + 1

            # Variation
            if self.verbosity > 2:
                print("variation...")
            offspring,elite,elite_index = self.variation(self.pop.individuals)

            # evaluate offspring
            if self.verbosity > 2:
                print("output...")
            X_offspring = self.transform(x_t,offspring).transpose()
            #parallel:
            # X_offspring = np.asarray(Parallel(n_jobs=-1)(delayed(out)(O,x_t,y_t,self.otype) for O in offspring), order = 'F')
            if self.verbosity > 2:
                print("fitness...")
            F_offspring = self.calc_fitness(X_offspring,
                                            y_t,self.fit_choice,self.sel)
            # F_offspring = parallel(delayed(f[self.fit_choice])(y_t,yhat) for yhat in X_offspring)

            # Survival
            if self.verbosity > 2: print("survival..")
            survivors,survivor_index = self.survival(self.pop.individuals,
                                                     offspring,elite,
                                                     elite_index,F=self.F,
                                                     F_offspring=F_offspring)
            # set survivors
            self.pop.individuals[:] = survivors
            self.X = np.vstack((self.X, X_offspring))[survivor_index]
            if 'lexicase' in self.sel:
                self.F = np.asarray(
                        np.vstack((self.F, F_offspring))[survivor_index],
                        order='F')
            else:
                self.F = np.asarray(
                        np.hstack((self.F,F_offspring))[survivor_index],
                        order='F')

            if self.verbosity > 2:
                print("median fitness survivors: %0.2f" % np.median(
                        [np.mean(f) for f in self.F]))
            if self.verbosity>2:
                print("best features:",
                      stacks_2_eqns(self._best_inds) if self._best_inds
                      else 'original')
            pbar.set_description('Internal CV: {:1.3f}'.format(self._best_score))
            pbar.update(1)
        # end of main GP loop
            ####################
        if self.verbosity > 0: print('finished. best internal val score:'
                                     ' {:1.3f}'.format(self._best_score))
        if self.verbosity > 0: print("final model:\n",self.print_model())
        if not self._best_estimator:
            self._best_estimator = initial_estimator
        return self

    def transform(self,x,inds=None,labels = None):
        """return a transformation of x using population outputs"""
        if inds:
            # return np.asarray(Parallel(n_jobs=10)(delayed(self.out)(I,x,labels,self.otype) for I in inds)).transpose()
            return np.asarray(
                [self.out(I,x,labels,self.otype) for I in inds]).transpose()
        else:
            # return np.asarray(Parallel(n_jobs=10)(delayed(self.out)(I,x,labels,self.otype) for I in self._best_inds)).transpose()
            return np.asarray(
                [self.out(I,x,labels,self.otype) for I in self._best_inds]).transpose()


    def impute_data(self,x):
        """Imputes data set containing Nan values"""
        imp = Imputer(missing_values='NaN', strategy='mean', axis=0)
        return imp.fit_transform(x)

    def clean(self,x):
        """remove nan and inf rows from x"""
        return x[~np.any(np.isnan(x) | np.isinf(x),axis=1)]

    def clean_with_zeros(self,x):
        """ set nan and inf rows from x to zero"""
        x[~np.any(np.isnan(x) | np.isinf(x),axis=1)] = 0
        return x

    def predict(self, testing_features):
        """predict on a holdout data set."""
        # print("best_inds:",self._best_inds)
        # print("best estimator size:",self._best_estimator.coef_.shape)
        if self.clean:
            testing_features = self.impute_data(testing_features)

        if self._best_inds:
            X_transform = self.transform(testing_features)
            try:
                return self._best_estimator.predict(self.transform(testing_features))
            except ValueError as detail:
                pdb.set_trace()
                print('shape of X:',testing_features.shape)
                print('shape of X_transform:',X_transform.transpose().shape)
                print('best inds:',stacks_2_eqns(self._best_inds))
                print('valid locs:',self.valid_loc(self._best_inds))
                raise ValueError(detail)
        else:
            return self._best_estimator.predict(testing_features)

    def fit_predict(self, features, labels):
        """Convenience function that fits a pipeline then predicts on the
        provided features

        Parameters
        ----------
        features: array-like {n_samples, n_features}
            Feature matrix
        labels: array-like {n_samples}
            List of class labels for prediction

        Returns
        ----------
        array-like: {n_samples}
            Predicted labels for the provided features

        """
        self.fit(features, labels)
        return self.predict(features)

    def score(self, testing_features, testing_labels):
        """estimates accuracy on testing set"""
        # print("test features shape:",testing_features.shape)
        # print("testing labels shape:",testing_labels.shape)
        yhat = self.predict(testing_features)
        return self.scoring_function(testing_labels,yhat)

    def export(self, output_file_name):
        """exports engineered features

        Parameters
        ----------
        output_file_name: string
            String containing the path and file name of the desired output file

        Returns
        -------
        None

        """
        if self._best_estimator is None:
            raise ValueError('A model has not been optimized. Please call fit()'
                             ' first.')

        # Write print_model() to file
        with open(output_file_name, 'w') as output_file:
            output_file.write(self.print_model())
        # if decision tree, print tree into dot file
        if 'DecisionTree' in type(self.ml).__name__:
            export_graphviz(self._best_estimator,
                            out_file=output_file_name+'.dot',
                            feature_names = stacks_2_eqns(self._best_inds)
                            if self._best_inds else None,
                            class_names=['True','False'],
                            filled=False,impurity = True,rotate=True)

    def init_pop(self,num_features=1):
        """initializes population of features as GP stacks."""
        pop = Pop(self.population_size,num_features)
        # make programs
        if self.seed_with_ml:
            # initial population is the components of the default ml model
            if type(self.ml) == type(LassoLarsCV()):
                # add all model components with non-zero coefficients
                for i,(c,p) in enumerate(it.zip_longest(
                        [c for c in self.ml.coef_ if c !=0],pop.individuals,
                        fillvalue=None)):
                    if c is not None and p is not None:
                        p.stack = [node('x',loc=i)]
                    elif p is not None:
                        # make program if pop is bigger than model componennts
                        make_program(p.stack,self.func_set,self.term_set,
                                     np.random.randint(self.min_depth,
                                                       self.max_depth+1),
                                     self.otype)
                        p.stack = list(reversed(p.stack))
            else: # seed with raw features
                # if list(self.ml.coef_):
                #pdb.set_trace()
                try:
                    if self.population_size < self.ml.coef_.shape[0]:
                        # seed pop with highest coefficients
                        coef_order = np.argsort(self.ml.coef_[::-1])
                        for i,(c,p) in enumerate(zip(coef_order,pop.individuals)):
                            p.stack = [node('x',loc=i)]
                    else:
                        raise(AttributeError)
                except Exception: # seed pop with raw features
                     for i,p in it.zip_longest(
                         range(self._training_features.shape[1]),
                         pop.individuals,fillvalue=None):
                        if p is not None:
                            if i is not None:
                                p.stack = [node('x',loc=i)]
                            else:
                                make_program(p.stack,self.func_set,self.term_set,
                                             np.random.randint(self.min_depth,
                                                               self.max_depth+1),
                                             self.otype)
                                p.stack = list(reversed(p.stack))

            # print initial population
            if self.verbosity > 2:
                print("seeded initial population:",
                      stacks_2_eqns(pop.individuals))


        else:
            for I in pop.individuals:
                depth = np.random.randint(self.min_depth,self.max_depth+1)
                # print("hex(id(I)):",hex(id(I)))
                # depth = 2;
                # print("initial I.stack:",I.stack)

                make_program(I.stack,self.func_set,self.term_set,depth,
                             self.otype)
                # print(I.stack)
                I.stack = list(reversed(I.stack))

            # print(I.stack)

        return pop

    def print_model(self,sep='\n'):
        """prints model contained in best inds, if ml has a coefficient property.
        otherwise, prints the features generated by FEW."""
        model = ''
        # print('ml type:',type(self.ml).__name__)
        # print('ml:',self._best_estimator)

        if self._best_inds:

            if self.ml_type == 'GridSearchCV':
                ml = self._best_estimator.best_estimator_
            else:
                ml = self._best_estimator

            if self.ml_type != 'SVC' and self.ml_type != 'SVR':
            # this is need because svm has a bug that throws valueerror on attribute check:

                if hasattr(ml,'coef_'):
                    if ml.coef_.shape[0]==1 or len(ml.coef_.shape)==1:
                        if ml.coef_.shape[0]==1:
                            s = np.argsort(np.abs(ml.coef_[0]))[::-1]
                            scoef = ml.coef_[0][s]
                        else:
                            s = np.argsort(np.abs(ml.coef_))[::-1]
                            scoef = ml.coef_[s]
                        bi = [self._best_inds[k] for k in s]
                        model = (' +' + sep).join(
                            [str(round(c,3))+'*'+stack_2_eqn(f)
                             for i,(f,c) in enumerate(zip(bi,scoef))
                             if round(scoef[i],3) != 0])
                    else:
                        # more than one decision function is fit. print all.
                        for j,coef in enumerate(ml.coef_):
                            s = np.argsort(np.abs(coef))[::-1]
                            scoef = coef[s]
                            bi =[self._best_inds[k] for k in s]
                            model += sep + 'class'+str(j)+' :'+' + '.join(
                                [str(round(c,3))+'*'+stack_2_eqn(f)
                                 for i,(f,c) in enumerate(zip(bi,coef))
                                 if coef[i] != 0])
                elif hasattr(ml,'feature_importances_'):
                    s = np.argsort(ml.feature_importances_)[::-1]
                    sfi = ml.feature_importances_[s]
                    bi = [self._best_inds[k] for k in s]
                    # model = 'importance:feature'+sep

                    model += sep.join(
                        [str(round(c,3))+':'+stack_2_eqn(f)
                         for i,(f,c) in enumerate(zip(bi,sfi))
                         if round(sfi[i],3) != 0])
                else:
                    return sep.join(stacks_2_eqns(self._best_inds))
            else:
                return sep.join(stacks_2_eqns(self._best_inds))
        else:
            return 'original features'

        return model

    def representation(self):
        """return stacks_2_eqns output"""
        return stacks_2_eqns(self._best_inds)

    def valid_loc(self,F=None):
        """returns the indices of individuals with valid fitness."""
        if F is not None:
            return [i for i,f in enumerate(F) if np.all(f < self.max_fit) and np.all(f >= 0)]
        else:
            return [i for i,f in enumerate(self.F) if np.all(f < self.max_fit) and np.all(f >= 0)]

    def valid(self,individuals=None,F=None):
        """returns the sublist of individuals with valid fitness."""
        if F:
            valid_locs = self.valid_loc(F)
        else:
            valid_locs = self.valid_loc(self.F)

        if individuals:
            return [ind for i,ind in enumerate(individuals) if i in valid_locs]
        else:
            return [ind for i,ind in enumerate(self.pop.individuals) if i in valid_locs]

    def get_params(self, deep=None):
        """Get parameters for this estimator

        This function is necessary for FEW to work as a drop-in feature
        constructor in, e.g., sklearn.model_selection.cross_val_score

        Parameters
        ----------
        deep: unused
            Only implemented to maintain interface for sklearn

        Returns
        -------
        params: mapping of string to any
            Parameter names mapped to their values
        """
        return self.params

    def get_diversity(self,X):
        """compute mean diversity of individual outputs"""
        # diversity in terms of cosine distances between features
        feature_correlations = np.zeros(X.shape[0]-1)
        for i in np.arange(1,X.shape[0]-1):
            feature_correlations[i] = max(0.0,r2_score(X[0],X[i]))
        # pdb.set_trace()
        self.diversity.append(1-np.mean(feature_correlations))

def positive_integer(value):
    """Ensures that the provided value is a positive integer;
    throws an exception otherwise

    Parameters
    ----------
    value: int
        The number to evaluate

    Returns
    -------
    value: int
        Returns a positive integer
    """
    try:
        value = int(value)
    except Exception:
        raise argparse.ArgumentTypeError(
            'Invalid int value: \'{}\''.format(value))
    if value < 0:
        raise argparse.ArgumentTypeError(
            'Invalid positive int value: \'{}\''.format(value))
    return value

def float_range(value):
    """Ensures that the provided value is a float integer in the range (0., 1.)
    throws an exception otherwise

    Parameters
    ----------
    value: float
        The number to evaluate

    Returns
    -------
    value: float
        Returns a float in the range (0., 1.)
    """
    try:
        value = float(value)
    except:
        raise argparse.ArgumentTypeError(
            'Invalid float value: \'{}\''.format(value))
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError(
            'Invalid float value: \'{}\''.format(value))
    return value

# dictionary of ml options
ml_dict = {
        'lasso': LassoLarsCV(),
        'svr': SVR(),
        'lsvr': LinearSVR(),
        'lr': LogisticRegression(solver='sag'),
        'sgd': SGDClassifier(loss='log',penalty='l1'),
        'svc': SVC(),
        'lsvc': LinearSVC(),
        'rfc': RandomForestClassifier(),
        'rfr': RandomForestRegressor(),
        'dtc': DecisionTreeClassifier(),
        'dtr': DecisionTreeRegressor(),
        'dc': DistanceClassifier(),
        'knc': KNeighborsClassifier(),
        'knr': KNeighborsRegressor(),
        None: None
}
# main functions
def main():
    """Main function that is called when FEW is run on the command line"""
    parser = argparse.ArgumentParser(description='A feature engineering wrapper'
                                     ' for machine learning algorithms.',
                                     add_help=False)

    parser.add_argument('INPUT_FILE', type=str,
                        help='Data file to run FEW on; ensure that the '
                        'target/label column is labeled as "label" or "class".')

    parser.add_argument('-h', '--help', action='help',
                        help='Show this help message and exit.')

    parser.add_argument('-is', action='store',dest='INPUT_SEPARATOR',
                        default=None,type=str,
                        help='Character separating columns in the input file.')

    parser.add_argument('-o', action='store', dest='OUTPUT_FILE', default='',
                        type=str, help='File to export the final model.')

    parser.add_argument('-g', action='store', dest='GENERATIONS', default=100,
                        type=positive_integer,
                        help='Number of generations to run FEW.')

    parser.add_argument('-p', action='store', dest='POPULATION_SIZE',default=50,
                         help='Number of individuals in the GP population. '
                         'Follow the number with x to set population size as a'
                         'multiple of raw feature size.')

    parser.add_argument('-mr', action='store', dest='MUTATION_RATE',default=0.5,
                        type=float_range,
                        help='GP mutation rate in the range [0.0, 1.0].')

    parser.add_argument('-xr', action='store', dest='CROSSOVER_RATE',
                        default=0.5,type=float_range,
                        help='GP crossover rate in the range [0.0, 1.0].')

    parser.add_argument('-ml', action='store', dest='MACHINE_LEARNER',
                        default=None,
                        choices = ['lasso','svr','lsvr','lr','svc','rfc','rfr',
                                   'dtc','dtr','dc','knc','knr','sgd'],
                        type=str, help='ML algorithm to pair with features. '
                        'Default: Lasso (regression), LogisticRegression '
                        '(classification)')

    parser.add_argument('-min_depth', action='store', dest='MIN_DEPTH',
                        default=1,type=positive_integer,
                        help='Minimum length of GP programs.')

    parser.add_argument('-max_depth', action='store', dest='MAX_DEPTH',
                        default=2,type=positive_integer,
                        help='Maximum number of nodes in GP programs.')

    parser.add_argument('-max_depth_init', action='store',dest='MAX_DEPTH_INIT',
                        default=2,type=positive_integer,
                        help='Maximum nodes in initial programs.')

    parser.add_argument('-op_weight', action='store',dest='OP_WEIGHT',default=1,
                        type=bool, help='Weight attributes for incuded in'
                        ' features based on ML scores. Default: off')

    parser.add_argument('-ms', action='store', dest='MAX_STALL',default=10,
                        type=positive_integer, help='If model CV does not '
                        'improve for this many generations, end optimization.')

    parser.add_argument('-sel', action='store', dest='SEL',
                        default='epsilon_lexicase',
                        choices = ['tournament','lexicase','epsilon_lexicase',
                                   'deterministic_crowding','random'],
                        type=str, help='Selection method (Default: tournament)')

    parser.add_argument('-tourn_size', action='store', dest='TOURN_SIZE',
                        default=2, type=positive_integer,
                        help='Tournament size (Default: 2)')

    parser.add_argument('-fit', action='store', dest='FIT_CHOICE', default=None,
                        choices = ['mse','mae','r2','vaf','mse_rel','mae_rel',
                                   'r2_rel','vaf_rel','silhouette','inertia',
                                   'separation','fisher','random','relief'],
                        type=str,
                        help='Fitness metric (Default: dependent on ml used)')

    parser.add_argument('--no_seed', action='store_false', dest='SEED_WITH_ML',
                        default=True,
                        help='Turn off initial GP population seeding.')

    parser.add_argument('--elitism', action='store_true', dest='ELITISM',
                        default=False,
                        help='Force survival of best feature in GP population.')

    parser.add_argument('--erc', action='store_true', dest='ERC', default=False,
                    help='Use random constants in GP feature construction.')

    parser.add_argument('--bool', action='store_true', dest='BOOLEAN',
                        default=False,
                        help='Include boolean operators in features.')

    parser.add_argument('-otype', action='store', dest='OTYPE', default='f',
                        choices=['f','b'],
                        type=str,
                        help='Feature output type. f: float, b: boolean.')

    parser.add_argument('--class', action='store_true', dest='CLASSIFICATION',
                        default=False,
                        help='Conduct classification rather than regression.')

    parser.add_argument('--mdr', action='store_true',dest='MDR',default=False,
                        help='Use MDR nodes.')

    parser.add_argument('--diversity', action='store_true',
                        dest='TRACK_DIVERSITY', default=False,
                        help='Store diversity of feature transforms each gen.')

    parser.add_argument('--clean', action='store_true', dest='CLEAN',
                        default=False,
                        help='Clean input data of missing values.')

    parser.add_argument('--no_lib', action='store_false', dest='c',
                        default=True,
                        help='Don''t use optimized c libraries.')

    parser.add_argument('-s', action='store', dest='RANDOM_STATE',
                        default=np.random.randint(4294967295),
                        type=int,
                        help='Random number generator seed for reproducibility.'
                        'Note that using multi-threading may make exact results'
                        ' impossible to reproduce.')

    parser.add_argument('-v', action='store', dest='VERBOSITY', default=1,
                        choices=[0, 1, 2, 3], type=int,
                        help='How much information FEW communicates while it is'
                        ' running: 0 = none, 1 = minimal, 2 = lots, 3 = all.')

    parser.add_argument('--no-update-check', action='store_true',
                        dest='DISABLE_UPDATE_CHECK', default=False,
                        help='Don''t check the FEW version.')

    parser.add_argument('--version', action='version',
                        version='FEW {version}'.format(version=__version__),
                        help='Show FEW\'s version number and exit.')

    args = parser.parse_args()

    # if args.VERBOSITY >= 2:
    #     print('\nFEW settings:')
    #     for arg in sorted(args.__dict__):
    #         if arg == 'DISABLE_UPDATE_CHECK':
    #             continue
    #         print('{}\t=\t{}'.format(arg, args.__dict__[arg]))
    #     print('')

    # load data from csv file
    if args.INPUT_SEPARATOR is None:
        input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR,
                                 engine='python')
    else: # use c engine for read_csv is separator is specified
        input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR)

    # if 'Label' in input_data.columns.values:
    input_data.rename(columns={'Label': 'label','Class':'label','class':'label',
                               'target':'label'}, inplace=True)

    RANDOM_STATE = args.RANDOM_STATE if args.RANDOM_STATE > 0 else None

    train_i, test_i = train_test_split(input_data.index,
                                       stratify = None,
                                       #stratify=input_data['label'].values,
                                       train_size=0.75,
                                       test_size=0.25,
                                       random_state=RANDOM_STATE)

    training_features = input_data.loc[train_i].drop('label', axis=1).values
    training_labels = input_data.loc[train_i, 'label'].values

    testing_features = input_data.loc[test_i].drop('label', axis=1).values
    testing_labels = input_data.loc[test_i, 'label'].values

    learner = FEW(generations=args.GENERATIONS,
                  population_size=args.POPULATION_SIZE,
                  mutation_rate=args.MUTATION_RATE,
                  crossover_rate=args.CROSSOVER_RATE,
                  ml = ml_dict[args.MACHINE_LEARNER],
                  min_depth = args.MIN_DEPTH,max_depth = args.MAX_DEPTH,
                  sel = args.SEL, tourn_size = args.TOURN_SIZE,
                  seed_with_ml = args.SEED_WITH_ML, op_weight = args.OP_WEIGHT,
                  max_stall = args.MAX_STALL,
                  erc = args.ERC, random_state=args.RANDOM_STATE,
                  verbosity=args.VERBOSITY,
                  disable_update_check=args.DISABLE_UPDATE_CHECK,
                  fit_choice = args.FIT_CHOICE,boolean=args.BOOLEAN,
                  classification=args.CLASSIFICATION,clean = args.CLEAN,
                  track_diversity=args.TRACK_DIVERSITY,mdr=args.MDR,
                  otype=args.OTYPE,c=args.c)

    learner.fit(training_features, training_labels)
    # pdb.set_trace()
    if args.VERBOSITY >= 1:
        print('\nTraining accuracy: {:1.3f}'.format(
            learner.score(training_features, training_labels)))
        print('Test accuracy: {:1.3f}'.format(
            learner.score(testing_features, testing_labels)))

    if args.OUTPUT_FILE != '':
        learner.export(args.OUTPUT_FILE)


if __name__ == '__main__':
    main()
