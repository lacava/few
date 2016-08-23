"""
Copyright 2016 William La Cava

This file is part of the FEW library.

The FEW library is free software: you can redistribute it and/or
modify it under the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or (at your option)
any later version.

The FEW library is distributed in the hope that it will be useful, but
WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for more details.
You should have received a copy of the GNU General Public License along with
the FEW library. If not, see http://www.gnu.org/licenses/.

"""

import argparse
from ._version import __version__
from .evaluation import out, calc_fitness
from .population import *
from .variation import *
from .selection import *

from sklearn.linear_model import LassoLarsCV
from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import warnings
import copy
import itertools as it
import pdb
# import multiprocessing as mp
# NUM_THREADS = mp.cpu_count()

class FEW(object):
    """FEW uses GP to find a set of transformations from the original feature space
    that produces the best performance for a given machine learner.
    """
    def __init__(self, population_size=100, generations=100,
                 mutation_rate=0.2, crossover_rate=0.8,
                 machine_learner = 'lasso', min_depth = 1, max_depth = 5, max_depth_init = 5,
                 sel = 'tournament', tourn_size = 2, fit_choice = 'mse', op_weight = False,
                 seed_with_ml = False, erc = False, random_state=0, verbosity=0, scoring_function=None,
                 disable_update_check=False):
                # sets up GP.

        # Save params to be recalled later by get_params()
        self.params = locals()  # Must be placed before any local variable definitions
        # self.params.pop('self')

        # Do not prompt the user to update during this session if they ever disabled the update check
        # if disable_update_check:
        #     FEW.update_checked = True

        # Prompt the user if their version is out of date
        # if not disable_update_check and not FEW.update_checked:
        #     update_check('FEW', __version__)
        #     FEW.update_checked = True

        self._best_estimator = None
        self._training_features = None
        self._training_labels = None
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.machine_learner = machine_learner
        self.verbosity = verbosity
        self.gp_generation = 0
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.max_depth_init = max_depth_init
        self._best_inds = None
        self._fit_choice = fit_choice
        self._op_weight = False
        self.seed_with_ml = seed_with_ml
        self.erc = erc
        # self.op_weight = op_weight
        self.sel = sel
        if "lexicase" in sel and ("_vec" not in fit_choice or "_rel" not in fit_choice):
            self._fit_choice += "_vec"

        self.tourn_size = tourn_size
        # instantiate sklearn estimator according to specified machine learner
        if (self.machine_learner.lower() == "lasso"):
            self.ml = LassoLarsCV(n_jobs=-1)
        elif (self.machine_learner.lower() == "distance"):
            self.ml = DistanceClassifier()
        else:
            self.ml = LassoLarsCV(n_jobs=-1)

        # Columns to always ignore when in an operator
        self.non_feature_columns = ['label', 'group', 'guess']

        # function set
        self.func_set = [('+',2),('-',2),('*',2),('/',2),('sin',1),('cos',1),('exp',1),('log',1)]
        # terminal set
        self.term_set = []

    def fit(self, features, labels):
        """Fit model to data"""
        # setup data

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

        # Randomize the order of the columns so there is no potential bias introduced by the initial order
        # of the columns, e.g., the most predictive features at the beginning or end.
        # data_columns = list(train_val_data.columns.values)
        # np.random.shuffle(data_columns)
        # train_val_data = train_val_data[data_columns]

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

        # initial model
        self._best_estimator = copy.deepcopy(self.ml.fit(x_t,y_t))
        self._best_score = self._best_estimator.score(x_v,y_v)
        print("initial estimator size:",self._best_estimator.coef_.shape)
        print("initial score:",self._best_score)
        # create terminal set
        for i in np.arange(x_t.shape[1]):
            # (.,.,.): node type, arity, feature column index or value
            self.term_set.append(('x',0,i)) # features
            # add ephemeral random constants if flag
            if self.erc:
                self.term_set.append(('k',0,np.random.rand())) # ephemeral random constants

        # Create initial population
        pop = self.init_pop()

        # Evaluate the entire population
        # X represents a matrix of the population outputs (number os samples x population size)
        # pop.X = np.asarray(list(map(lambda I: out(I,x_t,labels), pop.individuals)))
        pop.X = self.transform(x_t,pop.individuals,y_t)
        # calculate fitness of individuals
        # fitnesses = list(map(lambda I: fitness(I,y_t,self.machine_learner),pop.X))
        fitnesses = calc_fitness(pop,y_t,self._fit_choice)
        # print("fitnesses:",fitnesses)
        # Assign fitnesses to inidividuals in population
        for ind, fit in zip(pop.individuals, fitnesses):
            if isinstance(fit,(list,np.ndarray)): # calc_fitness returned raw fitness values
                fit[fit < 0] = 999999.666
                fit[np.isnan(fit)] = 999999.666
                fit[np.isinf(fit)] = 999999.666
                ind.fitness_vec = fit
                ind.fitness = np.mean(ind.fitness_vec)
            else:
                ind.fitness = np.nanmin([fit,99999.666])
        # for each generation g
        for g in np.arange(self.generations):
            # pdb.set_trace()
            # print("X shape:",pop.X.shape)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                # print("population:",stacks_2_eqns())
                try:
                    self.ml.fit(pop.X.transpose(),y_t)
                except:
                    pdb.set_trace()
            # keep best model
            try:
                tmp = self.ml.score((self.transform(x_v,pop.individuals)).transpose(),y_v)
            except Exception:
                tmp = 0

            if tmp > self._best_score:
                print("best internal validation score:",self._best_score)
                self._best_estimator = copy.deepcopy(self.ml)
                self._best_score = tmp
                self._best_inds = pop.individuals[:]
                # print("best individuals updated")
            offspring = []

            # clone individuals for offspring creation
            if self.sel == 'lasso':
                # for lasso, filter individuals with 0 coefficients
                offspring = copy.deepcopy(list(x for i,x in zip(ml.coef_, pop.individuals) if  i != 0))
            else:
                offspring = copy.deepcopy(pop.individuals)

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if np.random.rand() < self.crossover_rate:
                    cross(child1.stack, child2.stack, self.max_depth)
                else:
                    mutate(child1.stack,self.func_set,self.term_set)
                    mutate(child2.stack,self.func_set,self.term_set)
                child1.fitness = -1
                child2.fitness = -1

            # for mutant in offspring:
            #     if np.random.rand() < self.mutation_rate:
            #         mutate(mutant.stack,self.func_set,self.term_set)
            #         # print("pop being mutated:",list(map(lambda p: stack_2_eqn(p), offspring)))
            #         mutant.fitness = -1

            # Survival the next generation individuals
            if self.sel == 'tournament':
                offspring = tournament(pop.individuals + offspring, self.tourn_size, num_selections = len(pop.individuals))
            elif self.sel == 'lexicase':
                offspring = lexicase(pop.inviduals + offspring, len(pop.individuals), survival = True)
            elif self.sel == 'epsilon_lexicase':
                # print("pop going in to ep lexicase:",stacks_2_eqns(pop.individuals + offspring))
                offspring = epsilon_lexicase(pop.individuals + offspring, len(pop.individuals), survival = True)
                # print("pop coming out of ep lexicase:",stacks_2_eqns(offspring))


            # The population is entirely replaced by the offspring
            pop.individuals[:] = offspring
            pop.X = self.transform(x_t,pop.individuals)
            # print("pop.X.shape:",pop.X.shape)
            # fitnesses = list(map(lambda I: fitness(I,y_t,self._fit_choice),pop.X))
            fitnesses = calc_fitness(pop,y_t,self._fit_choice)
            # print("fitnesses:",fitnesses)
            # Assign fitnesses to inidividuals in population
            for ind, fit in zip(pop.individuals, fitnesses):
                if isinstance(fit,(list,np.ndarray)): # calc_fitness returned raw fitness values
                    fit[fit < 0] = 999999.666
                    fit[np.isnan(fit)] = 999999.666
                    fit[np.isinf(fit)] = 999999.666
                    ind.fitness_vec = fit
                    ind.fitness = np.mean(ind.fitness_vec)
                else:
                    # print("fit.shape:",fit.shape)
                    ind.fitness = np.nanmin([fit,99999.666])

        print("best score:",self._best_score)
        print("features:",stacks_2_eqns(self._best_inds))
        return self.score(features,labels)

    def transform(self,x,inds,labels = None):
        """return a transformation of x using population outputs"""
        return np.asarray(list(map(lambda I: out(I,x,labels), inds)),order='F')

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
        if self._best_inds is None:
            return self._best_estimator.predict(testing_features)
        else:
            X_transform = (np.asarray(list(map(lambda I: out(I,testing_features), self._best_inds))))
            return self._best_estimator.predict(X_transform.transpose())

    def fit_predict(self, features, labels):
        """Convenience function that fits a pipeline then predicts on the provided features

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
        return r2_score(testing_labels,yhat)

    def get_params(self, deep=None):
        """returns parameters of the current FEW instance

        This function is necessary for FEW to work as a drop-in estimator in,
        e.g., sklearn.cross_validation.cross_val_score

        Parameters
        ----------
        deep: unused
            Only implemented to maintain interface for sklearn

        Returns
        -------
        params : mapping of string to any
            Parameter names mapped to their values.

        """

        return self.params
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
            raise ValueError('A model has not been optimized. Please call fit() first.')

        # Have the exported code import all of the necessary modules and functions
        # write model form from coefficients and features
        model = ''
        sym_model = ''
        x = 0
        for c,i in zip(self._best_estimator.coef_,stacks_2_eqns(self._best_inds)):
           if c != 0:
               if model:
                   model+= '+'
                   sym_model += '+'
               model+= str(c) + '*' + str(i)
               sym_model += "k_" + str(x) + '*' + str(i)
               x += 1

        print_text = "exact_model: " + model
        print_text += "\nsymbolic_model: " + sym_model
        print_text += "\ncoefficients: " + str([c for c in self._best_estimator.coef_ if c != 0])
        print_text += "\nfeatures: " + str([s for s,c in zip(stacks_2_eqns(self._best_inds),self._best_estimator.coef_) if c!=0])

        with open(output_file_name, 'w') as output_file:
            output_file.write(print_text)

    def init_pop(self):
    	"""initializes population of features as GP stacks."""
    	pop = Pop(self.population_size,self._training_features.shape[0])
    	# make programs
    	if self.seed_with_ml:
            # initial population is the components of the default ml model
            if self.machine_learner == 'lasso':
                # add all model components with non-zero coefficients
                for i,(c,p) in enumerate(it.zip_longest([c for c in self.ml.coef_ if c !=0],pop.individuals,fillvalue=None)):
                    if c is not None and p is not None:
                        p.stack = [('x',0,i)]
                    elif p is not None:
                        # make program if pop is bigger than model componennts
                        make_program(p.stack,self.func_set,self.term_set,np.random.randint(self.min_depth,self.max_depth+1))
                        p.stack = list(reversed(p.stack))
            # print initial population
            print("seeded initial population:",stacks_2_eqns(pop.individuals))

    	else:
    		for I in pop.individuals:
    			depth = np.random.randint(self.min_depth,self.max_depth+1)
    			# print("hex(id(I)):",hex(id(I)))
    			# depth = 2;
    			# print("initial I.stack:",I.stack)
    			make_program(I.stack,self.func_set,self.term_set,depth)
    			# print(I.stack)
    			I.stack = list(reversed(I.stack))

    		# print(I.stack)

    	return pop

def positive_integer(value):
    """Ensures that the provided value is a positive integer; throws an exception otherwise

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
        raise argparse.ArgumentTypeError('Invalid int value: \'{}\''.format(value))
    if value < 0:
        raise argparse.ArgumentTypeError('Invalid positive int value: \'{}\''.format(value))
    return value

def float_range(value):
    """Ensures that the provided value is a float integer in the range (0., 1.); throws an exception otherwise

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
        raise argparse.ArgumentTypeError('Invalid float value: \'{}\''.format(value))
    if value < 0.0 or value > 1.0:
        raise argparse.ArgumentTypeError('Invalid float value: \'{}\''.format(value))
    return value

# main functions
def main():
    """Main function that is called when FEW is run on the command line"""
    parser = argparse.ArgumentParser(description='A feature engineering wrapper for '
                                                 'machine learning algorithms using genetic programming.',
                                     add_help=False)

    parser.add_argument('INPUT_FILE', type=str, help='Data file to run FEW on; ensure that the target/label column is labeled as "label".')

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    parser.add_argument('-is', action='store', dest='INPUT_SEPARATOR', default=None,
                        type=str, help='Character used to separate columns in the input file.')

    parser.add_argument('-o', action='store', dest='OUTPUT_FILE', default='',
                        type=str, help='File to export the final model.')

    parser.add_argument('-g', action='store', dest='GENERATIONS', default=100,
                        type=positive_integer, help='Number of generations to run FEW.')

    parser.add_argument('-p', action='store', dest='POPULATION_SIZE', default=100,
                        type=positive_integer, help='Number of individuals in the GP population.')

    parser.add_argument('-mr', action='store', dest='MUTATION_RATE', default=0.8,
                        type=float_range, help='GP mutation rate in the range [0.0, 1.0].')

    parser.add_argument('-xr', action='store', dest='CROSSOVER_RATE', default=0.2,
                        type=float_range, help='GP crossover rate in the range [0.0, 1.0].')

    parser.add_argument('-ml', action='store', dest='MACHINE_LEARNER', default='lasso', choices = ['lasso'],
                        type=str, help='ML algorithm to pair with features. Default: Lasso')

    parser.add_argument('-min_depth', action='store', dest='MIN_DEPTH', default=1,
                        type=positive_integer, help='Minimum length of GP programs.')

    parser.add_argument('-max_depth', action='store', dest='MAX_DEPTH', default=2,
                        type=positive_integer, help='Maximum number of nodes in GP programs.')

    parser.add_argument('-max_depth_init', action='store', dest='MAX_DEPTH_INIT', default=1,
                        type=positive_integer, help='Maximum number of nodes in initialized GP programs.')

    parser.add_argument('-op_weight', action='store', dest='OP_WEIGHT', default=1,
                        type=bool, help='Weight variables for inclusion in synthesized features based on ML scores. Default: off')

    parser.add_argument('-sel', action='store', dest='SEL', default='tournament', choices = ['tournament','lexicase','epsilon_lexicase'],
                        type=str, help='Selection method (Default: tournament)')

    parser.add_argument('-tourn_size', action='store', dest='TOURN_SIZE', default=2,
                        type=positive_integer, help='Tournament size for tournament selection (Default: 2)')

    parser.add_argument('-fit', action='store', dest='FIT_CHOICE', default='mse', choices = ['mse','mae','mdae','r2','vaf',
                        'mse_rel','mae_rel','mdae_re','r2_rel','vaf_rel'],
                        type=str, help='Fitness metric (Default: mse)')

    parser.add_argument('--seed_with_ml', action='store_true', dest='SEED_WITH_ML', default=False,
                    help='Flag to seed initial GP population with components of the ML model.')

    parser.add_argument('--erc', action='store_true', dest='ERC', default=False,
                    help='Flag to use ephemeral random constants in GP feature construction.')

    parser.add_argument('-s', action='store', dest='RANDOM_STATE', default=0,
                        type=int, help='Random number generator seed for reproducibility. Set this seed if you want your FEW run to be reproducible '
                                       'with the same seed and data set in the future.')

    parser.add_argument('-v', action='store', dest='VERBOSITY', default=1, choices=[0, 1, 2],
                        type=int, help='How much information FEW communicates while it is running: 0 = none, 1 = minimal, 2 = all.')

    parser.add_argument('--no-update-check', action='store_true', dest='DISABLE_UPDATE_CHECK', default=False,
                        help='Flag indicating whether the FEW version checker should be disabled.')

    parser.add_argument('--version', action='version', version='FEW {version}'.format(version=__version__),
                        help='Show FEW\'s version number and exit.')

    args = parser.parse_args()

    if args.VERBOSITY >= 2:
        print('\nFEW settings:')
        for arg in sorted(args.__dict__):
            if arg == 'DISABLE_UPDATE_CHECK':
                continue
            print('{}\t=\t{}'.format(arg, args.__dict__[arg]))
        print('')

    # load data from csv file
    if args.INPUT_SEPARATOR is None:
        input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR,engine='python')
    else: # use c engine for read_csv is separator is specified
        input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR)

    if 'Label' in input_data.columns.values:
        input_data.rename(columns={'Label': 'label'}, inplace=True)

    RANDOM_STATE = args.RANDOM_STATE if args.RANDOM_STATE > 0 else None

    train_i, test_i = train_test_split(input_data.index,
                                                        stratify = None,#  stratify=input_data['label'].values,
                                                         train_size=0.75,
                                                         test_size=0.25,
                                                         random_state=RANDOM_STATE)

    training_features = input_data.loc[train_i].drop('label', axis=1).values
    training_labels = input_data.loc[train_i, 'label'].values

    testing_features = input_data.loc[test_i].drop('label', axis=1).values
    testing_labels = input_data.loc[test_i, 'label'].values

    learner = FEW(generations=args.GENERATIONS, population_size=args.POPULATION_SIZE,
                mutation_rate=args.MUTATION_RATE, crossover_rate=args.CROSSOVER_RATE,
                machine_learner = args.MACHINE_LEARNER, min_depth = args.MIN_DEPTH,
                max_depth = args.MAX_DEPTH, sel = args.SEL, tourn_size = args.TOURN_SIZE,
                seed_with_ml = args.SEED_WITH_ML, op_weight = args.OP_WEIGHT,
                erc = args.ERC, random_state=args.RANDOM_STATE, verbosity=args.VERBOSITY,
                disable_update_check=args.DISABLE_UPDATE_CHECK)

    learner.fit(training_features, training_labels)

    if args.VERBOSITY >= 1:
        print('\nTraining accuracy: {}'.format(learner.score(training_features, training_labels)))
        print('Holdout accuracy: {}'.format(learner.score(testing_features, testing_labels)))

    if args.OUTPUT_FILE != '':
        learner.export(args.OUTPUT_FILE)


if __name__ == '__main__':
    main()
