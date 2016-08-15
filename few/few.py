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
from _version import __version__
from .population import ind, Pop, init, make_program
from .variation import cross, mutate
from .selection import tournament


from sklearn import linear_model
import numpy as np
import pandas as pd

class FEW(object):
    """ FEW uses GP to find a set of transformations from the original feature space
    that produces the best performance for a given machine learner. """
    def __init__(self, population_size=100, generations=100,
                 mutation_rate=0.2, crossover_rate=0.8,
                 machine_learner = 'lasso', min_depth = 1, max_depth = 5, max_depth_init = 5,
                 sel = 'tournament', random_state=0, verbosity=0, scoring_function=None,
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

        self._optimized_estimator = None
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
        # self.op_weight = op_weight
        self.sel = sel
        # instantiate sklearn estimator according to specified machine learner
        if (self.machine_learner.lower() == "lasso"):
            ml = LassoLarsCV()
        elif (self.machine_learner.lower() == "distance"):
            ml = DistanceClassifier()
        else:
            ml = LassoLarsCV()
        # Columns to always ignore when in an operator
        self.non_feature_columns = ['class', 'group', 'guess']

        # function set
        self.func_set = [('+',2),('-',2),('*',2),('/',2),('sin',1),('cos',1),('exp',1),('log',1)]
        # terminal set
        self.term_set = []
        # numbers represent column indices of features
        for i in np.arange(n_features):
            term_set.append(('x'+str(i),0,i)) # features
            term_set.append(('k'+str(i),0,np.random.rand())) # ephemeral random constants

    def fit(self, features, labels):
        """ Fit model to data """
        # Create initial population
        pop = init(population_size,features.shape[0],features.shape[1],
        min_depth, max_depth, func_set, term_set)

        # Evaluate the entire population
        # X represents a matrix of the population outputs (number os samples x population size)
        pop.X = np.array(map(lambda I: ev.out(I,features,labels), pop.programs))

        # calculate fitness of individuals
        fitnesses = map(lambda I: ev.fitness(I,labels,machine_learner),pop.X)
        # Assign fitnesses to inidividuals in population
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit

        # for each generation g
        for g in generations:
            # use population output matrix as features for ML method
            ml.fit(X,labels)
            # Select the next generation individuals
            offspring = selection.tournament(pop, len(pop))
            # Clone the selected individuals
            offspring = list(map(clone, offspring))

            # Apply crossover and mutation on the offspring
            for child1, child2 in zip(offspring[::2], offspring[1::2]):
                if random.random() < crossover_rate:
                    variation.cross(child1, child2)
                    del child1.fitness.values
                    del child2.fitness.values

            for mutant in offspring:
                if random.random() < mutation_rate:
                    variation.mutate(mutant,func_set,term_set)
                    del mutant.fitness.values

            # Evaluate the individuals with an invalid fitness
            invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
            fitnesses = list(map(eval.fit, invalid_ind))
            for ind, fit in zip(invalid_ind, fitnesses):
                ind.fitness.values = fit

            # The population is entirely replaced by the offspring
            pop[:] = offspring


    def predict(self, testing_features):
        """ predict on a holdout data set. """
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
        """ estimates accuracy on testing set """
    def get_params(self, deep=None):
        """ returns parameters of the current FEW instance """
    def export(self, output_file_name):
        """ exports engineered features """

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

    parser.add_argument('INPUT_FILE', type=str, help='Data file to run FEW on; ensure that the class label column is labeled as "class".')

    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')

    parser.add_argument('-is', action='store', dest='INPUT_SEPARATOR', default='\t',
                        type=str, help='Character used to separate columns in the input file.')

    parser.add_argument('-o', action='store', dest='OUTPUT_FILE', default='',
                        type=str, help='File to export the final model.')

    parser.add_argument('-g', action='store', dest='GENERATIONS', default=100,
                        type=positive_integer, help='Number of generations to run FEW.')

    parser.add_argument('-p', action='store', dest='POPULATION_SIZE', default=100,
                        type=positive_integer, help='Number of individuals in the GP population.')

    parser.add_argument('-mr', action='store', dest='MUTATION_RATE', default=0.2,
                        type=float_range, help='GP mutation rate in the range [0.0, 1.0].')

    parser.add_argument('-xr', action='store', dest='CROSSOVER_RATE', default=0.8,
                        type=float_range, help='GP crossover rate in the range [0.0, 1.0].')

    parser.add_argument('-ml', action='store', dest='MACHINE_LEARNER', default='lasso',
                        type=str, help='ML algorithm to pair with features. Default: Lasso')

    parser.add_argument('-min_depth', action='store', dest='MIN_DEPTH', default=1,
                        type=int, help='Minimum length of GP programs.')

    parser.add_argument('-max_depth', action='store', dest='MAX_DEPTH', default=1,
                        type=int, help='Maximum number of nodes in GP programs.')

    parser.add_argument('-max_depth_init', action='store', dest='MAX_DEPTH', default=1,
                        type=int, help='Maximum number of nodes in initialized GP programs.')

    # parser.add_argument('-op_weight', action='store', dest='MAX_DEPTH', default=1,
    #                     type=int, help='Maximum number of nodes in initialized GP programs.')

    parser.add_argument('-sel', action='store', dest='SEL', default='tournament',
                        type=str, help='Selection method (tournament)')

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

    input_data = pd.read_csv(args.INPUT_FILE, sep=args.INPUT_SEPARATOR)

    if 'Class' in input_data.columns.values:
        input_data.rename(columns={'Class': 'class'}, inplace=True)

    RANDOM_STATE = args.RANDOM_STATE if args.RANDOM_STATE > 0 else None

    training_indices, testing_indices = train_test_split(input_data.index,
                                                         stratify=input_data['class'].values,
                                                         train_size=0.75,
                                                         test_size=0.25,
                                                         random_state=RANDOM_STATE)

    training_features = input_data.loc[training_indices].drop('class', axis=1).values
    training_labels = input_data.loc[training_indices, 'class'].values

    testing_features = input_data.loc[testing_indices].drop('class', axis=1).values
    testing_labels = input_data.loc[testing_indices, 'class'].values

    FEW = FEW(generations=args.GENERATIONS, population_size=args.POPULATION_SIZE,
                mutation_rate=args.MUTATION_RATE, crossover_rate=args.CROSSOVER_RATE,
                machine_learner = args.MACHINE_LEARNER, min_depth = args.MIN_DEPTH, max_depth = args.MAX_DEPTH,
                sel = args.SEL, random_state=args.RANDOM_STATE, verbosity=args.VERBOSITY,
                disable_update_check=args.DISABLE_UPDATE_CHECK)

    FEW.fit(training_features, training_labels)

    if args.VERBOSITY >= 1:
        print('\nTraining accuracy: {}'.format(FEW.score(training_features, training_labels)))
        print('Holdout accuracy: {}'.format(FEW.score(testing_features, testing_labels)))

    if args.OUTPUT_FILE != '':
        FEW.export(args.OUTPUT_FILE)


if __name__ == '__main__':
    main()
