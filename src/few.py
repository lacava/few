# William La Cava
# 2016 - 07 - 29
# few: a feature engineering wrapper for sci-kitlearn
#
# imports
#import scikit-learn
import argparse
from _version import __version__

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

    parser.add_argument('INPUT_FILE', type=str, help='Data file to optimize the pipeline on; ensure that the class label column is labeled as "class".')

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

    parser.add_argument('-ml', action='store', dest='MACHINE_LEARNER', default='lr',
                        type=str, help='ML algorithm to pair with features.')

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
    training_classes = input_data.loc[training_indices, 'class'].values

    testing_features = input_data.loc[testing_indices].drop('class', axis=1).values
    testing_classes = input_data.loc[testing_indices, 'class'].values

    FEW = FEW(generations=args.GENERATIONS, population_size=args.POPULATION_SIZE,
                mutation_rate=args.MUTATION_RATE, crossover_rate=args.CROSSOVER_RATE,
                random_state=args.RANDOM_STATE, verbosity=args.VERBOSITY,
                disable_update_check=args.DISABLE_UPDATE_CHECK)

    FEW.fit(training_features, training_classes)

    if args.VERBOSITY >= 1:
        print('\nTraining accuracy: {}'.format(FEW.score(training_features, training_classes)))
        print('Holdout accuracy: {}'.format(FEW.score(testing_features, testing_classes)))

    if args.OUTPUT_FILE != '':
        FEW.export(args.OUTPUT_FILE)


if __name__ == '__main__':
    main()
