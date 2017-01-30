"""
Copyright 2016 William La Cava

This file is part of the FEW library.

License: GPL 3
"""
from .test_variation import test_cross_makes_valid_program, test_mutate_makes_valid_program
from .test_population import test_pop_shape, test_pop_init
from .test_selection import *
from .test_evaluation import test_out_shapes, test_out_is_correct, test_calc_fitness_shape, test_inertia, test_separation
from .test_few import test_few_fit_shapes, test_few_at_least_as_good_as_default, test_few_classification
