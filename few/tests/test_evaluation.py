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
# unit tests for evaluation methods.
from few import FEW
from few.evaluation import divs
from few.population import *
from sklearn.datasets import load_boston
import numpy as np

def test_out_shapes():
    """test_evaluation.py: program output is correct size """
    # load test data set
    boston = load_boston()
    # boston.data = boston.data[::10]
    # boston.target = boston.target[::10]
    n_features = boston.data.shape[1]
    # function set

    # terminal set
    term_set = []
    # numbers represent column indices of features
    for i in np.arange(n_features):
        term_set.append(node('x',loc=i)) # features
        # term_set.append(('k',0,np.random.rand())) # ephemeral random constants

    # initialize population
    pop_size = 5;
    few = FEW(population_size=pop_size,seed_with_ml=False)
    few.term_set = term_set
    pop = few.init_pop(n_features)

    pop.X = np.asarray(list(map(lambda I: few.out(I,boston.data), pop.individuals)))

    #pop.X = out(pop.individuals[0],boston.data,boston.target)
    print("pop.X.shape:",pop.X.shape)
    print("boston.target.shape",boston.target.shape)
    assert pop.X.shape == (pop_size, boston.target.shape[0])

def test_out_is_correct():
    """test_evaluation.py: output matches known function outputs """

    boston = load_boston()
    n_features = boston.data.shape[1]
    X = boston.data
    Y = boston.target
    p1 = Ind()
    p2 = Ind()
    p3 = Ind()
    p4 = Ind()
    p5 = Ind()

    p1.stack = [node('x',loc=4), node('x',loc=5), node('-'),
                node('k',value=0.175), node('log'), node('-')]

    p2.stack = [node('x',loc=7),node('x',loc=8),node('*')]

    p3.stack =  [node('x',loc=0),node('exp'), node('x',loc=5),node('x',loc=7),
                node('*'),node('/')]

    p4.stack =  [node('x',loc=12),node('sin')]

    p5.stack = [node('k',value=178.3),node('x',loc=8),node('*'),node('x',loc=7),node('cos'),node('+')]
    few = FEW()
    y1 = few.safe(np.log(0.175) - (X[:,5] - X[:,4]))
    y2 = few.safe(X[:,7]*X[:,8])
    y3 = few.safe(divs(X[:,5]*X[:,7],np.exp(X[:,0])))
    y4 = few.safe(np.sin(X[:,12]))
    y5 = few.safe(178.3*X[:,8]+np.cos(X[:,7]))

    # y1,y2,y3,y4,y5 = safe(y1),safe(y2),safe(y3),safe(y4),safe(y5)
    few = FEW()
    assert np.array_equal(y1,few.out(p1,X))
    print("y1 passed")
    assert np.array_equal(y2,few.out(p2,X))
    print("y2 passed")
    assert np.array_equal(y3, few.out(p3,X))
    print("y3 passed")
    # print("y4:",y4,"y4hat:",few.out(p4,X,Y))
    assert np.array_equal(y4, few.out(p4,X))
    print("y4 passed")
    assert np.array_equal(y5, few.out(p5,X))

def test_calc_fitness_shape():
    """test_evaluation.py: calc_fitness correct shapes """
    # load test data set
    boston = load_boston()
    # boston.data = boston.data[::10]
    # boston.target = boston.target[::10]
    n_features = boston.data.shape[1]
    # terminal set
    term_set = []
    # numbers represent column indices of features
    for i in np.arange(n_features):
        term_set.append(node('x',loc=i)) # features
        # term_set.append(('k',0,np.random.rand())) # ephemeral random constants

    # initialize population
    pop_size = 5;
    few = FEW(population_size=pop_size,seed_with_ml=False)
    few.term_set = term_set
    pop = few.init_pop(n_features)

    pop.X = np.asarray(list(map(lambda I: few.out(I,boston.data), pop.individuals)))

    fitnesses = few.calc_fitness(pop.X,boston.target,'mse','tournament')
    assert len(fitnesses) == len(pop.individuals)

    # test vectorized fitnesses
    vec_fitnesses = few.calc_fitness(pop.X,boston.target,'mse','lexicase')
    fitmat = np.asarray(vec_fitnesses)
    print("fitmat.shape:",fitmat.shape)
    assert fitmat.shape == (len(pop.individuals),boston.target.shape[0])

def test_inertia():
    """test_evaluation.py: inertia works"""
    import pdb
    # perfect inertia
    x = np.hstack((np.zeros(50),np.ones(50)))
    y = np.hstack((np.zeros(50),np.ones(50)))
    few = FEW()
    mean_inertia = few.inertia(x,y)
    sample_inertia = few.inertia(x,y,samples=True)
    assert(mean_inertia==0)
    assert(np.mean(sample_inertia)==mean_inertia)

    # half inertia
    x = np.hstack((np.ones(25),np.zeros(25),np.ones(25),np.zeros(25)))
    y = np.hstack((np.zeros(50),np.ones(50)))

    mean_inertia = few.inertia(x,y)
    sample_inertia = few.inertia(x,y,samples=True)
    print('mean_inertia:',mean_inertia)
    print('sample_inertia',sample_inertia)
    assert(mean_inertia==0.25)
    assert(np.mean(sample_inertia)==mean_inertia)

def test_separation():
    """test_evaluation: separation"""
    # perfect separation
    x = np.hstack((np.zeros(50),np.ones(50)))
    y = np.hstack((np.zeros(50),np.ones(50)))
    few = FEW()
    mean_separation = few.separation(x,y)
    sample_separation = few.separation(x,y,samples=True)
    print('mean_separation:',mean_separation)
    print('sample_separation',sample_separation)
    assert(mean_separation==1)
    assert(np.mean(sample_separation)==mean_separation)

    # perfect separation
    x = np.hstack((np.ones(50),np.zeros(50)))
    y = np.hstack((np.zeros(50),np.ones(50)))

    mean_separation = few.separation(x,y)
    sample_separation = few.separation(x,y,samples=True)
    print('mean_separation:',mean_separation)
    print('sample_separation',sample_separation)
    assert(mean_separation==1)
    assert(np.mean(sample_separation)==mean_separation)

    # half separation
    x = np.hstack((np.ones(25),np.zeros(25),np.ones(25),np.zeros(25)))
    y = np.hstack((np.zeros(50),np.ones(50)))

    mean_separation = few.separation(x,y)
    sample_separation = few.separation(x,y,samples=True)
    print('mean_separation:',mean_separation)
    print('sample_separation',sample_separation)
    assert(mean_separation==0.25)
    assert(np.mean(sample_separation)==mean_separation)
