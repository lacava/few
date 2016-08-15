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
from few.evaluation import out
from few.population import init,ind
from sklearn.datasets import load_boston
import numpy as np
def test_out_shapes():
    """program output is correct size """
    # load test data set
    boston = load_boston()
    # boston.data = boston.data[::10]
    # boston.target = boston.target[::10]
    n_features = boston.data.shape[1]
    # function set
    func_set = [('+',2),('-',2),('*',2),('/',2),('sin',1),('cos',1),('exp',1),('log',1)]
    # terminal set
    term_set = []
    # numbers represent column indices of features
    for i in np.arange(n_features):
        term_set.append(('x',0,i)) # features
        # term_set.append(('k',0,np.random.rand())) # ephemeral random constants

    # initialize population
    pop_size = 5;
    pop = init(5,boston.data.shape[0],func_set,term_set,1,2)

    pop.X = np.asarray(list(map(lambda I: out(I,boston.data,boston.target), pop.individuals))).transpose()

    #pop.X = out(pop.individuals[0],boston.data,boston.target)
    print("pop.X.shape:",pop.X.shape)
    print("boston.target.shape",boston.target.shape)
    assert pop.X.shape == (boston.target.shape[0],pop_size)

def test_out_is_correct():
    """ output matches known function outputs """

    boston = load_boston()
    n_features = boston.data.shape[1]
    X = boston.data
    Y = boston.target
    p1 = ind()
    p2 = ind()
    p3 = ind()
    p4 = ind()
    p5 = ind()
    p1.stack = [('x', 0, 4), ('x', 0, 5), ('-', 2), ('k', 0, .175), ('log', 1), ('-', 2)]
    p2.stack = [('x', 0, 7), ('x', 0, 8), ('*', 2)]
    p3.stack =  [('x', 0, 0), ('exp', 1), ('x', 0, 5), ('x', 0, 7), ('*', 2), ('/', 2)]
    p4.stack =  [('x', 0, 12), ('sin', 1)]
    p5.stack = [('k', 0, 178.3), ('x', 0, 8), ('*', 2), ('x', 0, 7), ('cos', 1), ('+', 2)]

    y1 = np.log(0.175) - (X[:,5] - X[:,4])
    y2 = X[:,7]*X[:,8]
    y3 = X[:,5]*X[:,7] / np.exp(X[:,0])
    y4 = np.sin(X[:,12])
    y5 = 178.3*X[:,8]+np.cos(X[:,7])


    assert np.array_equal(y1,out(p1,X,Y))
    print("y1 passed")
    assert np.array_equal(y2,out(p2,X,Y))
    print("y2 passed")
    assert np.array_equal(y3, out(p3,X,Y))
    print("y3 passed")
    print("y4:",y4,"y4hat:",out(p4,X,Y))
    assert np.array_equal(y4, out(p4,X,Y))
    print("y4 passed")
    assert np.array_equal(y5, out(p5,X,Y))
