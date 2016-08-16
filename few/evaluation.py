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
import numpy as np
# evaluation functions. these can be sped up using a GPU!
eval_dict = {
    '+': lambda n,features,stack_float: stack_float.pop() + stack_float.pop(),
    '-': lambda n,features,stack_float: stack_float.pop() - stack_float.pop(),
    '*': lambda n,features,stack_float: stack_float.pop() * stack_float.pop(),
    '/': lambda n,features,stack_float: stack_float.pop() / stack_float.pop(),
    'sin': lambda n,features,stack_float: np.sin(stack_float.pop()),
    'cos': lambda n,features,stack_float: np.cos(stack_float.pop()),
    'exp': lambda n,features,stack_float: np.exp(stack_float.pop()),
    'log': lambda n,features,stack_float: np.log(stack_float.pop()),
    'x':  lambda n,features,stack_float: features[:,n[2]],
    'k': lambda n,features,stack_float: np.ones(features.shape[0])*n[2]
}

def eval(n, features, stack_float):

    if len(stack_float) >= n[1]:
        stack_float.append(eval_dict[n[0]](n,features,stack_float))

def out(I,features,labels=None):
    """computes the output for individual I """
    stack_float = []
    # print("stack:",I.stack)
    # evaulate stack over rows of features,labels
    for n in I.stack:
        eval(n,features,stack_float)
        # print("stack_float:",stack_float)

    return stack_float[-1]

def fitness(yhat,labels,machine_learner):
    """computes fitness of individual output yhat.
    yhat: output of a program.
    labels: correct outputs
    machine_learner: machine learner from sklearn. """
    return np.sum((yhat-labels)**2)
