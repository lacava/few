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

def out(I,features,labels):
    """computes the output for individual I """
    stack_float = []
    # evaulate stack over rows of features,labels
    for e in I.stack:
        eval(e,features,stack_float)

    return stack_float[-1]

def fit():
    """computes fitness of individual I """
def eval(node, features, stack_float):
    if len(stack_float) >= n[1]:
        stack_float.append(eval_dict(n,features,stack_float))

def eval_dict(n,features,stack_float):

    return {
        '+': stack_float.pop() + stack_float.pop(),
        '-': stack_float.pop() - stack_float.pop(),
        '*': stack_float.pop() * stack_float.pop(),
        '/': stack_float.pop() / stack_float.pop(),
        'sin': sin(stack_float.pop()),
        'cos': cos(stack_float.pop()),
        'exp': exp(stack_float.pop()),
        'log': log(stack_float.pop()),
        'n':  features[:,n[2]],
        'erc': np.ones(features.shape[0])*n[2]
    }[n[0]]
