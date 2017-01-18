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
# unit tests for variation methods.
import numpy as np

from few.tests.test_population import is_valid_program,node
from few.variation import cross, mutate

def test_cross_makes_valid_program():
    """test_variation.py: crossover makes valid programs """
    # np.random.seed(65)
    # I = (a+b)*x
    p1 = [node('x',loc=1),node('x',loc=2),node('+'),node('x',loc=3),node('*')]
    # p1 = [{'name':'x','arity':0,'loc':1,'in_type':None,'out_type':'f'},
    #       {'name':'x','arity':0,'loc':2,'in_type':None,'out_type':'f'},
    #       {'name':'+','arity':2,'in_type':'f','out_type':'f'},
    #       {'name':'x','arity':0,'loc':3,'in_type':None,'out_type':'f'},
    #       {'name':'*','arity':2,'in_type':'f','out_type':'f'}]
    # J = (x/z)-(n*b)
    p2 = [node('x',loc=1), node('x',loc=2), node('/'), node('k',value=3.7), node('x',loc=4), node('*'), node('-')]
    # p2 = [{'name':'x','arity':0,'loc':1,'in_type':None,'out_type':'f'},
    #       {'name':'x','arity':0,'loc':2,'in_type':None,'out_type':'f'},
    #       {'name':'/','arity':2,'in_type':'f','out_type':'f'},
    #       {'name':'k','arity':0,'value':3.7,'in_type':None,'out_type':'f'},
    #       {'name':'x','arity':0,'loc':4,'in_type':None,'out_type':'f'},
    #       {'name':'*','arity':2,'in_type':'f','out_type':'f'},
    #       {'name':'-','arity':2,'in_type':'f','out_type':'f'}]

    for i in np.arange(1000):
        cross(p1,p2)
        assert is_valid_program(p1) and is_valid_program(p2)

def test_mutate_makes_valid_program():
    """test_variation.py: mutation makes valid programs """
    func_set = [node('+'), node('-'), node('*'), node('/'), node('sin'),
                 node('cos'), node('exp'),node('log'), node('^2'),
                 node('^3'), node('sqrt')]
    # terminal set
    term_set = []
    # numbers represent column indices of features
    term_set = [node('x',loc=i) for i in np.arange(10)]
    term_set += [node('k',value=np.random.rand()) for i in np.arange(10)]
    # for i in np.arange(10):
    #     term_set.append({'name':'x','arity':0,'loc':i,'in_type':None,'out_type':'f'}) # features
    #     term_set.append({'name':'k','arity':0,'value':np.random.rand(),'in_type':None,'out_type':'f'}) # ephemeral random constants
    p = [node('k',value=5),node('x',loc=6),node('/'),node('k',value=7),node('x',loc=8),node('*'),node('-')]
    # p = [{'name':'k','arity':0,'value':5,'in_type':None,'out_type':'f'},{'name':'x','arity':0,'loc':6,'in_type':None,'out_type':'f'},{'name':'/','arity':2,'in_type':'f','out_type':'f'},
    #      {'name':'k','arity':0,'value':7,'in_type':None,'out_type':'f'},{'name':'x','arity':0,'loc':8,'in_type':None,'out_type':'f'},
    #      {'name':'*','arity':2,'in_type':'f','out_type':'f'},{'name':'-','arity':2,'in_type':'f','out_type':'f'}]
    for i in np.arange(1000):
        mutate(p,func_set,term_set)
        assert is_valid_program(p)
