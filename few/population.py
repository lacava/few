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

class ind(object):
	""" class for features, represented as GP stacks."""
	def __init__(self,fitness = -1,stack = []):
		""" initializes empty individual with invalid fitness. """
		self.fitness = fitness
		self.stack = stack

class pop(object):
	""" class representing population """
	def __init__(self,pop_size=100,n_samples=1):
		""" initializes population of inds of size pop_size """
		self.programs = []
		# initialize empty output matrix
		self.X = np.array(n_samples,pop_size)
		# initialize empty error matrix
		self.E = np.array(n_samples,pop_size)
		# initialize population programs
		for i in pop_size:
			self.programs.append(ind())

def init(population_size,n_samples,n_features,min_len,max_len,p):
	""" initializes population of features as GP stacks. """
	pop = pop(population_size,n_samples)
	# build programs
	typ = 'f'
	# make programs
	for I in pop.programs:
		depth = np.random.randint(min_depth,max_depth)
		make_program(I,depth,1,typ)

	return pop

def make_program(I,func_set,term_set,max_d):
	""" makes a program stack. """
	if max_d == 0 or np.random.rand() < len(term_set)/(len(term_set)+len(func_set)):
		I.append(np.random.choice(term_set))
	else:
		I.append(np.random.choice(func_set))
		for i in np.arange(I[-1][1]):
			make_program(I,func_set,term_set,max_d-1)
	# return in post-fix notation
	return list(reversed(I))
