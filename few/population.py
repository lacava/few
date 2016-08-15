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
		self.stack = stack[:]

class Pop(object):
	""" class representing population """
	def __init__(self,pop_size=100,n_samples=1):
		""" initializes population of inds of size pop_size """
		print("pop_size:",pop_size)
		print("n_samples:",n_samples)
		self.individuals = []
		# initialize empty output matrix
		self.X = np.empty([n_samples,pop_size],dtype=float)
		# initialize empty error matrix
		self.E = np.empty([n_samples,pop_size],dtype=float)
		# initialize population programs
		for i in np.arange(pop_size):
			self.individuals.append(ind())

def init(population_size,n_samples,func_set,term_set,min_depth,max_depth):
	""" initializes population of features as GP stacks. """
	pop = Pop(population_size,n_samples)
	# build programs
	typ = 'f'
	# make programs
	for I in pop.individuals:
		depth = np.random.randint(min_depth,max_depth)
		print("hex(id(I)):",hex(id(I)))
		# depth = 2;
		print("initial I.stack:",I.stack)
		make_program(I.stack,func_set,term_set,depth)
		# print(I.stack)
		I.stack = list(reversed(I.stack))

		print(I.stack)

	return pop

def make_program(stack,func_set,term_set,max_d):
	""" makes a program stack. """
	print("stack:",stack,"max d:",max_d)
	if max_d == 0 or np.random.rand() < float(len(term_set))/(len(term_set)+len(func_set)):
		stack.append(term_set[np.random.choice(len(term_set))])
	else:
		stack.append(func_set[np.random.choice(len(func_set))])
		for i in np.arange(stack[-1][1]):
			make_program(stack,func_set,term_set,max_d-1)
	# return stack
	# print("current stack:",stack)
