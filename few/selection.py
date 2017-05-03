# -*- coding: utf-8 -*-
"""
Copyright 2016 William La Cava

license: GNU/GPLv3

"""
import numpy as np
import copy
import pdb
from sklearn.metrics import r2_score
from .population import stacks_2_eqns
from few_lib import ep_lex
# from profilehooks import profile

class SurvivalMixin(object):
    """
    class for implementing survival methods.
    """
    def survival(self,parents,offspring,elite=None,elite_index=None,X=None,F=None,F_offspring=None):
        """routes to the survival method, returns survivors"""
        if self.sel == 'tournament':
            survivors, survivor_index = self.tournament(parents + offspring, self.tourn_size, num_selections = len(parents))
        elif self.sel == 'lexicase':
            survivors, survivor_index = self.lexicase(parents + offspring, num_selections = len(parents), survival = True)
        elif self.sel == 'epsilon_lexicase':
            # survivors, survivor_index = self.epsilon_lexicase(parents + offspring, num_selections = len(parents), survival = True)
            survivor_index = self.epsilon_lexicase(np.vstack((F,F_offspring)), num_selections = F.shape[0], survival = True)
            survivors = [(parents+ offspring)[s] for s in survivor_index]
        elif self.sel == 'deterministic_crowding':
            survivors, survivor_index = self.deterministic_crowding(parents,offspring,X,X_offspring)
        elif self.sel == 'random':
            # pdb.set_trace()
            survivor_index = np.random.permutation(np.arange(2*len(parents)))[:len(parents)]
            survivors = [(parents + offspring)[s] for s in survivor_index]
        # elitism
        if self.elitism:
            if min([x.fitness for x in survivors]) > elite.fitness:
                # if the elite individual did not survive and elitism is on, replace worst individual with elite
                rep_index = np.argmax([x.fitness for x in survivors])
                survivors[rep_index] = elite
                survivor_index[rep_index] = elite_index
        # return survivors

        return survivors,survivor_index

    def tournament(self,individuals,tourn_size, num_selections=None):
        """conducts tournament selection of size tourn_size"""
        winners = []
        locs = []
        if num_selections is None:
            num_selections = len(individuals)

        for i in np.arange(num_selections):
            # sample pool with replacement
            pool_i = np.random.choice(len(individuals),size=tourn_size)
            pool = []
            for i in pool_i:
                pool.append(np.mean(individuals[i].fitness))
            # winner
            locs.append(pool_i[np.argmin(pool)])
            winners.append(copy.deepcopy(individuals[locs[-1]]))

        return winners,locs

    def lexicase(self,individuals, num_selections=None, epsilon = False, survival = False):
        """conducts lexicase selection for de-aggregated fitness vectors"""
        if num_selections is None:
            num_selections = len(individuals)
        winners = []
        locs = []

        if epsilon: # use epsilon lexicase selection
            # calculate epsilon thresholds based on median absolute deviation (MAD)
            mad_for_case = np.empty([len(individuals[0].fitness_vec),1])
            global_best_val_for_case = np.empty([len(individuals[0].fitness_vec),1])
            for i in np.arange(len(individuals[0].fitness_vec)):
                mad_for_case[i] = self.mad(np.asarray(list(map(lambda x: x.fitness_vec[i], individuals))))
                global_best_val_for_case[i] = min(map(lambda x: x.fitness_vec[i], individuals))
            # convert fitness values to pass/fail based on epsilon distance
            for I in individuals:
                fail_condition = np.array(I.fitness_vec > global_best_val_for_case[:,0] + mad_for_case[:,0]) #[f > global_best_val_for_case+mad_for_case for f in I.fitness_vec]
                I.fitness_vec = fail_condition.astype(int)

        for i in np.arange(num_selections):

            candidates = individuals
            can_locs = range(len(individuals))
            cases = list(np.arange(len(individuals[0].fitness_vec)))
            np.random.shuffle(cases)
            # pdb.set_trace()
            while len(cases) > 0 and len(candidates) > 1:
                # get best fitness for case among candidates
                # print("candidates:",stacks_2_eqns(candidates),"locations:",can_locs)
                # print("fitnesses for case "+str(cases[0])+":",[x.fitness_vec[cases[0]] for x in candidates])
                best_val_for_case = min([x.fitness_vec[cases[0]] for x in candidates])
                # print("best_val_for_case:",best_val_for_case)
                # filter individuals without an elite fitness on this case
                # tmp_c,tmp_l = zip(*((x,l) for x,l in zip(candidates,can_locs) if x.fitness_vec[cases[0]] == best_val_for_case))
                candidates,can_locs = zip(*((x,l) for x,l in zip(candidates,can_locs) if x.fitness_vec[cases[0]] == best_val_for_case))
                cases.pop(0)

            choice = np.random.randint(len(candidates))
            winners.append(copy.deepcopy(candidates[choice]))
            locs.append(can_locs[choice])
            if survival: # filter out winners from remaining selection pool
                individuals = list(filter(lambda x: x.stack != candidates[choice].stack, individuals))

        return winners, locs

    def epsilon_lexicase(self, F, num_selections=None, survival = False):
        """conducts epsilon lexicase selection for de-aggregated fitness vectors"""
        # pdb.set_trace()
        if self.c: # use c library
            # define c types
            locs = np.empty(num_selections,dtype='int32',order='F')
            # self.lib.epsilon_lexicase(F,F.shape[0],F.shape[1],num_selections,locs)
            ep_lex(F,F.shape[0],F.shape[1],num_selections,locs)
            return locs
        else: # use python version
            if num_selections is None:
                num_selections = F.shape[0]

            locs = []
            individual_locs = np.arange(F.shape[0])
            # calculate epsilon thresholds based on median absolute deviation (MAD)
            mad_for_case = np.array([self.mad(f) for f in F.transpose()])
            for i in np.arange(num_selections):

                can_locs = individual_locs
                cases = list(np.arange(F.shape[1]))
                np.random.shuffle(cases)
                # pdb.set_trace()
                while len(cases) > 0 and len(can_locs) > 1:
                    # get best fitness for case among candidates
                    best_val_for_case = np.min(F[can_locs,cases[0]])
                    # filter individuals without an elite fitness on this case
                    can_locs = [l for l in can_locs if F[l,cases[0]] <= best_val_for_case + mad_for_case[cases[0]]]
                    cases.pop(0)

                choice = np.random.randint(len(can_locs))
                locs.append(can_locs[choice])
                if survival: # filter out winners from remaining selection pool
                    individual_locs = [i for i in individual_locs if i != can_locs[choice]]

            while len(locs) < num_selections:
                locs.append(individual_locs[0])

            return locs


    def mad(self,x, axis=None):
        """median absolute deviation statistic"""
        return np.median(np.abs(x - np.median(x, axis)), axis)

    def deterministic_crowding(self,parents,offspring,X_parents,X_offspring):
        """deterministic crowding implementation (for non-steady state).
        offspring compete against the parent they are most similar to, here defined as
        the parent they are most correlated with.
        the offspring only replace their parent if they are more fit.
        """
        # get children locations produced from crossover
        cross_children = [i for i,o in enumerate(offspring) if len(o.parentid) > 1]
        # order offspring so that they are lined up with their most similar parent
        for c1,c2 in zip(cross_children[::2], cross_children[1::2]):
            # get parent locations
            p_loc = [j for j,p in enumerate(parents) if p.id in offspring[c1].parentid]
            if len(p_loc) != 2:
                pdb.set_trace()
            # if child is more correlated with its non-root parent
            if r2_score(X_parents[p_loc[0]],X_offspring[c1]) + r2_score(X_parents[p_loc[1]],X_offspring[c2]) < r2_score(X_parents[p_loc[0]],X_offspring[c2]) + r2_score(X_parents[p_loc[1]],X_offspring[c1]):
                # swap offspring
                offspring[c1],offspring[c2] = offspring[c2],offspring[c1]

        survivors = []
        survivor_index = []

        for i,(p,o) in enumerate(zip(parents,offspring)):
            if p.fitness >= o.fitness:
                survivors.append(copy.deepcopy(p))
                survivor_index.append(i)
            else:
                survivors.append(copy.deepcopy(o))
                survivor_index.append(i+len(parents))

        # return survivors along with their indices
        return survivors, survivor_index
