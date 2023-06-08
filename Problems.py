
import numpy as np
import pandas as pd
import networkx as nx
import random
import math
import copy

from itertools import combinations
from typing import Callable, Optional, Tuple, Union
from abc import ABC, abstractmethod
from GBFLAT.utils import random_list_gen


class ProblemInstance(ABC):

    @abstractmethod

    def maximize(self):
        pass

    def strictly_better(self, a, b):
        return a > b if self.maximize() else a < b

    def better_or_equal(self, a, b):
        return a >= b if self.maximize() else a <= b

    @abstractmethod
    def full_eval(self, sol):
        pass

    @abstractmethod
    def has_flip_delta_eval(self):
        pass

    @abstractmethod
    def flip_delta_eval(self, i):
        pass

    @abstractmethod
    def flip_with_delta(self, i):
        pass

    @abstractmethod
    def two_rnd_flips(self):
        pass


class Solution:
    """Represents a solution.

    Parameters
    ---------
    problem_name: str
        the name associated with the problem
        must be the return of the problem_instance.name attribute

    lst: Optional, list
        solution representation

    fitness: Optional, default 0
        the fitness value associated to the solution

    invalid: Optional, bool, default False
        the invalid flag should be set true if the fitness
        needs to be recomputed following some modification
        of the solution
    """

    def __init__(
        self, 
        problem_name:str, 
        lst:Optional[list]=[], 
        fitness:Optional[int]=0, 
        invalid:Optional[bool]=False):

        self.fitness = fitness
        self.invalid = invalid
        self.lst = lst
        self.problem_name = problem_name

    def __str__(self) -> str:
        #return str(self.fitness) + (" (invalid) " if self.invalid else " ") + ','.join(str(i) for i in self.lst)
        return str(self.fitness) + " " + str(self.lst)

    def init_rnd_bitstring(self, n:int):
        """
        Initialize the lst attribute to a uniformly random bitstring of length n.

        Parameters
        ---------
        n: length of the bitstring
        """
        if self.problem_name == "NPP":
            self.lst = [random.randint(0, 1) for i in range(n)]
        if self.problem_name == "MaxSat":
            self.lst = [random.randint(0, 1) for i in range(n)]
        if self.problem_name == "KP":
            self.lst = [random.randint(0, 1) for i in range(n)]
        if self.problem_name == "TSP":
            self.lst = list(range(n))
            random.shuffle(self.lst)

class Max_Sat(ProblemInstance):
    """
    Instance generator for Max Satisfiability problem, 

    Parameters
    --------- 
    n: number of variables
    k: number of clauses / number of variables
    seed: random seed
    """

    def __init__(self, n:int, k:int, seed:Optional[int]=42):
        
        self.n = n
        self.k = k
        self.seed = seed
        self.name = "MaxSat"
        rnd_gen = random.Random()
        rnd_gen.seed(seed)
        n_clause = int(n * k)

        comb_list = list(combinations(list(range(-n+1,n)), 3))
        posi_list = [rnd_gen.randrange(0, len(comb_list)) for j in range(n_clause)]
        clause_list = []
        for i in range(n_clause):
            clause_list.append(comb_list[posi_list[i]])
            
        self.clause_list = clause_list

    def maximize(self):
        return False

    def full_eval(self, sol:Callable):
        """
        evaluate the fitness of a given solution 
        """
        self.n = len(sol.lst)
        cnt = 0
        for clause in self.clause_list:
            l = []
            for literal in clause:
                if literal < 0:
                    l.append(1) if sol.lst[abs(literal)] == 0 else l.append(0)
                else:
                    l.append(sol.lst[literal])
            if sum(l) != 0:
                cnt += 1
        sol.fitness = self.n * self.k - cnt
        sol.invalid = False
    
    @staticmethod
    def has_flip_delta_eval():
        return False

    @staticmethod
    def flip_delta_eval(self, i):
        raise NotImplementedError()

    @staticmethod
    def flip_with_delta(self, i):
        raise NotImplementedError()

    def two_rnd_flips(self, sol:Callable, distance:Optional[int]=3):
        """
        draw random sample from k-bit neighbours, where k = distance
        an alternative implementation could be used via neighbour_explorer function
        """
        
        bits = random.sample(list(range(len(sol.lst))), distance)
        for bit in bits:
            sol.lst[bit] = 0 if sol.lst[bit] == 1 else 1
        self.full_eval(sol)


class Knapsack(ProblemInstance):
    """
    Instance generator for Knapsack problem, 

    Parameters
    --------- 
    n: number of variables
    k: NaN
    seed: NaN
    """

    def __init__(self,n,k=0,seed=0):

        self.name = "KP"
        self.n = n
        self.k = k
        self.seed = seed

        rnd_gen = random.Random()
        rnd_gen.seed(seed)
        # inverse strongly correlated KP instances
        if k == "inv_strong_corr":
            value_list = [rnd_gen.randrange(1, 1000) for j in range(n)]
            weight_list = [min(1000, value_list[j] + 100) for j in range(n)]
        else:
            weight_list = [rnd_gen.randrange(1, 1000) for j in range(n)]
            # uncorrelated KP instances
            if k == "un_corr":
                value_list = [rnd_gen.randrange(1, 1000) for j in range(n)]
            # weakly correlated KP instances
            if k == "weak_corr" :
                value_list = [rnd_gen.randrange(
                    max(1, weight_list[j] - 100), 
                    min(1000, weight_list[j] + 100)
                    ) for j in range(n)]
            # strongly correlated KP instances
            if k == "strong_corr":
                value_list = [min(1000, weight_list[j] + 100) for j in range(n)]
        item_list = [(weight_list[i],value_list[i]) for i in range(n)]
        self.items = item_list 
        self.c = 0.55 * sum([weight for weight in weight_list])

    def maximize(self):
        return True

    def full_eval(self, sol):
        l = len(sol.lst)
        assert(l == self.n)
        weight = sum([sol.lst[i] * self.items[i][1] for i in range(l)])
        if weight > self.c:
            fitness = 0
            sol.invalid = True
        else:
            fitness = sum([sol.lst[i] * self.items[i][0] for i in range(l)])
            sol.invalid = False
        sol.fitness = fitness
        sol.weight = weight
        

    def weight(self, sol:Callable):
        """
        evaluate the fitness of solution
        """
        return sum([sol.lst[i] * self.items[i][1] for i in range(len(sol.lst))])

    @staticmethod
    def has_flip_delta_eval():
        return False

    @staticmethod
    def flip_delta_eval(sol, i):
        raise NotImplementedError()

    @staticmethod
    def flip_with_delta(sol, i, delta_fitness):
        raise NotImplementedError()

    def two_rnd_flips(self, sol:Callable, distance:Optional[int]=2):
        """
        draw random sample from k-bit neighbours, where k = distance
        an alternative implementation could be used via neighbour_explorer function
        """
        
        bits = random.sample(list(range(len(sol.lst))), distance)
        for bit in bits:
            sol.lst[bit] = 0 if sol.lst[bit] == 1 else 1
        self.full_eval(sol)
        if sol.invalid == True:
            while sol.invalid != False:
                bits = random.sample(list(range(len(sol.lst))), distance)
                for bit in bits:
                    sol.lst[bit] = 0 if sol.lst[bit] == 1 else 1
                self.full_eval(sol)


class NumberPartitioning(ProblemInstance):

    def __init__(self, n, k, seed):
        """
        Instance generator for number partitioning problem

        Parameters
        --------- 
        n: number of variables
        k: number of clauses / number of variables
        seed: random seed
        """
        self.name = "NPP"
        rnd_gen = random.Random()
        rnd_gen.seed(seed)
        self.n = n
        self.k = k
        self.seed = seed
        l = int(round(math.pow(2, n*k)))
        self.a = [rnd_gen.randrange(1, l+1) for j in range(n)]

    def __str__(self):
        return "NPP n=" + str(self.n) + " k=" + str(self.k) + " seed=" + str(self.seed) + " " + str(self.a)

    def maximize(self):
        return False

    def full_eval(self, sol):

        l = len(sol.lst)
        assert (l == self.n)
        cost1 = sum([self.a[i] for i in range(l) if sol.lst[i] == 0])
        cost2 = sum([self.a[i] for i in range(l) if sol.lst[i] == 1])
        sol.fitness = cost1 - cost2 if cost1 > cost2 else cost2 - cost1
        sol.invalid = False

    @staticmethod
    def has_flip_delta_eval():
        return False

    @staticmethod
    def flip_delta_eval(self, i):
        raise NotImplementedError()

    @staticmethod
    def flip_with_delta(self, i):
        raise NotImplementedError()

    def two_rnd_flips(self, sol:Callable, distance:Optional[int]=2):
        """
        draw random sample from k-bit neighbours, where k = distance
        an alternative implementation could be used via neighbour_explorer function
        """
        bits = random.sample(list(range(len(sol.lst))), distance)
        for bit in bits:
            sol.lst[bit] = 0 if sol.lst[bit] == 1 else 1
        self.full_eval(sol)




class TSP(ProblemInstance):

    def __init__(self, n=None, k=None, seed=None, cities=None, kind=None, name=None):
        """
        Instance generator for traveling salesman problem. In particular, here we 
        implement random Eulidean TSP instances, where the location of each city is 
        randomly drawn in a k * k coordinate.

        Parameters
        ----------
        n: number of cities to visit
        k: the length of the coordinate, i.e., the location of each city will be 
        limited in a (k,k) square whose lower left corner is located at the origin.
        seed: random seed, to allow for repetition.
        """
        # initialize basic parameters
        if name != None:
            self.name = name
        else: 
            self.name = "TSP"
            
        random.seed(seed)
        self.n = n
        self.k = k
        self.seed = seed 

        if cities == None:

            # generate random coordinates for a list of cities
            cities = []
            for i in range(n):
                x = random.uniform(0, n * 100)
                y = random.uniform(0, n * 100)
                cities.append((x, y))
        else:
            n = len(cities)
            self.n = n
            print("number of cities:",n)

        # generate a distance matrix
        dist_matrix = np.zeros((n, n))
        for i in range(n):
            for j in range(n):
                x1, y1 = cities[i]
                x2, y2 = cities[j]
                if kind == "ATT":
                    dist_matrix[i][j] = int(math.sqrt(((x2 - x1) ** 2 + (y2 - y1) ** 2) / 10))
                else: 
                    dist_matrix[i][j] = int(math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))
        self.dist_matrix = dist_matrix

    def maximize(self):
        """
        TSP is a minimization problem
        """
        return False
    
    @staticmethod
    def has_flip_delta_eval():
        return False

    @staticmethod
    def flip_delta_eval(self, i):
        raise NotImplementedError()

    @staticmethod
    def flip_with_delta(self, i):
        raise NotImplementedError()
    
    def full_eval(self, sol):
        """
        Evaluate the fitness value of a given solution, which in case of TSP, 
        is the total distance of the tour. 
        """
        total_distance = 0
        # for each city except the last one, add the distance to its predecessor, 
        for i in range(self.n - 1):
            total_distance += self.dist_matrix[sol.lst[i]][sol.lst[i + 1]]

        # add distance from last city to first city
        total_distance += self.dist_matrix[sol.lst[self.n - 1]][sol.lst[0]]
        sol.fitness = int(total_distance)
        sol.invalid = False

    def two_rnd_flips(self, sol, distance=None):
        """
        TO IMPLEMENT: the name of this function should be changed to "perturb"
        Perturb the solution. Here double-bridge kick is implemented. 

        Perform double bridge perturbation on a solution. 
        """
        # get the indicies to perform perturbation
        perturb_indicies = get_perturbation_indices(self.n)
        # perform perturbation 
        sol.lst = apply_double_bridge(sol.lst, perturb_indicies)
        # evaluate the fitness of the new solution 
        self.full_eval(sol)


def get_perturbation_indices(n):
    """
    Returns a pair of a pair of indices representing a double-bridge perturbation.
    Each pair represents 2 edges removed in a non-sequential 2-opt move.
    An index i represents the edge from i to i+1.

    Parameters
    ----------
    n: the number of cities of the problem instance.
    """

    assert(n > 4) # minimum of 5 edges to perform a non-trivial perturbation.
    first = random.randrange(n)

    # there must be a gap of at least one edge between first and this edge.
    # there are 3 edges that cannot be chosen, first, first + 1, and first - 1.
    second = random.randrange(n - 3)
    second = (first + 2 + second) % n

    # normalize first to be min and second to be max.
    pair = (first, second)
    first = min(pair)
    second = max(pair)

    # get third edge, in between first and second.
    assert(second - first > 1)
    third = random.randrange(first + 1, second)
    excluded_edges = second - first + 1
    available_edges = n - excluded_edges

    # get fourth edge outside of first and second edge.
    fourth = random.randrange(available_edges)
    fourth = (second + 1 + fourth) % n

    index_set = set([first, second, third, fourth])
    if len(index_set) != 4:
        print([first, second, third, fourth])
    assert(len(index_set) == 4)

    assert(third - first > 0)
    assert(second - third > 0)
    assert(fourth > second or fourth < first)

    return ((first, second), (third, fourth))

def apply_double_bridge(tour, indices):
    """
    Apply a double-bridge (4-opt) kick on a given solution of TSP,
    which is usually a local optimum tour. 

    Parameters
    ----------
    tour: the initial solution to be perturbed.
    indices: the indices of the cities to apply the perturbation. This 
    should be generated using the `get_perturbation_indices` function. 
    """

    pair1, pair2 = indices
    first, second = pair1
    third, fourth = pair2
    seg1 = tour[:first + 1]
    seg2 = tour[first + 1:third + 1]
    seg3 = tour[third + 1:second + 1]
    seg4 = tour[second + 1:]

    assert(len(seg1) + len(seg2) + len(seg3) + len(seg4) == len(tour))

    if fourth < first:
        seg0 = seg1[:fourth + 1]
        seg1 = seg1[fourth + 1:]
        new_tour = seg0 + seg3 + seg2 + seg1 + seg4

    else:
        assert(fourth > second)
        seg5 = seg4[fourth - second:]
        seg4 = seg4[:fourth - second]
        new_tour = seg1 + seg4 + seg3 + seg2 + seg5

    assert(not same_tour(tour, new_tour))

    return new_tour

def same_tour(t1, t2):
    """
    determine whether two tours are the same 
    """

    for i in range(len(t1)):
        if t1[i] != t2[i]:
            return False
    return True

def double_bridge(sol):

    n = len(sol)
    # get the indicies to perform perturbation
    perturb_indicies = get_perturbation_indices(n)
    # perform perturbation 
    sol = apply_double_bridge(sol, perturb_indicies)
    # evaluate the fitness of the new solution 
    return sol 

def one_bit_flip(sol):

    neighbours = neighbour_explorer(sol,distance=1)
    sol = random.choice(neighbours)
    return sol

def neighbour_explorer(sol,distance=1):
    """
    return all d-flip neighbours of a given binary solution
    """
    n = len(sol)
    neighbours = []
    for j in range(1,distance+1):
        neighbours_ = []
        posi_list = list(combinations(list(range(0,n)), j))
        for comb in posi_list:
            flip_sol = copy.deepcopy(list(sol))
            for i in range(j):
                posi = comb[i]
                flip_sol[posi] = 0 if sol[posi] == 1 else 1
            neighbours_.append(flip_sol)
        neighbours = neighbours + neighbours_

    return neighbours