import random
import copy
from itertools import combinations
from typing import Callable, Optional, Tuple, Union
from matplotlib.pyplot import scatter
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import statistics 
import networkx as nx
import sys

def random_list_gen(lower:int,upper:int,length:int):
    random_list = []
    for i in range(0,length):
        n = random.randint(lower,upper)
        random_list.append(n)
    return random_list

def _allBits(n:str):
    """
    this is an internal function used by function allBits
    """
    if n: yield from ( bits+bit for bits in _allBits(n-1) for bit in ("0","1") )
    else: yield ""

def allBits(n:int,bit_type="int"):
    """
    returns all possible binary solutions of length n

    Parameters:
    -----------
    n: length of solution
    bit_type: the type of element for each bit in a solution, e.g., [1,0,1,1] or ["1","0","1","1"]
    return: a list of all possible binary solutions of length n, each of which is in the form of a binary list
    """
    comb_list = []
    for comb in _allBits(n):
        l = []
        for element in comb:
            if bit_type == "int":
                l.append(int(element))
            if bit_type == "str":
                l.append(element)
        comb_list.append(l)
    return comb_list

def random_list_gen(lower,upper,length):
    random_list = []
    for i in range(0,length):
        n = random.randint(lower,upper)
        random_list.append(n)
    return random_list

def list_to_str(sol):
    sol_str = ""
    for item in sol:
        sol_str += str(item)
    return sol_str

def is_local_optimum(instance,sol):
    """
    determine whether a given a solution is a local optimum
    """
    is_lo = True
    for i in range(len(sol.lst)):
        sol_copy = copy.deepcopy(sol)
        sol_copy.lst[i] = 1 if sol.lst[i] == 0 else 0
        instance.full_eval(sol_copy)
        instance.full_eval(sol)
        if sol_copy.fitness < sol.fitness:
            is_lo = False
    if is_lo == False:
        print("RESULT: it is NOT a local optimum")
    else:
        print("RESULT: it is a local optimum")

def neighbour_explorer_with_eva(instance,sol,distance=1,as_lo_checker=False):

    assert(len(sol.lst) == instance.n)
    instance.full_eval(sol)
    n = instance.n

    # determine neighbour set
    neighbours = []
    for j in range(1,distance+1):
        neighbours_ = []
        posi_list = list(combinations(list(range(0,n)), j))
        for comb in posi_list:
            flip_sol = copy.deepcopy(sol)
            for i in range(j):
                posi = comb[i]
                flip_sol.lst[posi] = 0 if sol.lst[posi] == 1 else 1
            neighbours_.append(flip_sol)
        neighbours = neighbours + neighbours_

    # evaluate the fitness of neighbours
    neighbour_fit_list = []
    for neighbour in neighbours:
        instance.full_eval(neighbour)
        neighbour_fit_list.append(neighbour.fitness)

    # find best neighbour
    best_neighbour_fit = max(neighbour_fit_list) if instance.maximize() else min(neighbour_fit_list)
    best_neighbour_index = neighbour_fit_list.index(best_neighbour_fit)
    best_neighbour = neighbours[best_neighbour_index]

    # determine whether improvement has been made
    improved = False
    if instance.maximize():
        improved = True if sol.fitness < best_neighbour_fit else False

    else:
        improved = True if sol.fitness > best_neighbour_fit else False


    # return
    if as_lo_checker == False:
        if improved: 
            return best_neighbour, improved
        else:
            return sol, improved
        
    if as_lo_checker == True:
        print("initial solution:",sol.lst)
        print("fitness of initial solution:",sol.fitness)
        print("fitness of neighbours:",neighbour_fit_list)
        print("fitness of best neighbour:",best_neighbour_fit)
        print("best neighbour:",best_neighbour.lst)

def hill_climbing(instance,init_sol,neighbour_explorer,mute=True,return_steps=True):

    steps = 0
    instance.full_eval(init_sol)
    if mute != True:
        print("step:",steps,"sol:",init_sol.lst,"fit:",init_sol.fitness)
    sol, improved = neighbour_explorer(instance, init_sol)
    steps += 1 if improved else 0
    if mute != True:
        print("step:",steps,"sol:",sol.lst,"fit:",sol.fitness)
    while improved:
        sol, improved = neighbour_explorer(instance, sol)
        if improved:
            steps += 1
            if mute != True:
                print("step:",steps,"sol:",sol.lst,"fit:",sol.fitness)
    if return_steps:
        return sol, steps
    else:
        return sol

def draw_embedding(embeddings_low):

    cmap = plt.cm.RdBu
    fig = plt.figure(figsize = (10, 8))

    plot = scatter(
        embeddings_low["cmp1"],embeddings_low["cmp2"],
        c=embeddings_low["fit"],
        s=embeddings_low["degree"],
        linewidths=0.25,
        edgecolors="black",
        cmap=cmap)

    scatter_fig = fig.get_figure()

def str2list(sol):

    new_sol = []
    for item in sol:
        if item in ["0","1"]:
            new_sol.append(int(item))
    return new_sol

def logger_to_list(logger: list) -> list:

    value_list = []
    string_list = []
    for item in logger:
        value_list.append(item[1])
        string_list.append(item[0])

    return string_list, value_list


def add_attribute(graph, data, attri_data):
        
    """
    attri_data: pd.DataFrame
        the index of attri_data must be range(0,len(data))
    """
    if isinstance(attri_data,list):
        attri_data = pd.Series(attri_data)
    
    assert len(attri_data) == len(data), "The length of attri_data does not match with the original graph data."

    # extract edge data
    edge_list = list(graph.edges())
    # add attri_data
    data = data.merge(attri_data,left_index=True, right_index=True)
    # convert node data to list
    _data = data.to_dict(orient="index")
    attri_node_list = []
    for item in _data.items():
        attri_node_list.append(item)
    
    graph = nx.Graph()
    graph.add_nodes_from(attri_node_list)
    graph.add_edges_from(edge_list)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph

def get_reduction(embeddings: pd.DataFrame, reducer) -> pd.DataFrame:
    """
    get the embedded data in lower dimensions through techniques like t-SNE, UMAP or PCA
    """
    embeddings_low = reducer.fit_transform(embeddings)
    embeddings_low = pd.DataFrame(data=embeddings_low)
    embeddings_low.columns=["cmp1","cmp2"]
    return embeddings_low

def plot_residual(df):

    # plot the mean as a solid line
    plt.plot(df['mean'], color='red', linewidth=1)
    # shade the region between the 0.25 and 0.75 quantiles
    plt.fill_between(df.index, df['q25'], df['q75'], alpha=0.2, color='red',linewidth=0.0)
    # add labels and title
    plt.xlabel('Fitness')
    plt.ylabel('Residual (Mean)')
    # plt.title('Title of the plot')

    # display the plot
    plt.show()

def progress(count, total, status=''):

    bar_len = 50
    filled_len = int(round(bar_len * count / float(total)))

    percents = round(100.0 * count / float(total), 1)
    bar = '|' * filled_len + '_' * (bar_len - filled_len)

    sys.stdout.write('\r%s %s%s %s' % (bar, percents, '%', status))
    sys.stdout.flush()  

def parity_checker(sol1: list,sol2: list) -> int:
        """
        calculate the number of mismatch bits for two solutions
        """

        if len(sol1) != len(sol2):
            raise Exception('The lengths of the two input do not match')
        length = len(sol1) 
        mismatch = 0
        for index in range(length):
            if sol1[index] != sol2[index]:
                mismatch += 1
        return mismatch 

def two_opt_local_search(instance, sol, distance=None, as_lo_checker=False):

    assert(len(sol.lst) == instance.n)
    instance.full_eval(sol)
    n = instance.n 
    neighbours, neighbour_fit_list = [], []
    for i in range(1, n - 2):
        for j in range(i + 1, n):
            if j - i == 1:
                continue
            new_sol = copy.deepcopy(sol)
            new_sol.lst[i:j] = reversed(sol.lst[i:j])
            instance.full_eval(new_sol)
            if new_sol.lst != sol.lst:
                neighbours.append(new_sol)
                neighbour_fit_list.append(new_sol.fitness)

    # find the best neighbour 
    best_neighbour_fit = min(neighbour_fit_list)
    best_neighbour_index = neighbour_fit_list.index(best_neighbour_fit)
    best_neighbour = neighbours[best_neighbour_index]

    improved = True if sol.fitness > best_neighbour_fit else False
    is_lo = True if sol.fitness <= best_neighbour_fit else False

    # return
    if as_lo_checker == False:
        if improved: 
            return best_neighbour, improved
        else:
            return sol, improved 
    else:
        print("is local optimum:",str(is_lo))
        print("initial solution:",sol.lst)
        print("fitness of initial solution:",sol.fitness)
        print("fitness of neighbours:",neighbour_fit_list)
        print("fitness of best neighbour:",best_neighbour_fit)
        print("best neighbour:",best_neighbour.lst)


def two_opt_neighbor_explorer(sol):

    n = len(sol)
    neighbours = []
    for i in range(1, n - 2):
        for j in range(i + 1, n):
            if j - i == 1:
                continue
            new_sol = copy.deepcopy(sol)
            new_sol[i:j] = reversed(sol[i:j])
            if new_sol != sol:
                neighbours.append(new_sol)
    return neighbours

def two_opt_neighbour_explorer_ils(instance, sol, distance=None, as_lo_checker=False):

    assert(len(sol.lst) == instance.n)
    instance.full_eval(sol)
    n = instance.n 
    neighbours, neighbour_fit_list = [], []

    for i in range(1, n - 2):
        for j in range(i + 1, n):
            if j - i == 1:
                continue
            new_sol = copy.deepcopy(sol)
            new_sol.lst[i:j] = reversed(sol.lst[i:j])
            instance.full_eval(new_sol)
            if new_sol.lst != sol.lst:
                neighbours.append(new_sol)
                neighbour_fit_list.append(new_sol.fitness)
        #     if new_sol.lst < sol.lst:
        #         break
        # if new_sol.lst < sol.lst:
        #     break


    # find the best neighbour 
    best_neighbour_fit = min(neighbour_fit_list)
    best_neighbour_index = neighbour_fit_list.index(best_neighbour_fit)
    best_neighbour = neighbours[best_neighbour_index]

    improved = True if sol.fitness > best_neighbour_fit else False
    is_lo = True if sol.fitness <= best_neighbour_fit else False

    # return
    if as_lo_checker == False:
        if improved: 
            return best_neighbour, improved
        else:
            return sol, improved 
    else:
        print("is local optimum:",str(is_lo))
        print("initial solution:",sol.lst)
        print("fitness of initial solution:",sol.fitness)
        print("fitness of neighbours:",neighbour_fit_list)
        print("fitness of best neighbour:",best_neighbour_fit)
        print("best neighbour:",best_neighbour.lst)

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


import contextlib
import joblib
from tqdm import tqdm
from joblib import Parallel, delayed

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()
        
# import statistics

# def basin_explorer(instance,sol,sol_neighbor):

#     # sol = Solution(instance.name,sol)
#     instance.full_eval(sol)
#     continued = True
#     neighbour_lst, neighbour_fits = neighbour_explorer_with_eva(instance,sol_neighbor)
#     if statistics.min(neighbour_fits) < sol.fitness:
#         continued = False
#         weight = 0
#     elif statistics.min(neighbour_fits) > sol.fitness:
#         weight = 1
#     else:
#         min_fitness = statistics.min(neighbour_fits)
#         weight = 1 / (1 + neighbour_fits.count(min_fitness))
#     return continued, weight 

# def cal_basin(instance,lo):
#     lo = Solution(instance.name,lo)
#     neighbour_lst, neighbour_fits = neighbour_explorer_with_eva(instance,lo)
#     for neighbour in neighbour_lst:
#         continued, weight = basin_explorer(instance,sol,neighbour)