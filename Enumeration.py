import copy
import random 
import math
import pandas as pd
import networkx as nx
from GBFLAT.Problems import Solution
from GBFLAT.utils import neighbour_explorer, allBits, list_to_str, progress
from GBFLAT.utils import hill_climbing, neighbour_explorer, neighbour_explorer_with_eva


def enumerate_lon(instance,distance=2,return_graph=True,progress_bar=True):

    """
    performing enumeration on problem instance in a exhuastive search manner 
    """
    logger = pd.DataFrame()
    sol_list, fit_list, neighour_list, neighbour_fit_list, best_neighbour_list = [], [], [], [], []
    is_local_optima_list = []
    length = len(allBits(instance.n))
    
    i = 0
    for sol in allBits(instance.n):


        sol = Solution(problem_name=instance.name,lst=sol)
        sol_list.append(sol.lst)
        # calculate fintess
        instance.full_eval(sol) 
        fit_list.append(sol.fitness)
        # exploring neighbourhood
        neighbours = neighbour_explorer(sol.lst)
        neighbour_fits = []
        for neighbour in neighbours:
            neighbour = Solution(problem_name=instance.name,lst=neighbour)
            instance.full_eval(neighbour)
            neighbour_fits.append(neighbour.fitness)
        best_neighbour_fit = max(neighbour_fits) if instance.maximize() else min(neighbour_fits)
        # determine whether current solution is local optima 
        if instance.maximize():
            is_local_optima_list.append(True) if sol.fitness >= best_neighbour_fit else is_local_optima_list.append(False)
        else:
            is_local_optima_list.append(True) if sol.fitness <= best_neighbour_fit else is_local_optima_list.append(False)
        # record data
        neighour_list.append(neighbours)
        neighbour_fit_list.append(neighbour_fits)
        best_neighbour_list.append(best_neighbour_fit)
        if progress_bar:
            progress(i, length, status='')
        i += 1

    logger["sol"] = sol_list
    logger["fitness"] = fit_list
    logger["neighbours"] = neighour_list
    logger["neighbour_fit"] = neighbour_fit_list
    logger["best_neighbour_fit"] = best_neighbour_list
    logger["is_local_optima"] = is_local_optima_list

    lo_list = logger[logger["is_local_optima"] == True]["sol"]
    lo_list = list(lo_list.values)

    lon = logger[logger["is_local_optima"] == True]
    

    edge_list = []
    edge_list_with_fit = []
    
    length = len(lon)
    i = 0
    for index, row in lon.iterrows():
        sol = row["sol"]
        sol = Solution(problem_name=instance.name,lst=sol)
        all_neighbours = neighbour_explorer(sol.lst,distance=distance)

        for neighbour in all_neighbours:
            if neighbour in lo_list:
                neighbour = Solution(problem_name=instance.name,lst=neighbour)
                edge_list.append((list_to_str(sol.lst),list_to_str(neighbour.lst)))
                instance.full_eval(sol)
                instance.full_eval(neighbour)
                edge_list_with_fit.append(((sol.lst,sol.fitness),(neighbour.lst,neighbour.fitness)))
        if progress_bar:
            progress(i, length, status='')
        i += 1 
                
    graph = nx.Graph()
    for i in range(len(lo_list)):
        lo_list[i] = list_to_str(lo_list[i])
    
    sol_list = []
    lon_for_graph = copy.copy(lon)
    for i in range(len(lon_for_graph)):
        sol_list.append(list_to_str(lon_for_graph["sol"].values[i]))
    lon_for_graph["sol"] = sol_list
    lon_for_graph.index = lon_for_graph["sol"]
    lon_for_graph = lon_for_graph.drop(columns=["sol","neighbours","neighbour_fit","is_local_optima"])
    
    attri_node_list = df_node_list(lon_for_graph)

    graph.add_nodes_from(attri_node_list)
    graph.add_edges_from(edge_list)

    node_list = list(lon["sol"].values)
    for i in range(len(node_list)):
        node_list[i] = list_to_str(node_list[i])
    index_list = list(range(len(node_list)))
    map_dic = {}
    for index in range(len(node_list)):
        map_dic[node_list[index]] = index_list[index]

    graph = nx.relabel_nodes(graph,mapping=map_dic)

    # add degree information to lon
    degree_list = []
    for tuple in graph.degree():
        degree_list.append(tuple[1])
    lon["degree"] = degree_list
    
    lon["sol"] = lon["sol"].astype("str")
    lon.index = range(len(lon))
    if return_graph == True:
        return graph, lon[["sol","fitness","best_neighbour_fit","degree"]], logger
    else: 
        return lon[["sol","fitness","best_neighbour_fit","degree"]], logger

def enumerate_hc(instance, progress_bar=True):
    """
    performing enumeration on problem instance in a hill-climbing manner.
    could be used to determine the exact size of the basin of attraction of each local optima
    """
    sol_list, sol_fit_list, lo_list, lo_fit_list = [], [], [], []

    length = len(allBits(instance.n))
    i = 0
    for sol in allBits(instance.n):
        
        # initialize solution class and determine fitness
        sol = Solution(problem_name=instance.name,lst=sol)
        instance.full_eval(sol) 
        sol_list.append(sol.lst)
        sol_fit_list.append(sol.fitness)
        # performing hill-climbing
        lo = hill_climbing(instance,sol,neighbour_explorer_with_eva,return_steps=False)
        lo_list.append(lo.lst)
        lo_fit_list.append(lo.fitness)
        if progress_bar:
            progress(i, length, status='')
        i += 1
        
    logger = pd.DataFrame(
        data=list(zip(sol_list,sol_fit_list,lo_list,lo_fit_list)),
        columns=["sol","sol_fit","lo","lo_fit"])

    df = pd.DataFrame(logger["lo"].value_counts())
    df = df.reset_index()
    df.columns = ["lo","basin_size"]
    lo_dropped_list = df["lo"]
    lo_dropped_fit_list = []
    for lo in lo_dropped_list:
        lo = Solution(problem_name=instance.name,lst=lo)
        instance.full_eval(lo)
        lo_dropped_fit_list.append(lo.fitness)
    df["lo_fit"] = lo_dropped_fit_list
    df["lo"] = df["lo"].astype("str")
        
    return df

def create_graph_from(node_data,edge_data):

    graph = nx.Graph()
    graph.add_nodes_from(node_data)
    graph.add_edges_from(edge_data)
    graph.remove_edges_from(nx.selfloop_edges(graph))

    return graph

def df_node_list(data):

    _data = data.to_dict(orient="index")
    attri_node_list = []
    for item in _data.items():
        attri_node_list.append(item)

    return attri_node_list