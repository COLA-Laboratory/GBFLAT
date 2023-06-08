import networkx as nx
import pandas as pd
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns

import random
import statistics
import warnings
import os 
import copy
import sys

from karateclub import HOPE, FeatherNode
from sklearn.manifold import TSNE
from typing import Any, Optional, Union, Literal, Callable, Tuple

from GBFLAT.Problems import Solution
from GBFLAT.utils import hill_climbing, neighbour_explorer_with_eva, str2list, logger_to_list, progress


class LON():
    """
    A Local Optima Network (LON)
    """

    def __init__(self):

        None

    def read_ils(
        self,
        problem_name = None, 
        nb_runs = None,
        nb_iter = None,
        n = None, 
        k = None, 
        seed = None, 
        top_dir = None, 
        directed = False, 
        weighted = False, 
        addi_attributes = False,
        file_path = None
    ) -> None:
        """
        Create LON using recorded data from iterated local search (ILS)

        Parameters
        ----------
        problem_name : {"NPP", "MaxSat", "KP"},
            The name of the problem instance. This will be used as part of the identifier
            to find/read the ILS file.  

        nb_runs : int
            Number of runs performed during ILS. This will be used as part of the identifier
            to find/read the ILS file.  
        
        nb_iter : int
            Number of maximum iterations in each run when performing ILS. This will be used 
            as part of the identifier to find/read the ILS file.  

        n : int
            The dimension of the problem instance. This will be used as part of the identifier
            to find/read the ILS file.

        k : float
            Additional control parameter of the problem. Notice that in our implementation,
            despite no control parameters are used for KP, this value is still required to be
            specified during ILS, and the same value should be written here to correctly locate 
            the ILS data.
        
        seed : int
            Random seed used when performing ILS. This will be used as part of the identifier
            to find/read the ILS file.

        top_dir : path to a directory 
            Path to the directory where the ILS data is stored. A possible example could be:
            "/home/Arwen/Downloads/TEVC_LON_MH/data/"

        directed : bool, default=False
            Whether include edge directions in the created LON
            - if True: then the resulting LON will be based on NetworkX.DiGraph(), where the an
            edge will be drawn from a source node (i.e., the current local optimum where the 
            alogrithm get stuck at) to the target ndoe (i.e., an improving move to a better local
            optimum via perturbation and further hill-climbing). This could prohibit the use of 
            methods such as nx.number_connected_components(), but will enables more node attributes 
            based on edge direction (e.g., in_degree and out_degree), as well as a better 
            understanding of the convergence of the algorithm.
            - if False: the LON will be constructed via NetworkX.Graph(), which has better 
            compatitability but will discard some the searching progress of ILS as mentioned above. 

        weighted : bool, default=Fasle,
            Whether include edge weights in the created LON 
            - if True: then weights representing the probabilites of visiting the transition between 
            the two local optima will be added to edges (as edge attributes). For each edge, the 
            associated weight is given by dividing the frequencies of visiting this transition by the
            total number of transitions encountered during ILS. Thereby, all weights should sum up to 
            1. The introducing of edge weights will also enable many node attributes, e.g., weighted 
            degree, weighted clustering coefficient. 
            - if False, no edge weights will be included in the LON (or say, all edges have equal 
            weights).

        addi_attributes : bool, default=False,
            Whether to calculate additional node attributes, e.g., clustering coefficients and various 
            centrality metrics. This could leads to additional time for creating the LON. 
        """

        # read data
        if file_path != None:
            logger = pd.read_csv(file_path,index_col=0)
        else: 
            dir = problem_name + "/" + "seed" + str(seed) + "/" + "ils_run" + str(nb_runs) + "_nimpr" + str(nb_iter) + "/"
            file_name = "logger_" + problem_name + str(n) + "_k_" + str(k) + "_seed_" + str(seed) + ".csv"
            path = top_dir + dir + file_name
            logger = pd.read_csv(path,index_col=0)
        self.raw_data = logger
        # only keep local optima 
        logger = logger[logger["position"].isin(["Improving","Initialized"])]
        # for KP, drop those with fit == 0
        # if problem_name == "KP":
        #     logger = logger[~(logger["fit"] == 0)]
        #     logger = logger[~(logger["best_fit"] == 0)]
        logger.index = range(len(logger))
        # calculate edge weights
        logger["weight"] = logger.groupby(["best_sol","sol"])["sol"].transform("count")
        logger["weight"] = logger["weight"] / len(logger)
        logger["neg_reci_weight"] = (1 / logger["weight"])
        # logger["weight"] = (logger["weight"] - logger["weight"].min()) / (logger["weight"].max() - logger["weight"].min())
        
        logger = logger.drop_duplicates(["best_sol","sol"])

        # calcualte frequency of solution
        frequencies = pd.DataFrame(self.raw_data["sol"].value_counts()).reset_index()
        frequencies.columns = ["sol","freq"]
        logger = pd.merge(left=logger,right=frequencies,left_on="sol",right_on="sol")
        # create graph 
        edges = logger[["best_sol","sol","weight","neg_reci_weight"]]
        nodes = logger.drop_duplicates("sol")

        edge_attr = ["weight","neg_reci_weight"] if weighted else None

        if directed:
            graph = nx.from_pandas_edgelist(df=edges,source="best_sol",target="sol",
                                            edge_attr=edge_attr,create_using=nx.DiGraph())
        else: 
            graph = nx.from_pandas_edgelist(df=edges,source="best_sol",target="sol",
                                            edge_attr=edge_attr,create_using=nx.Graph())
        # add node attributes
        attributes = ["sol","fit","cnt","climbing","position","freq"]
        for attribute in attributes:
            nx.set_node_attributes(graph, pd.Series(nodes[attribute].values, index=nodes["sol"]).to_dict(), attribute)
        # add length information 
        maximize = {
            "KP": True,
            "NPP": False,
            "MaxSat": False,
            "TSP": False,
        }
        avg_lenghts = cal_len(graph, maximize[problem_name], method="mean")
        min_lengths = cal_len(graph, maximize[problem_name], method="min")
        min_lengths = min_lengths.fillna(min_lengths.max())
        avg_lenghts = avg_lenghts.fillna(avg_lenghts.max())
        paths_to_go = cal_len(graph, maximize[problem_name], method="count_path")   
        nx.set_node_attributes(graph, pd.Series(avg_lenghts["avg_len"].values, index=avg_lenghts["sol"]).to_dict(), 'avg_len')
        nx.set_node_attributes(graph, pd.Series(min_lengths["min_len"].values, index=min_lengths["sol"]).to_dict(), 'min_len')
        nx.set_node_attributes(graph, pd.Series(paths_to_go["paths_to_go"].values, index=paths_to_go["sol"]).to_dict(), 'paths_to_go')

        distances = cal_dist(graph, maximize[problem_name])
        nx.set_node_attributes(graph, pd.Series(distances["avg_dist"].values, index=distances["config"].values).to_dict(), 'avg_dist')
        nx.set_node_attributes(graph, pd.Series(distances["min_dist"].values, index=distances["config"].values).to_dict(), 'min_dist')
        # remove self edges
        graph.remove_edges_from(nx.selfloop_edges(graph))

        # relabel nodes 
        graph = relabel(graph)

        # add degree information 
        degrees = dict(graph.degree())
        nx.set_node_attributes(graph, degrees, 'degree')
        if directed:
            in_degrees = dict(graph.in_degree())
            out_degrees = dict(graph.out_degree())
            nx.set_node_attributes(graph, in_degrees, 'in_degree')
            nx.set_node_attributes(graph, out_degrees, 'out_degree')

        if weighted:
            degrees_w = dict(graph.degree(weight="weight"))
            nx.set_node_attributes(graph, degrees_w, 'degree_w')
            if directed:
                in_degrees_w = dict(graph.in_degree(weight="weight"))
                out_degrees_w = dict(graph.out_degree(weight="weight"))
                nx.set_node_attributes(graph, in_degrees_w, 'in_degree_w')
                nx.set_node_attributes(graph, out_degrees_w, 'out_degree_w')

        # add additional attributes
        if addi_attributes:
            attributes = cal_metrics(graph).reset_index()
            for attribute in attributes.columns:
                nx.set_node_attributes(graph, pd.Series(attributes[attribute].values, 
                                                        index=attributes["index"]).to_dict(), attribute)

        self.graph = graph
        self.data = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
        self.edge_list = list(graph.edges())
        self.name = problem_name + str(n)


    def describe(self) -> None:
        """
        Generate basic descriptions of LON
        """

        graph = self.graph
        print("#nodes:",nx.number_of_nodes(graph))
        print("#edges:",nx.number_of_edges(graph))
        print("density:",f'{nx.density(graph):.3}')
        print("clustering:",f'{nx.average_clustering(graph):.3}')
        print("degree assortativity:",f'{nx.degree_assortativity_coefficient(graph):.3}')
        
    def cal_basin(self,instance,max_iter=100,verbose=0):
        """
        Approximately determien the (relative) basin size of each local optimum. This process 
        could be time-consuming to perform when the number of local optima exceeds 5,000.

        Parameters
        ----------
        instance : problem instance
            This is requried since hill-climbing will be performed during the sampling 
        
        max_iter : int, default=100
            Number of random walks to perform during the sampling.
            - for problems with more than 1,000 local optima, consider to reduce this value 
            in order to alleviate time-consumption. Note that since this algorithm is based on 
            sampling, thus the results will be more useful in comparing the relative size of the 
            basins of attractions rather than determing their exact sizes. Therefore, it is not 
            that bad to set this parameter to a relatively small number, though, a higher value 
            could reduce the error caused by randomness. 

        verbose : bool, default=True
            Verbosity of output message
            - if 0: silent mode 
            - if 1: print a tiny progress bar, which is very suitable for Jupyter Notebook usage
            - >= 2: print a progress message for each local optimum, could be used for .py file 
        """

        df = self.data 
        sol_list = df["sol"].values
        basin = []
        l = len(sol_list)
        i = 0
        for sol in sol_list:
            if verbose >= 2:
                print(instance.name + "-" + str(instance.n),"k:",instance.k,"seed:",instance.seed,"sol:",i,"/",l)
            sol = str2list(sol)
            basin_size, avg_walk_len, max_walk_len = get_basin(instance,sol,max_iter)
            basin.append([basin_size,avg_walk_len,max_walk_len])
            progress(i,l)
            i += 1 

        df_basin = pd.DataFrame(basin,columns=["basin_size","avg_walk_len","max_walk_len"])
        nx.set_node_attributes(self.graph, pd.Series(df_basin["basin_size"].values), 'basin_size')
        nx.set_node_attributes(self.graph, pd.Series(df_basin["avg_walk_len"].values),'avg_walk_len')
        nx.set_node_attributes(self.graph, pd.Series(df_basin["max_walk_len"].values),'max_walk_len')
        self.data = pd.DataFrame.from_dict(dict(self.graph.nodes(data=True)), orient='index')

    def draw_lon(self):
        """
        Draw LON directly. This is only recommended to be used on small graphs (e.g., <1,000 ndoes).

        Will be further elaborated.
        """
        
        nx.draw(self.graph,node_size=3)

    def save_lon(self,problem_name,nb_runs,nb_iter,n,k,seed,top_dir):
        """
        Save the LON as two datasets: a node dataset, and an edge dataset.
        """
        graph = self.graph 
        data = self.data 

        edge_list = list(graph.edges())
        edge = pd.DataFrame(data=edge_list)

        dir = "Graph/" + problem_name + "/" + "seed" + str(seed) + "/" + "ils_run" + str(nb_runs) + "_nimpr" + str(nb_iter) + "/"
        path = top_dir + dir
        if not os.path.exists(path):
            os.makedirs(path)

        data.to_csv(path + problem_name + str(nb_runs) + "x" + str(nb_iter) + "-" + \
                    str(n) + "-" + str(k) + "-seed" + str(seed) + '-NODE.csv')
        edge.to_csv(path + problem_name + str(nb_runs) + "x" + str(nb_iter) + "-" + \
                    str(n) + "-" + str(k) + "-seed" + str(seed) + '-EDGE.csv')

    def read_lon(self,problem_name,nb_runs,nb_iter,n,k,seed,top_dir,weighted=False,directed=False,include_sol=False):
        """
        Read LON from saved data.
        """

        dir = problem_name + "/" + "seed" + str(seed) + "/" + "ils_run" + str(nb_runs) + "_nimpr" + str(nb_iter) + "/"
        path = top_dir + dir

        nodes = pd.read_csv(path + problem_name + str(nb_runs) + "x" + str(nb_iter) + "-" + \
                            str(n) + "-" + str(k) + "-seed" + str(seed) + '-NODE.csv',index_col=0)
        edges = pd.read_csv(path + problem_name + str(nb_runs) + "x" + str(nb_iter) + "-" + \
                            str(n) + "-" + str(k) + "-seed" + str(seed) + '-EDGE.csv',index_col=0)

        if directed:
            graph = nx.from_pandas_edgelist(df=edges,source="0",target="1",create_using=nx.DiGraph())
        else:
            graph = nx.from_pandas_edgelist(df=edges,source="0",target="1",create_using=nx.Graph())

        for attribute in nodes.columns:
            nx.set_node_attributes(graph, pd.Series(nodes[attribute].values, index=nodes.index).to_dict(), attribute)

        self.graph = relabel(graph)
        if include_sol == False:
            self.data = nodes.drop(columns=["sol"])
        else:
            self.data = nodes
        self.name = problem_name + str(n)
        self.full_name = problem_name + "-" + str(n) + "-" + str(k) + "-" + str(seed)

    def draw_embedding(
        self,
        # model=FeatherNode(reduction_dimensions=64,svd_iterations=20,eval_points=50,order=10),
        model=HOPE(),
        reducer=TSNE(n_components=2,perplexity=10,n_jobs=-1),
        attribute=None):

        graph = self.graph 
        data = self.data

        # features = data[attribute].values.reshape(-1,1)
        # model.fit(graph,X=features)
        model = HOPE()
        model.fit(graph)

        embeddings = model.get_embedding()
        embeddings = pd.DataFrame(data=embeddings)

        embeddings_low = reducer.fit_transform(embeddings)
        embeddings_low = pd.DataFrame(data=embeddings_low)
        embeddings_low.columns=["cmp1","cmp2"]
        embeddings_low = embeddings_low.join(data)

        cmap = plt.cm.RdBu
        fig = plt.figure(figsize = (8, 7))
        plot = plt.scatter(embeddings_low["cmp1"],embeddings_low["cmp2"],c=embeddings_low["fit"],
            s=embeddings_low["degree"],linewidths=0.25,edgecolors="black",cmap=cmap)
        scatter_fig = fig.get_figure()

    def get_embedding(self,model=HOPE(),reducer=TSNE(n_components=2,perplexity=10,n_jobs=-1)):
        
        graph = self.graph 
        data = self.data

        model = HOPE()
        model.fit(graph)

        embeddings = model.get_embedding()
        embeddings = pd.DataFrame(data=embeddings)

        embeddings_low = reducer.fit_transform(embeddings)
        embeddings_low = pd.DataFrame(data=embeddings_low)
        embeddings_low.columns=["cmp1","cmp2"]
        embeddings_low = embeddings_low.join(data)
        
        return embeddings_low
    
    def remove_invalid(self):
        """
        a remedy for previous implementation of LP
        """

        nodes_to_remove = [n for n, d in self.graph.nodes(data=True) if d.get('fit', 0) == 0]
        self.graph.remove_nodes_from(nodes_to_remove)
        self.graph = relabel(self.graph)
        self.data = pd.DataFrame.from_dict(dict(self.graph.nodes(data=True)), orient='index')

    def add_attri(self,feature,name):

        nx.set_node_attributes(self.graph, feature.to_dict(), name)
        self.data = pd.DataFrame.from_dict(dict(self.graph.nodes(data=True)), orient='index')

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

def neighbor_agg(graph:nx.classes.graph.Graph, attribute:str, method="mean") -> list:
    """
    Aggregate the value of attributes across the neighborhood
    """
    local_attri_list = []
    for node in list(graph.nodes()):
        neighbor_list = list(nx.neighbors(graph,node))
        agg_list = [0]
        for neighbor in neighbor_list:
            attri = nx.nodes(graph)[neighbor][attribute]
            agg_list.append(attri)
        if method == "mean":
            result = statistics.mean(agg_list)
        if method == "sum":
            result = sum(agg_list)
        local_attri_list.append(result)
        
    return local_attri_list

def relabel(graph):
        
    # relabel nodes 
    nodes = list(graph.nodes)
    index_list = list(range(len(nodes)))
    map_dic = {}
    for index in range(len(nodes)):
        map_dic[nodes[index]] = int(index_list[index])
    graph = nx.relabel_nodes(graph,mapping=map_dic)

    return graph


def general_iterator(
    graph_list: list,
    index_list: list,
    method: Callable,
    output: Optional[bool]=True,
    axis: Optional[int]=1,
    ex_method: Optional[Callable]=None,
    ) -> Union[pd.DataFrame,list]:

    """
    general iterator to iterate through a list of graphs(items) and a list of index_names
    and perform specified operations

    Parameters
    ----------
    graph_list: list[nx.classes.graph.Graph]
        a list of NetworkX graphs

    index_list: list[str]
        a list of the corresponding graph names 

    method: any callable function
        the operation to be performed on each item, could be a operation to 
        print/save calculated metrics of each graph, or a operation to conduct 
        manipulation on each graph

    output: bool, default True
        whether the passed method return data
        False: the method only print message, or make modification to each graph,
               In this case, the general_iterator will return the modified list of 
               graphs along with its original index_list
        True: the method will create log data on each iteration, which will then be 
              combined to return a integrated data in the form of pandas dataframe

    axis: {1 for "columns", 0 for "rows"}, default 1
        only effective when output=True
        Controls how the log data are organized through the dataframe 
        if 1 (columns): the calculated values will be stored as columns in the dataframe
                        with the col_name be as the graph name
        if 2 (rows): the calculated values will be stored as rows in the dataframe
                     the graph name will as appear in each row to serve as index 
    """
    if output == True:

        if axis == 1:
            logging_data = pd.DataFrame()
            logging_data["pre_index"] = list(range(10000))

            for index in range(len(graph_list)):
                graph = graph_list[index]
                graph_name = index_list[index]        
                log_data = method(graph,graph_name,ex_method)
                for tuples in log_data:
                    data = tuples[1]
                    data_name = tuples[0]
                    logging_data[data_name] = data

            return logging_data.dropna(thresh=2).iloc[:,1:]

        if axis == 0:
            sample_graph = graph_list[0]
            sample_name = index_list[0]
            sample_logger = method(sample_graph,sample_name)
            col_name_list, sample_value_list = logger_to_list(sample_logger)
            logging_data = pd.DataFrame(columns=col_name_list)
            for index in range(len(graph_list)):
                graph = graph_list[index]
                graph_name = index_list[index]        
                log_data = method(graph,graph_name)
                string_list, value_list = logger_to_list(log_data)
                logging_data.loc[index] = value_list

            return logging_data

    if output == False:
        
        new_graph_list = []
        for index in range(len(graph_list)):
            graph = graph_list[index]
            new_graph_list.append(method(graph))

        return new_graph_list, index_list

def find_sink(graph) -> int:
    sink = 0
    for node in list(graph.nodes()):
        neighbor_list = list(nx.neighbors(graph,node))
        fit_list = []
        for neighbor in neighbor_list:
            fit = nx.nodes(graph)[neighbor]["fit"]
            fit_list.append(fit)
        if len(fit_list) != 0:
            if nx.nodes(graph)[node]["fit"] <= min(fit_list):
                sink += 1
    return sink

def describe(graph: nx.classes.graph.Graph,graph_name: str,ex_method=None) -> list:

    # data = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    logger = []
    logger.append(("graph_name",graph_name))
    logger.append(("no_nodes",graph.number_of_nodes()))
    logger.append(("no_edges",graph.number_of_edges()))
    logger.append(("density",nx.density(graph)))
    logger.append(("connec_cmp",len(list(nx.connected_components(graph)))))
    logger.append(("num_sink",find_sink(graph)))
    logger.append(("avg_cluster",nx.average_clustering(graph)))
    logger.append(("degr_assort",nx.degree_assortativity_coefficient(graph)))
    logger.append(("fit_assort",nx.numeric_assortativity_coefficient(graph,"fit")))
    # if nx.is_directed(graph):
    #     print(len(data[data["out_degree"] == 0]))    
    return logger

def adc_analysis(graph: nx.classes.graph.Graph,graph_name: str,ex_method=None) -> list:
    """
    designed function to be used within graphiterator 
    to perform adc_analysis
    """

    logger = []
    adc = nx.average_degree_connectivity(graph)
    tuple1 = (graph_name + "_adc_keys", pd.Series(adc.keys()))
    tuple2 = (graph_name + "_adc_values", pd.Series(adc.values()))
    logger.append(tuple1)
    logger.append(tuple2)
    return logger

def cdf_analysis(graph: nx.classes.graph.Graph,graph_name: str,ex_method=None) -> list:
    """
    designed function to be used within graphiterator 
    to perform cdf_analysis
    """

    logger = []
    degree = nx.degree_histogram(graph) 
    cdf_dis = nx.utils.random_sequence.cumulative_distribution(degree)
    degree = [i for i in range(len(degree))]
    del cdf_dis[0]

    tuple1 = (graph_name + "_cdf_degree", pd.Series(degree))
    tuple2 = (graph_name + "_cdf_values", pd.Series(cdf_dis))
    logger.append(tuple1)
    logger.append(tuple2)
    return logger

def rcc_analysis(graph: nx.classes.graph.Graph,graph_name: str,ex_method=None) -> list:
    """
    designed function to be used within graphiterator 
    to perform rcc_analysis
    """

    logger = []
    graph.remove_edges_from(nx.selfloop_edges(graph))
    rcc = nx.rich_club_coefficient(graph,normalized=False)
    tuple1 = (graph_name + "_rcc_keys", pd.Series(rcc.keys()))
    tuple2 = (graph_name + "_rcc_values", pd.Series(rcc.values()))
    logger.append(tuple1)
    logger.append(tuple2)
    return logger

def get_embedding(graph: nx.classes.graph.Graph,
                  graph_name: str,
                  ex_method: Callable):
    """
    designed function to be used within graphiterator 
    to perform graph embedding for a single graph
    """

    if not isinstance(graph, list): graph = [graph]
    logger = []
    # if verbose != 0:
    #     print("Now fitting",graph_name)
    connected = nx.is_connected(graph[0])
    if connected == False:
        warnings.warn("Spotted unconnected graphs, you may use the '.connect' method to get the graphs connected")
    ex_method.fit(graph)
    embedded_features = ex_method.get_embedding()
    tuple1 = (graph_name,pd.Series(embedded_features[0]))
    logger.append(tuple1)
    return logger

class GraphAnalyzer():
    """
    perform specific analysis on a list of graphs
    """

    def __init__(self,graph_list: list,index_list: list):
        self.graph_list = graph_list 
        self.index_list = index_list 

    def describe(self, addi_data=None, columns=None) -> pd.DataFrame:
        des = general_iterator(self.graph_list,self.index_list,method=describe,axis=0)
        if addi_data != None:
            mean_list = []
            for data in addi_data:
                mean = data[columns].mean()
                mean_list.append(mean)
            addi_df = pd.DataFrame(data=mean_list)
            des = des.join(addi_df)
        return des

    def adc_analysis(self) -> pd.DataFrame:
        return general_iterator(self.graph_list,self.index_list,method=adc_analysis,output=True)

    def rcc_analysis(self) -> pd.DataFrame:
        return general_iterator(self.graph_list,self.index_list,method=rcc_analysis,output=True)

    def cdf_analysis(self) -> pd.DataFrame:
        return general_iterator(self.graph_list,self.index_list,method=cdf_analysis,output=True)

    def get_embedding(self,model) -> pd.DataFrame:
        return general_iterator(self.graph_list,self.index_list,method=get_embedding,ex_method=model,output=True)

def cal_len(graph,maximize,method="mean") -> pd.Series:

    data = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    data = data.reset_index(names = ["sol_"] + list(data.columns))
    if maximize == True:
        go_list = data[data["fit"] == data["fit"].max()]["sol_"].to_list()
    else:
        go_list = data[data["fit"] == data["fit"].min()]["sol_"].to_list()
    distances = pd.DataFrame(index=data["sol_"].to_list())

    i = 0
    for go in go_list:
        try:
            distance = pd.Series(nx.shortest_path_length(graph,target=go),name=str(i))
            distances = pd.merge(left=distances,right=distance,left_index=True,right_index=True,how="outer")
            i += 1
        except:
            None

    if method == "mean":
        distances = distances.T.mean().reset_index()
        distances.columns = ["sol","avg_len"]

    if method == "min":
        distances = distances.T.min().reset_index()
        distances.columns = ["sol","min_len"]

    if method == "count_path":
        distances = distances.T.isnull().sum().reset_index()
        distances.columns = ["sol","paths_to_go"]
        distances["paths_to_go"] = len(go_list) - distances["paths_to_go"]

    return distances

def cal_dist(graph,maximize):

    data = pd.DataFrame.from_dict(dict(graph.nodes(data=True)), orient='index')
    data = data.reset_index(names = ["sol_"] + list(data.columns))
    if maximize == True:
        go_list = data[data["fit"] == data["fit"].max()]["sol_"].to_list()
    else:
        go_list = data[data["fit"] == data["fit"].min()]["sol_"].to_list()
    distances = pd.DataFrame(index=data["sol_"].to_list())

    config_list = data["sol_"].to_list()

    i = 0
    for go in go_list:
        distance_list = []
        for config in config_list:
            distance_list.append(cal_distance(str2list(go),str2list(config)))
        distance = pd.Series(distance_list,name=str(i),index=data["sol_"].to_list())
        distances = pd.merge(left=distances,right=distance,left_index=True,right_index=True)
        i += 1

    dist_mean = distances.T.mean().reset_index()
    dist_mean.columns = ["config","avg_len"]

    dist_min = distances.T.min().reset_index()
    dist_min.columns = ["config","min_len"]

    distances = pd.DataFrame(list(zip(dist_mean["avg_len"].to_list(),dist_min["min_len"].to_list())),
                            index=dist_mean["config"],columns=["avg_dist","min_dist"]).reset_index()
    distances["config"] = distances["config"].astype(str)
    return distances

def cal_distance(config1,config2):

    distance = 0
    n = len(config1)
    for i in range(n):
        distance += abs(config1[i] - config2[i])
    return distance 

def walker(sol):

    n = len(sol)
    i = random.randint(0, n - 1)
    new_sol = copy.deepcopy(sol)
    new_sol[i] = 0 if new_sol[i] == 1 else 1
    return new_sol

def get_basin(instance,sol,n_iter):

    basin_list = []
    length_list = []
    for i in range(1,n_iter+1):
        # progress(i,n_iter+1)

        length = 0
        new_solution = walker(sol)
        new_solution = Solution(instance.name,new_solution)
        if hill_climbing(instance,new_solution,neighbour_explorer_with_eva,return_steps=False).lst == sol:
            basin_list.append(new_solution.lst)
            length += 1
            while hill_climbing(instance,new_solution,neighbour_explorer_with_eva,return_steps=False).lst == sol:
                basin_list.append(new_solution.lst)
                new_solution = walker(new_solution.lst)
                new_solution = Solution(instance.name,new_solution)
                length += 1
            length_list.append(length)
        else:
            length_list.append(length)
        basin = pd.Series(basin_list)
        basin = basin.astype(str)
        basin = basin.drop_duplicates()

    return len(basin), statistics.mean(length_list), max(length_list)


def cal_metrics(graph):

    
    thresh = 1000
    if nx.number_of_nodes(graph) > thresh:
        betweenness_centrality = pd.Series(nx.betweenness_centrality(graph,k=thresh)) 
    else:
        betweenness_centrality = pd.Series(nx.betweenness_centrality(graph)) 

    attri_data = pd.DataFrame()
    # add centrality metrics
    attri_data["betw_centr"] = betweenness_centrality
    attri_data["egv_centr"] = pd.Series(nx.eigenvector_centrality(graph,max_iter=10000))
    attri_data["close_centr"] = pd.Series(nx.closeness_centrality(graph))
    attri_data["dgr_centr"] = pd.Series(nx.degree_centrality(graph))
    attri_data["pagerank"] = pd.Series(nx.pagerank(graph))

    # add clustering metrics
    # attri_data["triangles"] = pd.Series(nx.triangles(graph))
    # attri_data["node_clique"] = pd.Series(nx.node_clique_number(graph))
    attri_data["clustering"] = pd.Series(nx.clustering(graph))
    attri_data["avg_neighb_dgr"] = pd.Series(nx.average_neighbor_degree(graph))   

    # adding neighborhood fitness aggregation
    attri_data["fit_agg"] = neighbor_agg(graph,"fit",method="mean")
    
    weighted = nx.is_weighted(graph)
    if weighted:
        attri_data["clustering_w"] = pd.Series(nx.clustering(graph,weight="weight")) 
        attri_data["avg_neighb_dgr_w"] = pd.Series(nx.average_neighbor_degree(graph,weight="weight")) 
        attri_data["pagerank_w"] = pd.Series(nx.pagerank(graph,weight="weight"))

    return attri_data
