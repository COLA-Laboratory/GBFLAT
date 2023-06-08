import copy
import os
import random
import pandas as pd
import numpy as np
import colorama

from typing import Callable, Optional, Tuple, Union
from itertools import combinations
from GBFLAT.Problems import Solution
from GBFLAT.utils import hill_climbing, neighbour_explorer_with_eva, progress, two_opt_neighbour_explorer_ils, tqdm_joblib
from tqdm import tqdm
from joblib import Parallel, delayed

def _parity_checker(sol1: list,sol2: list) -> int:
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

def _ils_message(instance,run,non_improvement_cnt,best_lo,curr_sol,posi,dumb):

    if dumb != True:
        print(instance.name + "-" + str(instance.n).ljust(3), \
            "k:",str(instance.k).ljust(3),"runs:",str(run).ljust(3), \
            "cnt:",str(non_improvement_cnt).ljust(3),\
            "best fit:", str(best_lo.fitness).ljust(4),"curr fit:", \
            str(curr_sol.fitness).ljust(10),posi)

def _ils_message_e(instance,run,non_improvement_cnt,best_lo,curr_sol,posi,dumb,steps,seed):

    if dumb != True:
        print(instance.name + "-" + str(instance.n).ljust(3), \
            "k:",str(instance.k).ljust(3),"seed:",str(seed).ljust(3), \
            "runs:",str(run).ljust(3), \
            "cnt:",str(non_improvement_cnt).ljust(3),\
            "best fit:", str(best_lo.fitness).ljust(4),"curr fit:", \
            str(curr_sol.fitness).ljust(10),
            "steps:",str(steps).ljust(3),posi.ljust(7),_parity_checker(curr_sol.lst,best_lo.lst))

def _ils_record(message,best_lo,curr_sol,run,non_improvement_cnt,climbing_cnt) -> list:

    info = []
    info.append(curr_sol.fitness)
    info.append(curr_sol.lst)
    info.append(run)
    info.append(non_improvement_cnt)
    info.append(climbing_cnt)
    info.append(best_lo.fitness)
    info.append(best_lo.lst)
    info.append(message)
    return info

def ILSearcher(
    logger_dir,
    instance,
    local_search = hill_climbing,
    neighbour_explorer = neighbour_explorer_with_eva,
    nb_runs: Optional[int] = 100, 
    non_impr_iters: Optional[int] = 100, 
    seed: Optional[int] = 42, 
    dumb: Optional[bool] = False) -> pd.DataFrame:
    """Performing ILS on a specified problem instance, record the search data, 
    including all the solutions encountered during the exploration (not only local optima)

    Parameters
    ----------
    instance : problem instance to study

    logger_dir : path
        Path for the directory to save the logger data

    local_search : local search method, default = hill_climbing

    neighbour_explorer : neighbourhood explorer method, default = neighbour_explorer_with_eva

    nb_runs : int, default = 100
        Number of independent runs to perform during ILS. 
        Each run will be independent of each other.
        This will directly control the sampling strength of ILS

    non_impr_iters : int, default = 100
        Maximum number of non-improving perturbations allowed to perform before terminating an 
        ILS run. This along with the perturbation strength (not implemented in this version) would 
        have impact on the searcher's ability in finding the global optimum or local optima with 
        high quality

    seed : int, default 42
        Note that this seed is used only for ILS. There is another seed parameter that is related
        to the reproducibility of the experiment, which is the seed used for instance generation

    dumb : bool, default False
        Whether to mute real time search information. For .py file, it is recommended to show such 
        messages to monitor the search progress. However, for .ipynb (Jupyter Notebook) usages, 
        dumb = True is recommended since this massive amounts of printings could cause the kernel to crash

    Return
    ------
    logger: pd.DataFrame
        A pandas dataframe storing the search data,
    """

    # random seed
    # random.seed(seed)

    # file path 
    file_dir = "ils_run" + str(nb_runs) + "_nimpr" + str(non_impr_iters)
    path = logger_dir + "/" + file_dir
    if not os.path.exists(path):
        os.makedirs(path)

    logger = []

    for run in range(1,nb_runs+1):
        
        # initialize sol
        sol = Solution(problem_name=instance.name)
        sol.init_rnd_bitstring(instance.n)
        random.shuffle(sol.lst)
        instance.full_eval(sol)   

        # obtain the first local optimum
        lo = local_search(instance, sol, neighbour_explorer)
        best_lo = copy.deepcopy(lo)

        # initialize countings
        non_improvement_cnt = 0
        climbing_cnt = 0
        
        # 1st: record the first lo
        logger.append(_ils_record("P1_Intialize",best_lo,lo,run,non_improvement_cnt,climbing_cnt))
        _ils_message(instance,run,non_improvement_cnt,best_lo,lo,"Initialized lo",dumb)

        while (non_improvement_cnt < non_impr_iters):

            # perform perturbation
            s = copy.deepcopy(best_lo)
            instance.two_rnd_flips(s)

            climbing_cnt = 0

            # 2nd: Record the just-flipped sol
            logger.append(_ils_record("P2_Flip",best_lo,s,run,non_improvement_cnt,climbing_cnt))
            _ils_message(instance,run,non_improvement_cnt,best_lo,s,"Flipped!!",dumb)

            # perform local search on the just-flipped solution for one time
            lo, improved = neighbour_explorer(instance, s, distance=1)
            climbing_cnt += 1
            
            # 3rd: record the first-mutated sol
            logger.append(_ils_record("P3_FirstMutate",best_lo,lo,run,non_improvement_cnt,climbing_cnt))
            _ils_message(instance,run,non_improvement_cnt,best_lo,lo,"1stMutated",dumb)

            # perform local search until new lo is found
            while improved:

                lo, improved = neighbour_explorer(instance, lo, distance=1)

                if improved:
                    
                    climbing_cnt += 1

                    # 4th: record further mutated sol until new lo is found
                    logger.append(_ils_record("P4_Mutate",best_lo,lo,run,non_improvement_cnt,climbing_cnt))
                    _ils_message(instance,run,non_improvement_cnt,best_lo,lo,"Mutated",dumb)

            if instance.better_or_equal(lo.fitness, best_lo.fitness):
                if instance.strictly_better(lo.fitness, best_lo.fitness):
                    non_improvement_cnt = 0
                else:
                    non_improvement_cnt += 1
                best_lo = copy.deepcopy(lo)
            else:
                non_improvement_cnt += 1

    logger = pd.DataFrame(data=logger,columns=["fit","sol","run","cnt","climbing","best_fit","best_sol","position"])
    logger.to_csv(path + "/" + "logger_" + instance.name + str(instance.n) + "_k_" + str(instance.k) + "_seed_" + str(instance.seed) + ".csv")



def ILSearcher_E(
    logger_dir,
    instance,
    local_search = hill_climbing,
    neighbour_explorer = neighbour_explorer_with_eva,
    nb_runs: Optional[int] = 100, 
    non_impr_iters: Optional[int] = 100, 
    seed: Optional[int] = 42, 
    dumb: Optional[bool] = False,
    distance=2) -> pd.DataFrame:
    """Performing SIMPLIFIED ILS on a specified problem instance, record the search data, 
    including all the solutions encountered during the exploration (not only local optima)

    Parameters
    ----------
    instance : problem instance to study

    logger_dir : path
        Path for the directory to save the logger data

    local_search : local search method, default = hill_climbing

    neighbour_explorer : neighbourhood explorer method, default = neighbour_explorer_with_eva

    nb_runs : int, default = 100
        Number of independent runs to perform during ILS. 
        Each run will be independent of each other.
        This will directly control the sampling strength of ILS

    non_impr_iters : int, default = 100
        Maximum number of non-improving perturbations allowed to perform before terminating an 
        ILS run. This along with the perturbation strength (not implemented in this version) would 
        have impact on the searcher's ability in finding the global optimum or local optima with 
        high quality

    seed : int, default 42
        Note that this seed is used only for ILS. There is another seed parameter that is related
        to the reproducibility of the experiment, which is the seed used for instance generation

    dumb : bool, default False
        Whether to mute real time search information. For .py file, it is recommended to show such 
        messages to monitor the search progress. However, for .ipynb (Jupyter Notebook) usages, 
        dumb = True is recommended since this massive amounts of printings could cause the kernel to crash

    Return
    ------
    logger: pd.DataFrame
        A pandas dataframe storing the search data,
    """

    # random seed
    random.seed(seed)

    logger = []

    for run in range(1,nb_runs+1):
        
        if dumb == True:
            progress(run,nb_runs+1)
        # initialize sol
        sol = Solution(problem_name=instance.name)
        sol.init_rnd_bitstring(instance.n)
        instance.full_eval(sol)   

        # obtain the first local optimum
        lo, steps = local_search(instance, sol, neighbour_explorer, mute=True)
        best_lo = copy.deepcopy(lo)

        # initialize countings
        non_improvement_cnt = 0
        
        # 1st: record the first lo
        logger.append(_ils_record("Initialized",best_lo,lo,run,non_improvement_cnt,steps))
        if dumb != True:
            _ils_message_e(instance,run,non_improvement_cnt,best_lo,lo,"Initialized",dumb,steps,seed)

        while (non_improvement_cnt < non_impr_iters):

            # perform perturbation
            s = copy.deepcopy(best_lo)
            instance.two_rnd_flips(s,distance=distance)

            lo, steps = local_search(instance,s,neighbour_explorer,mute=True)

            if instance.better_or_equal(lo.fitness, best_lo.fitness):
                if instance.strictly_better(lo.fitness, best_lo.fitness):
                    logger.append(_ils_record("Improving",best_lo,lo,run,non_improvement_cnt,steps))
                    if dumb != True:
                        _ils_message_e(instance,run,non_improvement_cnt,best_lo,lo,"Improving",dumb,steps,seed)
                    non_improvement_cnt = 0
                    best_lo = copy.deepcopy(lo)
                else:
                    logger.append(_ils_record("Equal",best_lo,lo,run,non_improvement_cnt,steps))
                    if dumb != True:
                        _ils_message_e(instance,run,non_improvement_cnt,best_lo,lo,"Equal",dumb,steps,seed)
                    non_improvement_cnt += 1
            else:
                logger.append(_ils_record("F",best_lo,lo,run,non_improvement_cnt,steps))
                if dumb != True:
                    _ils_message_e(instance,run,non_improvement_cnt,best_lo,lo,"F",dumb,steps,seed)
                non_improvement_cnt += 1

    # file path 
    file_dir = "data/" + instance.name + "/" + "seed" + str(seed) + "/" + "ils_run" + str(nb_runs) + "_nimpr" + str(non_impr_iters)
    path = logger_dir + "/" + file_dir
    if not os.path.exists(path):
        os.makedirs(path)

    logger = pd.DataFrame(data=logger,columns=["fit","sol","run","cnt","climbing","best_fit","best_sol","position"])
    logger.to_csv(path + "/" + "logger_" + instance.name + str(instance.n) + "_k_" + str(instance.k) + "_seed_" + str(instance.seed) + ".csv")


def valid_sol_generator(instance):
    sol = Solution(instance.name)
    sol.init_rnd_bitstring(instance.n)
    instance.full_eval(sol)
    if sol.invalid == True:
        while sol.invalid == True:
            sol.init_rnd_bitstring(instance.n)
            instance.full_eval(sol)
    return sol


def _ILS(run,instance,local_search,neighbour_explorer,non_impr_iters,distance,seed):

    # initialize sol
    sol = valid_sol_generator(instance)  

    # obtain the first local optimum
    lo, steps = local_search(instance, sol, neighbour_explorer, mute=True)
    best_lo = copy.deepcopy(lo)

    # initialize countings
    non_improvement_cnt = 0
    
    logger = []
    # 1st: record the first lo
    logger.append(_ils_record("Initialized",best_lo,lo,run,non_improvement_cnt,steps))

    while (non_improvement_cnt < non_impr_iters):

        # perform perturbation
        s = copy.deepcopy(best_lo)
        instance.two_rnd_flips(s,distance=distance)

        lo, steps = local_search(instance,s,neighbour_explorer,mute=True)

        if instance.better_or_equal(lo.fitness, best_lo.fitness):
            if instance.strictly_better(lo.fitness, best_lo.fitness):
                logger.append(_ils_record("Improving",best_lo,lo,run,non_improvement_cnt,steps))
                non_improvement_cnt = 0
                best_lo = copy.deepcopy(lo)
            else:
                logger.append(_ils_record("Equal",best_lo,lo,run,non_improvement_cnt,steps))
                non_improvement_cnt += 1
        else:
            logger.append(_ils_record("F",best_lo,lo,run,non_improvement_cnt,steps))
            non_improvement_cnt += 1
    return(logger)

def ILSearcher_MT(
    logger_dir,
    instance,
    local_search = hill_climbing,
    neighbour_explorer = two_opt_neighbour_explorer_ils,
    nb_runs: Optional[int] = 100, 
    non_impr_iters: Optional[int] = 100, 
    seed: Optional[int] = 42, 
    dumb: Optional[bool] = False,
    distance=2,
    n_jobs=60) -> pd.DataFrame:
    """Performing SIMPLIFIED ILS on a specified problem instance, record the search data, 
    including all the solutions encountered during the exploration (not only local optima)

    Parameters
    ----------
    instance : problem instance to study

    logger_dir : path
        Path for the directory to save the logger data

    local_search : local search method, default = hill_climbing

    neighbour_explorer : neighbourhood explorer method, default = neighbour_explorer_with_eva

    nb_runs : int, default = 100
        Number of independent runs to perform during ILS. 
        Each run will be independent of each other.
        This will directly control the sampling strength of ILS

    non_impr_iters : int, default = 100
        Maximum number of non-improving perturbations allowed to perform before terminating an 
        ILS run. This along with the perturbation strength (not implemented in this version) would 
        have impact on the searcher's ability in finding the global optimum or local optima with 
        high quality

    seed : int, default 42
        Note that this seed is used only for ILS. There is another seed parameter that is related
        to the reproducibility of the experiment, which is the seed used for instance generation

    dumb : bool, default False
        Whether to mute real time search information. For .py file, it is recommended to show such 
        messages to monitor the search progress. However, for .ipynb (Jupyter Notebook) usages, 
        dumb = True is recommended since this massive amounts of printings could cause the kernel to crash

    Return
    ------
    logger: pd.DataFrame
        A pandas dataframe storing the search data,
    """
    colorama.init()
    custom_style = "{desc}: {percentage:.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"

    # random seed
    random.seed(seed)
    # colorama.Fore.GREEN
    with tqdm_joblib(tqdm(desc = instance.name + "-n" + str(instance.n) + "-k" + str(instance.k) + "-seed" + str(instance.seed), 
                          total=nb_runs,
                          colour="red",
                          bar_format=custom_style,
                          ncols=80,)) as progress_bar:
        logger = Parallel(n_jobs=n_jobs)(delayed(_ILS)(run,
                                                        instance,
                                                        local_search = local_search,
                                                        neighbour_explorer = neighbour_explorer,
                                                        non_impr_iters = non_impr_iters,
                                                        distance = distance,
                                                        seed = instance.seed
                                                        ) for run in range(nb_runs))
    logger = [num for sublist in logger for num in sublist]

    # file path 
    file_dir = "data/" + instance.name + "/" + "seed" + str(instance.seed) + "/" + "ils_run" + str(nb_runs) + "_nimpr" + str(non_impr_iters)
    path = logger_dir + "/" + file_dir
    if not os.path.exists(path):
        os.makedirs(path)

    logger = pd.DataFrame(data=logger,columns=["fit","sol","run","cnt","climbing","best_fit","best_sol","position"])
    logger.to_csv(path + "/" + "logger_" + instance.name + str(instance.n) + "_k_" + str(instance.k) + "_seed_" + str(instance.seed) + ".csv")
    # logger.to_csv(logger_dir + "att48.csv")
