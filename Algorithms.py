import copy
import random 
import math
import pandas as pd
from GBFLAT.Problems import Solution
from GBFLAT.Problems import one_bit_flip, double_bridge
from GBFLAT.LocalOptimaNetwork import LON
from joblib import Parallel, delayed
from decimal import Decimal


def temp_updator(x,initial_temp):
    return initial_temp * (0.995 ** (x))

def sim_anneal(
        instance,
        neighbor_explorer,
        temp_updator,
        n_iter,
        initial_temp=40000):
    
    # initialization 
    sol = Solution(problem_name=instance.name)
    sol.init_rnd_bitstring(instance.n)
    instance.full_eval(sol)
    best_sol = copy.deepcopy(sol)
    # logger
    sol_list, fit_list, candi_list, candi_fit_list, move_list, prob_list, t_list = [],[],[],[],[],[],[]
    # main loop
    for i in range(n_iter):

        sol_list.append(sol.lst)
        fit_list.append(sol.fitness)
        # in each move, we random select a neighbour of current solution as the candidate for the next move
        neighbours = neighbor_explorer(sol.lst)
        candidate_sol = random.choice(neighbours)
        candidate_sol = Solution(problem_name=instance.name,lst=candidate_sol)
        instance.full_eval(candidate_sol)

        # calculate accept probability, which is dependent on 
        # the difference of the fitness of the two solutions, and the current temperature
        temp = temp_updator(i,initial_temp)
        t_list.append(temp)
        if instance.maximize():
            ac_prob = math.exp(-(candidate_sol.fitness - sol.fitness) / temp)
        else: 
            ac_prob = math.exp(-(candidate_sol.fitness - sol.fitness) / temp)
        
        prob_list.append(ac_prob)
        candi_list.append(candidate_sol.lst)
        candi_fit_list.append(candidate_sol.fitness)

        if instance.maximize():
            # if the fitness of this candidate is better than current solution, we accept it
            if candidate_sol.fitness > sol.fitness:
                sol = candidate_sol
                move_list.append("AC_Direc")
            # othewise, we accept it with a probability calculated above
            elif random.random() < ac_prob:
                sol = candidate_sol
                move_list.append("AC_Prob")
            else:
                move_list.append("Rej")
        else:
            if candidate_sol.fitness < sol.fitness:
                sol = candidate_sol
                move_list.append("AC_Direc")
            elif random.random() < ac_prob:
                sol = candidate_sol
                move_list.append("AC_Prob")
            else:
                move_list.append("Rej")
        # if the current solution is better than the best_sol, replace the best_sol with sol
        if instance.maximize():
            if sol.fitness > best_sol.fitness:
                best_sol = sol
        else:
            if sol.fitness < best_sol.fitness:
                best_sol = sol

    logger = pd.DataFrame(
        data=list(zip(sol_list,fit_list,candi_list,candi_fit_list,move_list,prob_list,t_list)),
        columns=["curr_sol","fit","candi_sol","candi_fit","next-move","prob","temp"])
    
    logger.loc[logger["prob"] >= 1, "prob"] = 1
    
    return best_sol, logger


# def temp_exp_s(x,initial_temp):
    
#     new_temp = initial_temp * (0.97 ** (x / 20))
#     if new_temp < 1e-5:
#         return 1e-5
#     else:
#         return new_temp
    
def temp_exp_s(x,initial_temp):
    
    new_temp = initial_temp * (0.95 ** (x / 100))
    if new_temp < 1e-5:
        return 1e-5
    else:
        return new_temp

def temp_exp_f(x,initial_temp):
    
    new_temp = initial_temp * (0.98 ** x)
    if new_temp < 1e-5:
        return 1e-5
    else:
        return new_temp

def temp_log(x,initial_temp):
    
    return initial_temp / (x + 1)

def sampling(instance):

    sol = Solution(problem_name=instance.name)
    sol.init_rnd_bitstring(instance.n)
    instance.full_eval(sol)
    return sol.fitness

def sim_anneal_target(instance,
                      target,
                      initial_temp,
                      max_iter=100000,
                      cooling_method="auto",
                      neighbor_explorer="auto",
                      summary=False,
                      mute=True
                      ):
    
    # initialization 
    sol = Solution(problem_name=instance.name)
    sol.init_rnd_bitstring(instance.n)
    instance.full_eval(sol)
    best_sol = copy.deepcopy(sol)
    # initialize FE value
    func_eva = 0

    # initialize strategies
    # set neighbor exploration method
    if neighbor_explorer == "auto":
        if instance.name == "TSP":
            neighbor_explorer = double_bridge
        else:
            neighbor_explorer = one_bit_flip

    # set cooling method
    if cooling_method == "auto":
        cooling_method = temp_exp_s
    elif cooling_method == "exp_slow":
        cooling_method = temp_exp_s
    elif cooling_method == "exp_fast":
        cooling_method = temp_exp_f
    elif cooling_method == "log":
        cooling_method = temp_log

    # main loop

    fit_list, temp_list = [], []
    for i in range(max_iter):

        candidate_sol = neighbor_explorer(sol.lst)
        candidate_sol = Solution(problem_name=instance.name,lst=candidate_sol)
        instance.full_eval(candidate_sol)

        temp = cooling_method(i,initial_temp)

        if instance.maximize():
            try:
                ac_prob = math.exp(-(candidate_sol.fitness - sol.fitness) / temp)
            except:
                ac_prob = 0
        else: 
            try:
                ac_prob = math.exp(-(candidate_sol.fitness - sol.fitness) / temp)
            except:
                ac_prob = 0

        if ac_prob > 1:
            ac_prob = 1
        
        if mute == False:
            # print("iter:",i,"temp:{:.2f}".format(temp),"fitness:",sol.fitness,"next_move:{:.5E}".format(Decimal(ac_prob)))
            print("iter:",i,"temp:{:.2f}".format(temp),"fitness:",str(sol.fitness).ljust(5),"next_move_fit:",str(candidate_sol.fitness).ljust(10),\
                  "prob:",ac_prob)
            
        if instance.maximize():
            # if the fitness of this candidate is better than current solution, we accept it
            if candidate_sol.fitness > sol.fitness:
                sol = candidate_sol
            elif random.random() < ac_prob:
                sol = candidate_sol
        else:
            if candidate_sol.fitness < sol.fitness:
                sol = candidate_sol
            elif random.random() < ac_prob:
                sol = candidate_sol
  
        # if the current solution is better than the best_sol, replace the best_sol with sol
        if instance.maximize():
            if sol.fitness > best_sol.fitness:
                best_sol = sol
        else:
            if sol.fitness < best_sol.fitness:
                best_sol = sol
        # if the current solution meets the target value, then terminate the algorithm
        if instance.maximize():
            if sol.fitness >= target:
                break
        else:
            if sol.fitness <= target:
                break

        func_eva += 1
        fit_list.append(sol.fitness)
        temp_list.append(temp)

    if summary == True:
        print("summary:\n","instance:",instance.name,"\n","target:",target,"\n","initial_temp:",initial_temp,"\n",
            "cooling_method:",cooling_method,"\n","neighbor_explorer:",neighbor_explorer,"\n","max_iter:",max_iter,
            "\n","best_sol:",best_sol.fitness,"\n","func_eva:",func_eva,"\n","final_temp:",temp)
        
    return best_sol, func_eva