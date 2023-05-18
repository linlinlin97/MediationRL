import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from scipy.special import expit
from evaluator_Baselines import evaluator
from Simulator import Simulator
from Policy import target_policy, control_policy
from probLearner import PMLearner, RewardLearner, PALearner
from ratioLearner import  RatioLinearLearner as RatioLearner
from qLearner_Linear import Qlearner

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(message)s', filename='triple_robust.log')

dim_state=1; dim_mediator = 2
std_M = 2; std_S = 2
simulator = Simulator(model_type='Gaussian_semi', dim_state=dim_state, dim_mediator = dim_mediator, std_M = std_M, std_S = std_S)

problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,40)}


# truth
true_DE,true_ME,true_SE = 0,0,0


def toy_robust(num_trajectory, num_time, expectation_MCMC_iter = 100, truncate = 20, seed = 0):
    simulator.sample_trajectory(num_trajectory, num_time, seed)
    simulator.trajectory2iid()
    sim_iid_dataset = simulator.iid_dataset

    est_obj1 = evaluator(sim_iid_dataset, num_trajectory, num_time,
                         Qlearner, PMLearner, RewardLearner, PALearner,
                         problearner_parameters = problearner_parameters, 
                         truncate = truncate,
                         target_policy=target_policy, control_policy = control_policy,
                         dim_state = 1, dim_mediator = 2,
                         expectation_MCMC_iter = expectation_MCMC_iter, 
                         seed = 0)
    
    est_obj1.estimate_DE_ME_SE()
    est_value1 = est_obj1.est_DEMESE
    se_value1 = est_obj1.se_DEMESE
    IDE_baseD = est_value1[0]
    IME_baseD = est_value1[1]
    
    IDE_baseIPW = est_value1[2]
    IME_baseIPW = est_value1[3]

    IDE_baseTR = est_value1[4]
    IME_baseTR = est_value1[5]
      
    return [IDE_baseD, IME_baseD,
            IDE_baseIPW, IME_baseIPW,
            IDE_baseTR, IME_baseTR,
            num_trajectory, num_time, seed]

def toy_robust_star(args):
    return toy_robust(*args)


def MC_toy_robust(nrep, num_trajectory, num_time, expectation_MCMC_iter, truncate):
    seed_list = np.arange(nrep, dtype='int64').tolist()
    num_trajectory_list = (np.ones(nrep, dtype='int64') * num_trajectory).tolist()
    max_time_list = (np.ones(nrep, dtype='int64') * num_time).tolist()
    MCMC_iter_list = (np.ones(nrep, dtype='int64') * expectation_MCMC_iter).tolist()
    truncatef_list = (np.ones(nrep, dtype='int64') * truncate).tolist()
    param_list = zip(num_trajectory_list, max_time_list,  MCMC_iter_list, truncatef_list, seed_list)

    with Pool() as p:
        Error = list(tqdm(p.imap(toy_robust_star, param_list), total=nrep))
    return Error

nrep = 100
Error = dict()
for num_time in np.arange(1, 2) * 25:
    Error[num_time] = dict()
    for num_trajectory in np.arange(1, 16) * 100:
        Error[num_time][num_trajectory]= MC_toy_robust(nrep,num_trajectory, num_time, 50, 500)
        with open("estimate_additive_identity_L7_stdM_"+str(std_M)+"_stdS_"+str(std_S)+'baselines.txt',"wb") as fp:
            pickle.dump(Error, fp)

