import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from scipy.special import expit
from evaluator_Linear import evaluator
from Simulator import Simulator
from Policy import target_policy, control_policy
from probLearner import PMLearner, RewardLearner, PALearner
from ratioLearner import  RatioLinearLearner as RatioLearner
from qLearner_Linear import Qlearner

import logging
logging.basicConfig(level=logging.INFO,
                    format='%(message)s', filename='triple_robust.log')

simulator = Simulator(model_type='Gaussian_toy', dim_state=1)


# truth
true_DE = 1.829490
true_ME =  1.784424
true_SE = 2.343208



def toy_robust(num_trajectory, num_time, seed):
    simulator.sample_trajectory(num_trajectory, num_time, seed)
    simulator.trajectory2iid()
    sim_iid_dataset = simulator.iid_dataset

    est_obj1 = evaluator(sim_iid_dataset, Qlearner, PMLearner, RewardLearner, PALearner, RatioLearner,
                         target_policy=target_policy, control_policy = control_policy, dim_state = 1,
                         dim_mediator = 1, scaler = 'Identity', expectation_MCMC_iter = 100, seed = seed)
        
    est_obj1.estimate_DE_ME_SE()
    est_value1 = est_obj1.est_DEMESE
    var_value1 = est_obj1.var_DEMESE
    DE_error_TR = est_value1[0] - true_DE
    ME_error_TR = est_value1[1] - true_ME
    SE_error_TR = est_value1[2] - true_SE
    DE_error_naive = est_value1[3] - true_DE
    ME_error_naive = est_value1[4] - true_ME
    SE_error_naive = est_value1[5] - true_SE

    DE_var_TR = var_value1[0]
    ME_var_TR = var_value1[1]
    SE_var_TR = var_value1[2]
    DE_var_naive = var_value1[3]
    ME_var_naive = var_value1[4]
    SE_var_naive = var_value1[5]
      
    return [DE_error_TR, SE_error_TR, ME_error_TR, DE_error_naive, SE_error_naive, ME_error_naive,
            DE_var_TR, SE_var_TR, ME_var_TR, DE_var_naive, SE_var_naive, ME_var_naive,
            num_trajectory, num_time, seed, est_obj1.time_rec, est_obj1.ratio_target, est_obj1.ratio_control, est_obj1.eta_pie, est_obj1.eta_a0]#, pmlearner.std, rewardlearner.std]

def toy_robust_star(args):
    return toy_robust(*args)


def MC_toy_robust(nrep, num_trajectory, num_time):
    seed_list = np.arange(nrep, dtype='int64').tolist()
    num_trajectory_list = (np.ones(nrep, dtype='int64') * num_trajectory).tolist()
    max_time_list = (np.ones(nrep, dtype='int64') * num_time).tolist()
    param_list = zip(num_trajectory_list, max_time_list, seed_list)

    with Pool() as p:
        Error = list(tqdm(p.imap(toy_robust_star, param_list), total=nrep))
    return Error

num_time = 50
nrep = 200
Error = dict()
for num_time in np.arange(1, 15) * 30:
    Error[num_time] = dict()
    for num_trajectory in np.arange(1, 2) * 30:
        Error[num_time][num_trajectory]= MC_toy_robust(nrep,num_trajectory, num_time)
        with open("Linear_Error_test_estimatedratio_Test.txt","wb") as fp:
            pickle.dump(Error, fp)

for num_time in np.arange(1, 2) * 30:
    for num_trajectory in np.arange(1, 15) * 30:
        Error[num_time][num_trajectory]= MC_toy_robust(nrep,num_trajectory, num_time)
        with open("Linear_Error_test_estimatedratio_Test.txt","wb") as fp:
            pickle.dump(Error, fp)