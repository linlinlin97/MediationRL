import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from scipy.special import expit
from evaluator_Baselines import evaluator
from Simulator import Simulator
from Policy import target_policy, control_policy
from probLearner import *
from probLearner import PMLearner, RewardLearner, PALearner
from ratioLearner import  RatioLinearLearner as RatioLearner
from qLearner_Linear_binary import Qlearner


import logging
logging.basicConfig(level=logging.INFO,
                    format='%(message)s', filename='triple_robust.log')

simulator = Simulator(model_type='toy_iid', dim_state=1)

# truth
true_IDE, true_IME, true_SE = -1.1354137700000002, -1.1669420600000002, 0.0

def toy_robust(num_trajectory, num_time, seed):
    simulator.sample_trajectory(num_trajectory, num_time, seed)
    simulator.trajectory2iid()
    sim_iid_dataset = simulator.iid_dataset
    
    est_obj0 = evaluator(sim_iid_dataset, num_trajectory, num_time,
                         Qlearner, PMLearner, 
                         RewardLearner, PALearner,
                         problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,20)},
                         toy_binary = True,
                         truncate = 500,
                         target_policy=target_policy, control_policy = control_policy, dim_state = 1, dim_mediator = 1,
                         seed = seed)
    
    est_obj0.estimate_DE_ME_SE()
    est_value1 = est_obj0.est_DEMESE
    IDE_baseD = est_value1[0] - true_IDE
    IME_baseD = est_value1[1] - true_IME
    
    IDE_baseIPW = est_value1[2] - true_IDE
    IME_baseIPW = est_value1[3] - true_IME

    IDE_baseTR = est_value1[4] - true_IDE
    IME_baseTR = est_value1[5] - true_IME
    return [IDE_baseD, IME_baseD,
            IDE_baseIPW, IME_baseIPW,
            IDE_baseTR, IME_baseTR,
            num_trajectory, num_time, seed]


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
Error[num_time] = dict()
for num_trajectory in np.arange(1, 11) * 30:
    Error[num_time][num_trajectory] = MC_toy_robust(nrep, num_trajectory, num_time)
    
    with open("Toy_2_iid_baselines.txt","wb") as fp:
        pickle.dump(Error, fp)
