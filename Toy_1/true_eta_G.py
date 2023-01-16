from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import pickle
from Policy import target_policy, control_policy
import Simulator as Simulator
simulator = Simulator.Simulator(dim_state=1, model_type='toy')

def mc_eta_target(num_trajectory, num_time, seed, policy = "control"):
    trajectory_target = simulator.sample_target_policy_trajectory(
            num_trajectory, num_time, seed, policy, return_trajectory=True)
    est_eta_target = np.mean(np.vstack([_trajectory[3] for _trajectory in trajectory_target]),0)
    est_ratio_target = np.mean(np.vstack([_trajectory[0].flatten() for _trajectory in trajectory_target]),0)
    out = [num_trajectory, num_time, est_eta_target,est_ratio_target]
    return out


def mc_eta_target_star(args):
    return mc_eta_target(*args)

def estimate_eta(mc_s0_time, num_trajectory, num_time):
    seed_list = np.arange(mc_s0_time, dtype='int64').tolist()
    num_trajectory_list = (np.ones(mc_s0_time, dtype='int64') * num_trajectory).tolist()
    max_time_list = (np.ones(mc_s0_time, dtype='int64') * num_time).tolist()
    policy_list = ["G"]*mc_s0_time
    param_list = zip(num_trajectory_list, max_time_list, seed_list, policy_list)

    with Pool() as p:
        ETA = list(tqdm(p.imap(mc_eta_target_star, param_list), total=mc_s0_time))
    return ETA


ETA_list = dict()
for num_trajectory in [500]:
    ETA_list[num_trajectory] = dict()
    for num_time in [2000]:
        ETA_list[num_trajectory][num_time] = estimate_eta(200, num_trajectory, num_time)
    
        with open("true_eta_G.txt","wb") as fp:
            pickle.dump(ETA_list, fp)
        
        
#ETA_list = dict()
#for num_trajectory in [500]:
#    num_time = 5000
#    ETA_list[num_trajectory] = estimate_eta(200, num_trajectory, num_time)
    
#    with open("true_eta_a0_simple.txt","wb") as fp:
#        pickle.dump(ETA_list, fp)

