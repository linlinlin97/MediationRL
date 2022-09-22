from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import pickle
from scipy.special import expit
from Policy import *
import Simulator as Simulator
model_type = 'Gaussian_toy'
simulator = Simulator.Simulator(dim_state=1, model_type=model_type)

expectation_MCMC_iter = 100


def mc_DE_ME_SE(num_trajectory=500, num_time=500, seed=1):

    target_trajectories = simulator.sample_target_policy_trajectory(
                num_trajectory, num_time, seed, policy = "target", return_trajectory=True)
    target_iid_dataset = simulator.trajectory2iid(trajectory = target_trajectories)
    
    control_trajectories = simulator.sample_target_policy_trajectory(
                num_trajectory, num_time, seed, policy = "control", return_trajectory=True)
    control_iid_dataset = simulator.trajectory2iid(trajectory = control_trajectories)
    
    a0 = control_policy(get_a = True)
    unique_action = np.unique(target_iid_dataset['action'])
    
    target_state = np.copy(target_iid_dataset['state'])
    control_state = np.copy(control_iid_dataset['state'])

    DE = est_DE(target_state, a0, unique_action, target_policy)
    ME = est_ME(target_state, a0, unique_action, target_policy)
    SE = est_SE(target_state, control_state, a0)
    
    return np.array([DE,ME,SE])

def est_DE(state, a0, unique_action, target_policy):
    DE = np.zeros(state.shape[0])
    for a in unique_action:
        pie_a = np.apply_along_axis(target_policy, 1, state, action=a).flatten()
        #Em_Sa = simulator.sa2mediator_model(state, a, random = False)
        sampled_reward_a = []
        sampled_reward_a0 = []
        for rep in range(expectation_MCMC_iter):
            np.random.seed(rep)
            m_i_a = simulator.sa2mediator_model(state, a, random = True)
            reward_i_a = simulator.sam2reward_model(state, action=a, mediator=m_i_a, random = False, matrix_based = True).flatten()
            reward_i_a0 = simulator.sam2reward_model(state, action=a0, mediator=m_i_a, random = False, matrix_based = True).flatten()
            sampled_reward_a.append(reward_i_a)
            sampled_reward_a0.append(reward_i_a0)
        Er_Sa = np.mean(sampled_reward_a,0)
        Er_Sa0 = np.mean(sampled_reward_a0,0)
        #Er_Sam = simulator.sam2reward_model(state, action=a, mediator=Em_Sa, 
        #                                  random = False, matrix_based = True).flatten()
        #Er_Sa0m = simulator.sam2reward_model(state, action=a0, mediator=Em_Sa, 
        #                                  random = False, matrix_based = True).flatten()
        #DE += pie_a * (Er_Sam - Er_Sa0m)
        DE += pie_a * (Er_Sa - Er_Sa0)
      
    return np.mean(DE)


def est_ME(state, a0, unique_action, target_policy):
    ME = np.zeros(state.shape[0])
    for a in unique_action:
        pie_a = np.apply_along_axis(target_policy, 1, state, action=a).flatten()
        sampled_reward_a = []
        sampled_reward_a0 = []
        for rep in range(expectation_MCMC_iter):
            np.random.seed(rep)
            m_i_a = simulator.sa2mediator_model(state, a, random = True)
            m_i_a0 = simulator.sa2mediator_model(state, a0, random = True)
            reward_i_a = simulator.sam2reward_model(state, action=a0, mediator=m_i_a, random = False, matrix_based = True).flatten()
            reward_i_a0 = simulator.sam2reward_model(state, action=a0, mediator=m_i_a0, random = False, matrix_based = True).flatten()
            sampled_reward_a.append(reward_i_a)
            sampled_reward_a0.append(reward_i_a0)
        Er_Sa = np.mean(sampled_reward_a,0)
        Er_Sa0 = np.mean(sampled_reward_a0,0)
        
        #Em_Sa = simulator.sa2mediator_model(state, a, random = False) 
        #Em_Sa0 = simulator.sa2mediator_model(state, a0, random = False) 
        #Er_Sa0m_mSa = simulator.sam2reward_model(state, action=a0, mediator=Em_Sa, 
        #                                  random = False, matrix_based = True).flatten()
        #Er_Sa0m_mSa0 = simulator.sam2reward_model(state, action=a0, mediator=Em_Sa0, 
        #                                  random = False, matrix_based = True).flatten()
        #ME += pie_a * (Er_Sa0m_mSa - Er_Sa0m_mSa0)
        ME += pie_a * (Er_Sa- Er_Sa0)

    return np.mean(ME)

def est_SE(target_state, control_state, a0):
    SE = np.zeros(target_state.shape[0])
    
    #target
    sampled_reward_a0 = []
    for rep in range(expectation_MCMC_iter):
        np.random.seed(rep)
        m_i_a0 = simulator.sa2mediator_model(target_state, a0, random = True)
        reward_i_a0 = simulator.sam2reward_model(target_state, action=a0, mediator=m_i_a0, random = False, matrix_based = True).flatten()
        sampled_reward_a0.append(reward_i_a0)
    Er_Sa0_target = np.mean(sampled_reward_a0,0)
    
    #Em_Sa0_target = simulator.sa2mediator_model(target_state, a0, random = False) 
    #Er_Sa0m_mSa0_target = simulator.sam2reward_model(target_state, action=a0, mediator=Em_Sa0_target, 
    #                                      random = False, matrix_based = True).flatten()
    
    #control
    sampled_reward_a0 = []
    for rep in range(expectation_MCMC_iter):
        np.random.seed(rep)
        m_i_a0 = simulator.sa2mediator_model(control_state, a0, random = True)
        reward_i_a0 = simulator.sam2reward_model(control_state, action=a0, mediator=m_i_a0, random = False, matrix_based = True).flatten()
        sampled_reward_a0.append(reward_i_a0)
    Er_Sa0_control = np.mean(sampled_reward_a0,0)
    
    #Em_Sa0_control = simulator.sa2mediator_model(control_state, a0, random = False) 
    #Er_Sa0m_mSa0_control = simulator.sam2reward_model(control_state, action=a0, mediator=Em_Sa0_control, 
    #                                      random = False, matrix_based = True).flatten()
    #SE += Er_Sa0m_mSa0_target - Er_Sa0m_mSa0_control
    
    SE += Er_Sa0_target - Er_Sa0_control

    return np.mean(SE)



def mc_DE_ME_SE_star(args):
    return mc_DE_ME_SE(*args)

def estimate_DE_ME_SE(mc_s0_time, num_trajectory, num_time):
    seed_list = np.arange(mc_s0_time, dtype='int64').tolist()
    num_trajectory_list = (np.ones(mc_s0_time, dtype='int64') * num_trajectory).tolist()
    max_time_list = (np.ones(mc_s0_time, dtype='int64') * num_time).tolist()
    param_list = zip(num_trajectory_list, max_time_list, seed_list)

    with Pool() as p:
        DE_ME_SE = list(tqdm(p.imap(mc_DE_ME_SE_star, param_list), total=mc_s0_time))
    return DE_ME_SE



def main(save_file):
    DE_ME_SE = dict()
    for num_time in [500, 1000]: 
        DE_ME_SE[num_time] = dict()
        for num_trajectory in [100, 200]:
            DE_ME_SE[num_time][num_trajectory] = estimate_DE_ME_SE(200, num_trajectory, num_time)
            with open(save_file,"wb") as fp:
                pickle.dump(DE_ME_SE, fp)
    return DE_ME_SE

    
DE_ME_SE = main("true_DE_ME_SE_direct_est.txt")











