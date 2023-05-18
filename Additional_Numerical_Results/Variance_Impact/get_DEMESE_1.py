from tqdm import tqdm
import numpy as np
from multiprocessing import Pool
import pickle
from scipy.special import expit
from Policy import *
from Simulator import Simulator
dim_state=1; dim_mediator = 2
std_M = 1; std_S = 1
simulator = Simulator(model_type='Gaussian_semi', dim_state=dim_state, dim_mediator = dim_mediator, std_M = std_M, std_S = std_S)

expectation_MCMC_iter = 1000


def mc_DE_ME_SE(num_trajectory=500, num_time=500, seed=1):

    target_trajectories = simulator.sample_target_policy_trajectory(
                num_trajectory, num_time, seed, policy = "target", return_trajectory=True)
    target_iid_dataset = simulator.trajectory2iid(trajectory = target_trajectories)
    
    control_trajectories = simulator.sample_target_policy_trajectory(
                num_trajectory, num_time, seed, policy = "control", return_trajectory=True)
    control_iid_dataset = simulator.trajectory2iid(trajectory = control_trajectories)
    
    G_trajectories = simulator.sample_target_policy_trajectory(
                num_trajectory, num_time, seed, policy = "G", return_trajectory=True)
    G_iid_dataset = simulator.trajectory2iid(trajectory = G_trajectories)
    
    a0 = control_policy(get_a = True)
    unique_action = np.unique(target_iid_dataset['action'])
    
    target_state = np.copy(target_iid_dataset['state'])
    control_state = np.copy(control_iid_dataset['state'])
    G_state = np.copy(G_iid_dataset['state'])

    np.random.seed(seed)
    eta1 = est_eta1(target_state, a0, unique_action)
    eta2 = est_eta2(target_state, a0, unique_action)
    eta3 = est_eta3(target_state, a0, unique_action)
    eta4 = est_eta4(G_state, a0, unique_action)
    eta5 = est_eta5(control_state, a0, unique_action)
    
    return np.array([eta1, eta2, eta3, eta4, eta5])


def est_eta1(state, a0, unique_action):
    eta1 = np.zeros(state.shape[0])
    for a in unique_action:
        pie_a = target_policy(state, dim_state, a).flatten()
        sampled_reward_a = np.zeros(state.shape[0])
        for rep in range(expectation_MCMC_iter):
            m_i_a = simulator.sa2mediator_model(state, a, random = True)
            reward_i_a = simulator.sam2reward_model(state, action=a, mediator=m_i_a, random = False, matrix_based = True).flatten()
            sampled_reward_a = update_exp(rep, sampled_reward_a, reward_i_a)
        eta1 += pie_a * sampled_reward_a
    return np.mean(eta1)

def est_eta2(state, a0, unique_action):
    eta2 = np.zeros(state.shape[0])
    for a in unique_action:
        pie_a = target_policy(state, dim_state, a).flatten()
        sampled_reward_a0 = np.zeros(state.shape[0])
        for rep in range(expectation_MCMC_iter):
            m_i_a = simulator.sa2mediator_model(state, a, random = True)
            reward_i_a0 = simulator.sam2reward_model(state, action=a0, mediator=m_i_a, random = False, matrix_based = True).flatten()
            sampled_reward_a0 = update_exp(rep, sampled_reward_a0, reward_i_a0)
        eta2 += pie_a * sampled_reward_a0
    return np.mean(eta2)

def est_eta3(state, a0, unique_action):
    eta3 = np.zeros(state.shape[0])
    for rep in range(expectation_MCMC_iter):
        m_i_a0 = simulator.sa2mediator_model(state, a0, random = True)
        reward_i_a0 = simulator.sam2reward_model(state, action=a0, mediator=m_i_a0, random = False, matrix_based = True).flatten()
        eta3 = update_exp(rep, eta3, reward_i_a0)
    return np.mean(eta3)

def est_eta4(state, a0, unique_action):
    eta4 = np.zeros(state.shape[0])
    for rep in range(expectation_MCMC_iter):
        m_i_a0 = simulator.sa2mediator_model(state, a0, random = True)
        reward_i_a0 = simulator.sam2reward_model(state, action=a0, mediator=m_i_a0, random = False, matrix_based = True).flatten()
        eta4 = update_exp(rep, eta4, reward_i_a0)
    return np.mean(eta4)
          

def est_eta5(state, a0, unique_action):
    eta5 = np.zeros(state.shape[0])
    for rep in range(expectation_MCMC_iter):
        m_i_a0 = simulator.sa2mediator_model(state, a0, random = True)
        reward_i_a0 = simulator.sam2reward_model(state, action=a0, mediator=m_i_a0, random = False, matrix_based = True).flatten()
        eta5 = update_exp(rep, eta5, reward_i_a0)
    return np.mean(eta5)   
    

def update_exp(rep, old_est, new_obs):
        return (rep*old_est + new_obs)/(rep+1)


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
    for num_time in [3200,6400]: 
        DE_ME_SE[num_time] = dict()
        for num_trajectory in [100]:
            DE_ME_SE[num_time][num_trajectory] = np.mean(estimate_DE_ME_SE(200, num_trajectory, num_time),0)
            with open(save_file,"wb") as fp:
                pickle.dump(DE_ME_SE, fp)
    return DE_ME_SE

    
DE_ME_SE = main("true_iDE_IME_DDE_DME_direct_est_stdM_"+str(std_M)+"_stdS_"+str(std_S)+'.txt')
