import numpy as np
#from problearner import PMLearner, PALearner
#from qlearner import Qlearner
#from rll import RatioLinearLearner
#from rnnl import RatioRKHSLearner, train
#from sklearn.model_selection import KFold
#import tensorflow as tf
from time import process_time

class evaluator:
    def __init__(self, dataset,N, T,
                 QLearner, PMLearner, 
                 RewardLearner, PALearner,
                 problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,20)},
                 truncate = 20,
                 t_depend_target = False, target_policy=None, control_policy = None, 
                 dim_state = 1, dim_mediator = 1, 
                 expectation_MCMC_iter = 100,
                 seed = 0):
        '''
        
        Parameters
        ----------
        dataset : A Dict
            A list with 6 elements. 
            They are: state, action, mediator, 
            reward, next state, action under policy to be evaluted.
        policy : TYPE
            DESCRIPTION.
        QLearner : TYPE
            A Q-learning model.
        RationLearner : TYPE
            A deep learning model for learning policy ratio.
        PMLearner : TYPE
            DESCRIPTION.
        PALearner : TYPE
            DESCRIPTION.
        gamma : TYPE
            DESCRIPTION.
        Returns
        -------
        None.
        '''
        np.random.seed(seed)
        self.seed = seed
        self.state = np.copy(dataset['state'])
        self.action = np.copy(dataset['action'])
        self.mediator = np.copy(dataset['mediator'])
        self.reward = np.copy(dataset['reward'])
        self.next_state = np.copy(dataset['next_state'])
        self.time_idx = np.copy(dataset['time_idx'])
        self.s0 = np.copy(dataset['s0'])
        self.N = N
        self.T = T
        
        self.dataset = dataset
        self.dim_state = dim_state
        self.dim_mediator = dim_mediator
        self.truncate = truncate
        
        self.target_policy = target_policy
        self.t_depend_target = t_depend_target
        #control_policy
        self.control_policy = control_policy
        self.a0 = control_policy(get_a = True)
        
        self.expectation_MCMC_iter = expectation_MCMC_iter
        
        t0 = process_time()
        self.pmlearner = PMLearner(dataset, problearner_parameters, seed, dim_state = dim_state,
                                   dim_mediator = dim_mediator)
        self.pmlearner.train()

        self.rewardlearner = RewardLearner(dataset, problearner_parameters, seed, dim_state = dim_state,
                                           dim_mediator = dim_mediator)
        self.rewardlearner.train()

        self.palearner = PALearner(dataset, problearner_parameters, seed, test = False, dim_state = dim_state,
                                   dim_mediator = dim_mediator)
        self.palearner.train()

        if self.t_depend_target:
            self.pie_A = self.target_policy(self.state, self.dim_state, action = self.action, time_idx = self.time_idx)
        else:
            self.pie_A = self.target_policy(self.state, self.dim_state, action = self.action)
        self.I_A = self.control_policy(self.state, self.dim_state, self.action)
        self.pieb_A = self.palearner.get_pa_prediction(self.state, self.action)
            
        self.unique_action = np.unique(self.action)
        self.unique_mediator = np.unique(self.mediator)
        self.est_DEMESE = None

        pass
    
    def estimate_DE_ME_SE(self):
        data_num = self.N * self.T
        self.ind_est = np.array([range(data_num)] * 6, dtype=float)

        self.rho_SAM = self._rho_SAM(data_num, self.state, self.action, self.mediator)
        self._Er_Sam(data_num, self.state, self.action, self.mediator)
        self._Er_Sa0m(data_num, self.state, self.action, self.mediator)
        self._Er_Sam_G(data_num, self.state, self.action, self.mediator)
        self._Er_SA(data_num, self.state, self.action, self.mediator)
        self._Er_SA_G(data_num, self.state, self.action, self.mediator)
        
        # Baseline Direct
        self.ind_est[0] = self.Er_Sam - self.Er_Sam_G
        self.ind_est[1] = self.Er_Sam_G - self.Er_Sa0m
        
        # Baseline IPW
        WISR1 = self.pie_A / self.pieb_A * np.copy(self.reward).flatten()
        WISR2 = self.I_A / self.pieb_A * np.clip(self.rho_SAM, a_min=None, a_max=self.truncate) * np.copy(self.reward).flatten()
        WISR3 = self.I_A / self.pieb_A * np.copy(self.reward).flatten()
        self.ind_est[2] = WISR1 - WISR2
        self.ind_est[3] = WISR2 - WISR3
        
        # Baseline TR
        Er_SAM = self.rewardlearner.get_reward_prediction(self.state, self.action, self.mediator)
        Er_Sa0M = self.rewardlearner.get_reward_prediction(self.state, self.a0, self.mediator)
        IS1 = self.pie_A / self.pieb_A * (np.copy(self.reward).flatten() - self.Er_SA.flatten())
        IS2 = self.I_A / self.pieb_A * np.clip(self.rho_SAM, a_min=None, a_max=self.truncate) * (np.copy(self.reward).flatten() - Er_SAM.flatten())
        IS3 = self.pie_A / self.pieb_A * (Er_Sa0M - self.Er_SA_G)
        IS4 = self.I_A / self.pieb_A * (np.copy(self.reward).flatten() - self.Er_SA.flatten())
        self.ind_est[4] = IS1 + self.Er_Sam - (IS2 + IS3 + self.Er_Sam_G)
        self.ind_est[5] = IS2 + IS3 + self.Er_Sam_G - (IS4 + self.Er_Sa0m)
        
        self.est_DEMESE = np.mean(self.ind_est,1)
        if data_num > 100:
            self.se_DEMESE = np.array([np.mean(self.ind_est[:,i*self.T:(i+1)*self.T],1) for i in range(self.N)]).std(0)/np.sqrt(self.N)
        else:
            self.se_DEMESE = np.std(self.ind_est,1)/np.sqrt(data_num)
        pass

    
    def _rho_SAM(self, data_num, state, action, mediator):
        pM_S = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pM_Sa = self.pmlearner.get_pm_prediction(state, np.array([a]), mediator)
            if self.t_depend_target:
                pie_a = self.target_policy(state, self.dim_state, a, time_idx)
            else:
                pie_a = self.target_policy(state, self.dim_state, a)
            pM_S += pie_a * pM_Sa
            
        pM_SA = self.pmlearner.get_pm_prediction(state, action, mediator)
        return pM_S / pM_SA
    
    
    def _Er_Sam(self, data_num, state, action, mediator):
        self.Er_Sam = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pie_a = self.target_policy(state, self.dim_state, action = a).flatten()
            Er_Sa = np.zeros(data_num, dtype=float)
            for rep in range(self.expectation_MCMC_iter):
                m_i_a = self.pmlearner.sample_m(state,  np.array([a]), random = True)
                reward_i_a = self.rewardlearner.get_reward_prediction(state,  np.array([a]), m_i_a)
                Er_Sa = self.update_exp(rep, Er_Sa, reward_i_a.reshape((-1,)))
            
            self.Er_Sam += pie_a * Er_Sa  
    
    def _Er_Sa0m(self, data_num, state, action, mediator):
        self.Er_Sa0m = np.zeros(data_num, dtype=float)
        for rep in range(self.expectation_MCMC_iter):
            m_i_a0 = self.pmlearner.sample_m(state, self.a0, random = True)
            reward_i_a0 = self.rewardlearner.get_reward_prediction(state, self.a0, m_i_a0)
            
            self.Er_Sa0m = self.update_exp(rep, self.Er_Sa0m, reward_i_a0.reshape((-1,)))
            
    
    def _Er_Sam_G(self, data_num, state, action, mediator):
        self.Er_Sam_G = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pie_a = self.target_policy(state, self.dim_state, action = a).flatten()
            Er_Sa0 = np.zeros(data_num, dtype=float)
            for rep in range(self.expectation_MCMC_iter):
                m_i_a = self.pmlearner.sample_m(state,  np.array([a]), random = True)
                reward_i_a0 = self.rewardlearner.get_reward_prediction(state, self.a0, m_i_a)
                Er_Sa0 = self.update_exp(rep, Er_Sa0, reward_i_a0.reshape((-1,)))
            
            self.Er_Sam_G += pie_a * Er_Sa0
            
    def _Er_SA(self, data_num, state, action, mediator):
        self.Er_SA = np.zeros(data_num, dtype=float)
        for rep in range(self.expectation_MCMC_iter):
            m_i_A = self.pmlearner.sample_m(state, action, random = True)
            reward_i_A = self.rewardlearner.get_reward_prediction(state, action, m_i_A)
            self.Er_SA = self.update_exp(rep, self.Er_SA, reward_i_A.reshape((-1,)))

    def _Er_SA_G(self, data_num, state, action, mediator):
        self.Er_SA_G = np.zeros(data_num, dtype=float)
        for rep in range(self.expectation_MCMC_iter):
            m_i_A = self.pmlearner.sample_m(state, action, random = True)
            reward_i_A = self.rewardlearner.get_reward_prediction(state, self.a0, m_i_A)
            self.Er_SA_G = self.update_exp(rep, self.Er_SA_G, reward_i_A.reshape((-1,)))
            
    def update_exp(self, rep, old_est, new_obs):
        return (rep*old_est + new_obs)/(rep+1)
        
        
