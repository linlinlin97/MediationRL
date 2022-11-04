import numpy as np
#from problearner import PMLearner, PALearner
#from qlearner import Qlearner
#from rll import RatioLinearLearner
#from rnnl import RatioRKHSLearner, train
#from sklearn.model_selection import KFold
#import tensorflow as tf
from time import process_time

class evaluator:
    def __init__(self, dataset,
                 QLearner, PMLearner, 
                 RewardLearner, PALearner, RatioLearner,
                 problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,20)},
                 ratio_ndim = 5, truncate = 20, l2penalty = 1.0,
                 target_policy=None, control_policy = None, dim_state = 1, dim_mediator = 1, 
                 Q_settings = {'scaler': 'Identity','product_tensor': True, 'beta': 3/7, 'include_intercept': False,
                               'expectation_MCMC_iter_Q3': 100, 'expectation_MCMC_iter_Q_diff':100, 'penalty': 10**(-9),
                              'd': 3, 'min_L': 7},
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
        self.s0 = np.copy(dataset['s0'])
        
        self.dataset = dataset
        self.dim_state = dim_state
        self.dim_mediator = dim_mediator
        self.truncate = truncate
        
        self.target_policy = target_policy
        #control_policy
        self.control_policy = control_policy
        self.a0 = control_policy(get_a = True)
        
        self.expectation_MCMC_iter = expectation_MCMC_iter
        
        self.qlearner = QLearner
        t0 = process_time()
        self.pmlearner = PMLearner(dataset, problearner_parameters, seed, dim_state = dim_state,
                                   dim_mediator = dim_mediator)
        self.pmlearner.train()
        t_m = process_time() - t0
        t0 = process_time()
        self.rewardlearner = RewardLearner(dataset, problearner_parameters, seed, dim_state = dim_state,
                                           dim_mediator = dim_mediator)
        self.rewardlearner.train()
        t_r= process_time() - t0
        t0 = process_time()
        self.palearner = PALearner(dataset, problearner_parameters, seed, test = False, dim_state = dim_state,
                                   dim_mediator = dim_mediator)
        self.palearner.train()
        t_a = process_time() - t0
        t0 = process_time()
        self.ratiolearner = RatioLearner(dataset, target_policy, control_policy, self.palearner, ndim=ratio_ndim,
                                         truncate=truncate, dim_state = dim_state, l2penalty = l2penalty)
        self.ratiolearner.fit()
        self.w_pie = self.ratiolearner.get_r_prediction(self.state, policy = 'target', normalize=True)
        self.w_a0 = self.ratiolearner.get_r_prediction(self.state, policy = 'control', normalize=True)
        self.pie_A = self.target_policy(self.state, self.dim_state, self.action, matrix_based = True)
        self.I_A = self.control_policy(self.state, self.dim_state, self.action, matrix_based = True)
        self.pieb_A = self.palearner.get_pa_prediction(self.state, self.action)
        t_ratio = process_time() - t0
        t0 = process_time()
        self.unique_action = np.unique(self.action)
        self.unique_mediator = np.unique(self.mediator)
        self.est_DEMESE = None
        
        self.Q_settings = Q_settings
        self.time_rec = {'t_m':t_m,'t_r':t_r,'t_a':t_a,'t_ratio':t_ratio}
        
        pass
    
    def estimate_DE_ME_SE(self):
        data_num = self.state.shape[0]
        self.ind_est = np.array([range(data_num)] * 8, dtype=float)
        t0 = process_time()
        Q_est = self.qlearner(self.dataset, self.target_policy, self.control_policy, self.pmlearner, self.rewardlearner, self.ratiolearner, self.palearner, self.unique_action, self.dim_state, self.dim_mediator, self.Q_settings, self.seed)
        self.time_rec['t_Q0'] = process_time() - t0
        t0 = process_time()
        Q_est.est_Q1()
        self.time_rec['t_Q1'] = process_time() - t0
        t0 = process_time()
        Q_est.est_Q2()
        self.time_rec['t_Q2'] = process_time() - t0
        t0 = process_time()
        Q_est.est_Q3()
        self.time_rec['t_Q3'] = process_time() - t0
        t0 = process_time()
        Q_est.est_Q4()
        self.time_rec['t_Q4'] = process_time() - t0
        t0 = process_time()
        Q_est.est_Qdiffs()
        self.time_rec['t_Qdiff'] = process_time() - t0
        self.Q1_diff, self.eta_pie = Q_est.Q1_diff, Q_est.eta_pie
        self.Q2_diff, self.eta_piea0 = Q_est.Q2_diff, Q_est.eta_piea0
        self.Q3_diff, self.eta_piea0star = Q_est.Q3_diff, Q_est.eta_piea0star
        self.Q4_diff, self.eta_a0 = Q_est.Q4_diff, Q_est.eta_a0
        self.Q1_est_beta = Q_est.Q1_est_beta
        self.Q2_est_beta = Q_est.Q2_est_beta
        self.Q3_est_beta = Q_est.Q3_est_beta
        self.Q4_est_beta = Q_est.Q4_est_beta
        self.bspline = Q_est.bspline
        self.para_dim = Q_est.para_dim 
        
        
        intercept_DE = self.eta_pie - self.eta_piea0
        intercept_ME = self.eta_piea0 - self.eta_piea0star
        intercept_SE = self.eta_piea0star - self.eta_a0
        

        termI1 = self.w_pie * self.pie_A / self.pieb_A * (np.copy(self.reward).flatten() + self.Q1_diff - self.eta_pie)
        termI2 = self.compute_termI2(data_num, self.state, self.action, self.reward, self.next_state, self.mediator)
        termI3 = self.compute_termI3(data_num, self.state, self.action, self.reward, self.next_state, self.mediator)
        termI4 = self.compute_termI4(data_num, self.state, self.action, self.reward, self.next_state)
        
        #DE
        self.ind_est[0] = intercept_DE + termI1 - termI2 #
        #ME
        self.ind_est[1] = intercept_ME +termI2 - termI3 # 
        #SE
        self.ind_est[2] = intercept_SE +termI3 - termI4 # 
        
        #DE
        self.ind_est[3] = intercept_DE
        #ME
        self.ind_est[4] = intercept_ME
        #SE
        self.ind_est[5] = intercept_SE
        
        #base DE
        self.ind_est[6] = self.compute_base_DE(data_num, self.state, self.action, self.reward, self.mediator)
        #base ME
        self.ind_est[7] = self.compute_base_ME(data_num, self.state, self.action, self.reward, self.mediator)

        est_DEMESE = np.mean(self.ind_est,1)
        var_DEMESE = np.var(self.ind_est,1)
        self.est_DEMESE = est_DEMESE
        self.var_DEMESE = var_DEMESE
        pass
        
        
    def compute_termI2(self, data_num, state, action, reward, next_state, mediator):
        pM_S = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pM_Sa = self.pmlearner.get_pm_prediction(state, np.array([a]), mediator)
            pie_a = self.target_policy(state, self.dim_state, a, matrix_based = True)
            pM_S += pie_a * pM_Sa
            
        pM_SA = self.pmlearner.get_pm_prediction(state, action, mediator)
        
        reward = np.copy(reward).flatten()
        Er_SAM = self.rewardlearner.get_reward_prediction(state, action, mediator)
        Er_Sa0M = self.rewardlearner.get_reward_prediction(state, np.array([self.a0]), mediator)
        
        termI2 = np.clip(pM_S / pM_SA, a_min=None, a_max=self.truncate)* self.w_pie * self.I_A / self.pieb_A * (reward - Er_SAM)
        #print(self.ratio_target.shape, Er_Sa0M.shape, self.Q2_diff.shape, self.eta_piea0.shape)
        termI2 += self.w_pie * self.pie_A / self.pieb_A * (Er_Sa0M + self.Q2_diff - self.eta_piea0)
        
        return termI2

        
    def compute_termI3(self, data_num, state, action, reward, next_state, mediator):
        reward = np.copy(reward).flatten()
        
        Er_Sa0 = np.zeros(data_num, dtype=float)
        Er_SA = np.zeros(data_num, dtype=float)
        sampled_reward_a0 = []
        sampled_reward_A = []
        for rep in range(self.expectation_MCMC_iter):
            m_i_A = self.pmlearner.sample_m(state, action, random = True)
            m_i_a0 = self.pmlearner.sample_m(state, self.a0, random = True)
            reward_i_A = self.rewardlearner.get_reward_prediction(state, action, m_i_A)
            reward_i_a0 = self.rewardlearner.get_reward_prediction(state, self.a0, m_i_a0)
            sampled_reward_A.append(reward_i_A)
            sampled_reward_a0.append(reward_i_a0)
        Er_SA = np.mean(sampled_reward_A,0)
        Er_Sa0 = np.mean(sampled_reward_a0,0)
        
        termI3 = self.w_pie * self.I_A / self.pieb_A * (reward - Er_SA)
        termI3 += self.w_pie * self.pie_A / self.pieb_A * (Er_Sa0 + self.Q3_diff - self.eta_piea0star)
        
        return termI3
         
    def compute_termI4(self, data_num, state, action, reward, next_state):
        reward = np.copy(reward).flatten()
        termI4 = self.w_a0 * self.I_A / self.pieb_A * (reward + self.Q4_diff - self.eta_a0)
        
        return termI4
    
    
    def compute_base_DE(self, data_num, state, action, reward, mediator):
        reward = np.copy(reward).flatten()
        base_DE = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pie_a = self.target_policy(state, self.dim_state, action = a, matrix_based = True).flatten()
            sampled_reward_a = []
            sampled_reward_a0 = []
            for rep in range(self.expectation_MCMC_iter):
                m_i_a = self.pmlearner.sample_m(state,  np.array([a]), random = True)
                reward_i_a = self.rewardlearner.get_reward_prediction(state,  np.array([a]), m_i_a)
                reward_i_a0 = self.rewardlearner.get_reward_prediction(state, self.a0, m_i_a)
                sampled_reward_a.append(reward_i_a)
                sampled_reward_a0.append(reward_i_a0)
            Er_Sa = np.mean(sampled_reward_a,0)
            Er_Sa0 = np.mean(sampled_reward_a0,0)
            
            base_DE += pie_a * (Er_Sa - Er_Sa0)
        return base_DE
    
    def compute_base_ME(self, data_num, state, action, reward, mediator):
        reward = np.copy(reward).flatten()
        base_ME = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pie_a = self.target_policy(state, self.dim_state, action = a, matrix_based = True).flatten()
            sampled_reward_a = []
            sampled_reward_a0 = []
            for rep in range(self.expectation_MCMC_iter):
                m_i_a = self.pmlearner.sample_m(state,  np.array([a]), random = True)
                m_i_a0 = self.pmlearner.sample_m(state, self.a0, random = True)
                reward_i_a = self.rewardlearner.get_reward_prediction(state, self.a0, m_i_a)
                reward_i_a0 = self.rewardlearner.get_reward_prediction(state, self.a0, m_i_a0)
                sampled_reward_a.append(reward_i_a)
                sampled_reward_a0.append(reward_i_a0)
            Er_Sa = np.mean(sampled_reward_a,0)
            Er_Sa0 = np.mean(sampled_reward_a0,0)
            
            base_ME += pie_a * (Er_Sa - Er_Sa0)
        return base_ME
        
        
