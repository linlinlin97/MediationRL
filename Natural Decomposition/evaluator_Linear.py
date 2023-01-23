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
                 RewardLearner, PALearner, RatioLearner,
                 problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,20)},
                 ratio_ndim = 5, truncate = 20, l2penalty = 1.0,
                 t_depend_target = False, target_policy=None, control_policy = None, 
                 dim_state = 1, dim_mediator = 1, 
                 Q_settings = {'scaler': 'Identity','product_tensor': True, 'beta': 3/7, 'include_intercept': False,
                               'expectation_MCMC_iter_Q3': 100, 'expectation_MCMC_iter_Q_diff':100, 'penalty': 10**(-9),
                              'd': 3, 'min_L': 7, "t_dependent_Q": False},
                 expectation_MCMC_iter = 100, seed = 0, nature_decomp = False):
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
        self.nature_decomp =  nature_decomp
        
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
        
        self.qlearner = QLearner
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

        self.ratiolearner = RatioLearner(dataset, target_policy, control_policy, self.palearner, self.pmlearner,
                                         ndim=ratio_ndim, truncate=truncate, dim_state = dim_state, dim_mediator = dim_mediator, 
                                         l2penalty = l2penalty, t_depend_target = t_depend_target, nature_decomp = nature_decomp)
        self.ratiolearner.fit()
        self.w_pie = self.ratiolearner.get_r_prediction(self.state, policy = 'target', normalize=True)
        self.w_a0 = self.ratiolearner.get_r_prediction(self.state, policy = 'control', normalize=True)
        self.w_G = self.ratiolearner.get_r_prediction(self.state, policy = 'G', normalize=True)
        if self.t_depend_target:
            self.pie_A = self.target_policy(self.state, self.dim_state, action = self.action, time_idx = self.time_idx)
        else:
            self.pie_A = self.target_policy(self.state, self.dim_state, action = self.action)
        self.I_A = self.control_policy(self.state, self.dim_state, self.action)
        self.pieb_A = self.palearner.get_pa_prediction(self.state, self.action)
        
        if np.sum(self.pie_A) != np.sum(self.ratiolearner.target_pa) or np.sum(self.I_A) != np.sum(self.ratiolearner.control_pa) or np.sum(self.pieb_A) != np.sum(self.ratiolearner.estimate_pa):
            raise ValueError('Inconsistent pa estimation.')
            
        self.unique_action = np.unique(self.action)
        self.unique_mediator = np.unique(self.mediator)
        self.est_DEMESE = None
        
        self.Q_settings = Q_settings

        pass
    
    def estimate_DE_ME_SE(self):
        if self.nature_decomp:
            self.estimate_DE_ME_SE_decopm2()
        else:
            self.estimate_DE_ME_SE_decopm1()
            
    def estimate_DE_ME_SE_decopm1(self):
        data_num = self.N * self.T
        self.ind_est = np.array([range(data_num)] * 17, dtype=float)
        #Multiply Robust - Related
        self._Q_terms()
        termI1, termI2, termI3, termI4, termI5 = self._I_terms(data_num)
        
        intercept_IDE = self.eta_pie - self.eta_piea0
        intercept_IME = self.eta_piea0 - self.eta_piea0star
        intercept_DDE = self.eta_piea0star - self.eta_G
        intercept_DME = self.eta_G - self.eta_a0
        
        #WIS - Related
        WISR1 = self.w_pie * self.pie_A / self.pieb_A * np.copy(self.reward).flatten()
        WISR2_1 = self.w_pie * self.I_A / self.pieb_A * np.clip(self.rho_SAM, a_min=None, a_max=self.truncate) * np.copy(self.reward).flatten()
        WISR2_2 = self.w_pie * self.pie_A / self.pieb_A * self.Er_Sa0M 
        WISR3 = self.w_pie * self.I_A / self.pieb_A * np.copy(self.reward).flatten()
        WISR4 = self.w_G * self.I_A / self.pieb_A * np.copy(self.reward).flatten()
        WISR5 = self.w_a0 * self.I_A / self.pieb_A * np.copy(self.reward).flatten()
        
        #Multiply Robust
        #IDE
        self.ind_est[0] = intercept_IDE + termI1 - termI2 #
        #IME
        self.ind_est[1] = intercept_IME + termI2 - termI3 # 
        #DDE
        self.ind_est[2] = intercept_DDE + termI3 - termI4 # 
        #DME
        self.ind_est[3] = intercept_DME + termI4 - termI5 # 
        
        #Direct
        #IDE
        self.ind_est[4] = intercept_IDE
        #IME
        self.ind_est[5] = intercept_IME
        #DDE
        self.ind_est[6] = intercept_DDE
        #DME
        self.ind_est[7] = intercept_DME
        
        #WIS1
        #IDE
        self.ind_est[8] = WISR1 - WISR2_1
        #IME
        self.ind_est[9] = WISR2_1 - WISR3
        #DDE
        self.ind_est[10] = WISR3 - WISR4
        #DME
        self.ind_est[11] = WISR4 - WISR5
        
        #WIS2
        #IDE
        self.ind_est[12] = WISR1 - WISR2_2
        #IME
        self.ind_est[13] = WISR2_2 - WISR3
        
        #Baseline
        #base DE
        self.ind_est[14] = self.compute_base_DE(data_num, self.state, self.action, self.reward, self.mediator, self.time_idx)
        #base ME
        self.ind_est[15] = self.compute_base_ME(data_num, self.state, self.action, self.reward, self.mediator, self.time_idx)
        
        #MR-Total Effect
        self.ind_est[16] = self.ind_est[0] + self.ind_est[1] + self.ind_est[2] + self.ind_est[3]
        
        self.est_DEMESE = np.mean(self.ind_est,1)
        if data_num > 100:
            self.se_DEMESE = np.array([np.mean(self.ind_est[:,i*self.T:(i+1)*self.T],1) for i in range(self.N)]).std(0)/np.sqrt(self.N)
        else:
            self.se_DEMESE = np.std(self.ind_est,1)/np.sqrt(data_num)
        pass

    def estimate_DE_ME_SE_decopm2(self):
        data_num = self.N * self.T
        self.ind_est = np.array([range(data_num)] * 5, dtype=float)
        #Multiply Robust - Related
        self._Q_terms()
        termI1, termI2, termI3, termI4, termI5 = self._I_terms(data_num)
        
        intercept_IDE = self.eta_piea0 - self.eta_a0
        intercept_IME = self.eta_piea0star - self.eta_piea0
        intercept_DDE = self.eta_G - self.eta_piea0star
        intercept_DME = self.eta_pie - self.eta_G
        
        #Multiply Robust
        #IDE
        self.ind_est[0] = intercept_IDE + termI2 - termI5 #
        #IME
        self.ind_est[1] = intercept_IME + termI3 - termI2 # 
        #DDE
        self.ind_est[2] = intercept_DDE + termI4 - termI3 # 
        #DME
        self.ind_est[3] = intercept_DME + termI1 - termI4 # 
        
        #MR-Total Effect
        self.ind_est[4] = self.ind_est[0] + self.ind_est[1] + self.ind_est[2] + self.ind_est[3]
        
        self.est_DEMESE = np.mean(self.ind_est,1)
        if data_num > 100:
            self.se_DEMESE = np.array([np.mean(self.ind_est[:,i*self.T:(i+1)*self.T],1) for i in range(self.N)]).std(0)/np.sqrt(self.N)
        else:
            self.se_DEMESE = np.std(self.ind_est,1)/np.sqrt(data_num)
        pass
    
    def _Q_terms(self):
        Q_est = self.qlearner(self.dataset, self.target_policy, self.control_policy, self.pmlearner, self.rewardlearner,
                              self.dim_state, self.dim_mediator, 
                              self.Q_settings, self.seed, self.t_depend_target, self.nature_decomp)
        Q_est.est_Q1()
        Q_est.est_Q2()
        Q_est.est_Q3()
        Q_est.est_Q4()
        Q_est.est_Q5()
        Q_est.est_Qdiffs()
        if self.nature_decomp:
            # in qlearner, we reverse the target and the control policy
            self.Q1_diff, self.eta_pie = Q_est.Q5_diff, Q_est.eta_a0
            self.Q5_diff, self.eta_a0 = Q_est.Q1_diff, Q_est.eta_pie
        else:
            self.Q1_diff, self.eta_pie = Q_est.Q1_diff, Q_est.eta_pie
            self.Q5_diff, self.eta_a0 = Q_est.Q5_diff, Q_est.eta_a0
        self.Q2_diff, self.eta_piea0 = Q_est.Q2_diff, Q_est.eta_piea0
        self.Q3_diff, self.eta_piea0star = Q_est.Q3_diff, Q_est.eta_piea0star
        self.Q4_diff_1, self.Q4_diff_2, self.eta_G = Q_est.Q4_diff_1, Q_est.Q4_diff_2, Q_est.eta_G
        self.Q1_est_beta = Q_est.Q1_est_beta
        self.Q2_est_beta = Q_est.Q2_est_beta
        self.Q3_est_beta = Q_est.Q3_est_beta
        self.Q4_est_beta = Q_est.Q4_est_beta
        self.Q5_est_beta = Q_est.Q5_est_beta
        self.bspline = Q_est.bspline
        self.para_dim = Q_est.para_dim 
        
    def _I_terms(self, data_num):
        self.Er_Sa0, self.Er_SA = self._Er_Sa0_SA(data_num)
        self.rho_SAM = self.rho_SAM(self.state, self.action, self.mediator, self.time_idx)
        
        termI1 = self.w_pie * self.pie_A / self.pieb_A * (np.copy(self.reward).flatten() + self.Q1_diff - self.eta_pie)
        termI2 = self.compute_termI2(data_num, self.state, self.action, self.reward, self.next_state, self.mediator,
                                     self.time_idx)
        termI3 = self.compute_termI3(data_num, self.state, self.action, self.reward, self.next_state, self.mediator)
        termI4 = self.compute_termI4(data_num, self.state, self.action, self.reward, self.next_state, self.mediator)
        
        termI5 = self.w_a0 * self.I_A / self.pieb_A * (np.copy(self.reward).flatten() + self.Q5_diff - self.eta_a0)
        return termI1, termI2, termI3, termI4, termI5
        
    def compute_termI2(self, data_num, state, action, reward, next_state, mediator, time_idx = None):
        #if self.nature_decomp get the I6
        reward = np.copy(reward).flatten()
        Er_SAM = self.rewardlearner.get_reward_prediction(state, action, mediator)
        self.Er_Sa0M = self._Er_Sa0M(data_num)
        #self.Er_Sa0M = self.rewardlearner.get_reward_prediction(state, np.array([self.a0]), mediator)
        
        if self.nature_decomp:
            termI2 = np.clip(self.rho_SAM, a_min=None, a_max=self.truncate)* self.w_a0 * self.pie_A / self.pieb_A * (reward - Er_SAM)
            termI2 += self.w_a0 * self.I_A / self.pieb_A * (self.Er_Sa0M + self.Q2_diff - self.eta_piea0)
        else:
            termI2 = np.clip(self.rho_SAM, a_min=None, a_max=self.truncate)* self.w_pie * self.I_A / self.pieb_A * (reward - Er_SAM)
            termI2 += self.w_pie * self.pie_A / self.pieb_A * (self.Er_Sa0M + self.Q2_diff - self.eta_piea0)
        
        return termI2

        
    def compute_termI3(self, data_num, state, action, reward, next_state, mediator):
        #if self.nature_decomp get the I7
        reward = np.copy(reward).flatten()
        if self.nature_decomp:
            termI3 = self.w_a0 * self.pie_A / self.pieb_A * (reward - self.Er_SA)
            termI3 += self.w_a0 * self.I_A / self.pieb_A * (self.Er_Sa0 + self.Q3_diff - self.eta_piea0star)
        else:
            termI3 = self.w_pie * self.I_A / self.pieb_A * (reward - self.Er_SA)
            termI3 += self.w_pie * self.pie_A / self.pieb_A * (self.Er_Sa0 + self.Q3_diff - self.eta_piea0star)
        
        return termI3
    
    def compute_termI4(self, data_num, state, action, reward, next_state, mediator):
        #if self.nature_decomp get the I8
        reward = np.copy(reward).flatten()
        if self.nature_decomp:
            termI4 = self.w_G * self.pie_A / self.pieb_A * (reward - self.Er_SA + self.rho_SAM * (self.Er_Sa0 + self.Q4_diff_1 - self.eta_G))
            termI4 += self.w_G * self.I_A / self.pieb_A * self.Q4_diff_2
        else:
            termI4 = self.w_G * self.I_A / self.pieb_A * (reward - self.Er_SA + self.rho_SAM * (self.Er_Sa0 + self.Q4_diff_1 - self.eta_G))
            termI4 += self.w_G * self.pie_A / self.pieb_A * self.Q4_diff_2
        
        return termI4  
    
    def _Er_Sa0_SA(self,data_num):
        Er_Sa0m = np.zeros(data_num, dtype=float)
        for a in self.unique_action:       
            if self.nature_decomp:
                p_a_S = self.target_policy(self.state, self.dim_state, a)
            else:
                p_a_S = self.control_policy(self.state, self.dim_state, a)
            if np.sum(p_a_S) > 0:
                Er_Sam = np.zeros(data_num, dtype=float)
                Er_SA = np.zeros(data_num, dtype=float)
                for rep in range(self.expectation_MCMC_iter):
                    #np.random.seed(rep)
                    m_S_A = self.pmlearner.sample_m(self.state, self.action, random = True)
                    m_Sa = self.pmlearner.sample_m(self.state, np.array([a]), random = True)
                    r_SAm = self.rewardlearner.get_reward_prediction(self.state, self.action, m_S_A)
                    r_Sam = self.rewardlearner.get_reward_prediction(self.state, np.array([a]), m_Sa)
                    Er_SA = self.update_exp(rep, Er_SA, r_SAm.reshape((-1,)))
                    Er_Sam = self.update_exp(rep, Er_Sam, r_Sam.reshape((-1,)))
                Er_Sa0m += p_a_S * Er_Sam
        return Er_Sa0m, Er_SA
    
    def rho_SAM(self, state, action, mediator, time_idx = None):
        data_num = len(action)
        pM_S = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pM_Sa = self.pmlearner.get_pm_prediction(state, np.array([a]), mediator)
            if self.nature_decomp:
                pi0_a = self.control_policy(state, self.dim_state, a)
                pM_S += pi0_a * pM_Sa
            else:
                if self.t_depend_target:
                    pie_a = self.target_policy(state, self.dim_state, a, time_idx)
                else:
                    pie_a = self.target_policy(state, self.dim_state, a)
                pM_S += pie_a * pM_Sa
            
        pM_SA = self.pmlearner.get_pm_prediction(state, action, mediator)
        
        return pM_S / pM_SA
    
    def compute_base_DE(self, data_num, state, action, reward, mediator, time_idx = None):
        reward = np.copy(reward).flatten()
        base_DE = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            if self.t_depend_target:
                pie_a = self.target_policy(state, self.dim_state, action = a, time_idx = time_idx).flatten()
            else:
                pie_a = self.target_policy(state, self.dim_state, action = a).flatten()
            Er_Sa = np.zeros(data_num, dtype=float)
            Er_Sa0 = np.zeros(data_num, dtype=float)
            for rep in range(self.expectation_MCMC_iter):
                m_i_a = self.pmlearner.sample_m(state,  np.array([a]), random = True)
                reward_i_a = self.rewardlearner.get_reward_prediction(state,  np.array([a]), m_i_a)
                reward_i_a0 = self.rewardlearner.get_reward_prediction(state, self.a0, m_i_a)
                Er_Sa = self.update_exp(rep, Er_Sa, reward_i_a.reshape((-1,)))
                Er_Sa0 = self.update_exp(rep, Er_Sa0, reward_i_a0.reshape((-1,)))
            
            base_DE += pie_a * (Er_Sa - Er_Sa0)
        return base_DE
    
    def compute_base_ME(self, data_num, state, action, reward, mediator, time_idx = None):
        reward = np.copy(reward).flatten()
        base_ME = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            if self.t_depend_target:
                pie_a = self.target_policy(state, self.dim_state, action = a, time_idx = time_idx).flatten()
            else:
                pie_a = self.target_policy(state, self.dim_state, action = a).flatten()
            Er_Sa = np.zeros(data_num, dtype=float)
            Er_Sa0 = np.zeros(data_num, dtype=float)
            for rep in range(self.expectation_MCMC_iter):
                m_i_a = self.pmlearner.sample_m(state,  np.array([a]), random = True)
                m_i_a0 = self.pmlearner.sample_m(state, self.a0, random = True)
                reward_i_a = self.rewardlearner.get_reward_prediction(state, self.a0, m_i_a)
                reward_i_a0 = self.rewardlearner.get_reward_prediction(state, self.a0, m_i_a0)
                Er_Sa = self.update_exp(rep, Er_Sa, reward_i_a.reshape((-1,)))
                Er_Sa0 = self.update_exp(rep, Er_Sa0, reward_i_a0.reshape((-1,)))
            
            base_ME += pie_a * (Er_Sa - Er_Sa0)
        return base_ME
    
    def update_exp(self, rep, old_est, new_obs):
        return (rep*old_est + new_obs)/(rep+1)
    
    def _Er_Sa0M(self,data_num):
        Er_Sa0M = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            if self.nature_decomp:
                p_a_S = self.target_policy(self.state, self.dim_state, a)
            else:
                p_a_S = self.control_policy(self.state, self.dim_state, a)
            Er_SaM = self.rewardlearner.get_reward_prediction(self.state, np.array([a]), self.mediator)
            Er_Sa0M += p_a_S * Er_SaM
        #else:
        #    Er_Sa0M = self.rewardlearner.get_reward_prediction(self.state, self.a0, self.mediator)
        return Er_Sa0M
        
        
