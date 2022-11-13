import numpy as np
#from problearner import PMLearner, PALearner
#from qlearner import Qlearner
#from rll import RatioLinearLearner
#from rnnl import RatioRKHSLearner, train
#from sklearn.model_selection import KFold
#import tensorflow as tf

class evaluator:
    def __init__(self, dataset,N, T,
                 QLearner, PMLearner, 
                 RewardLearner, PALearner, RatioLearner,
                 problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,20)},
                 toy_binary = False,
                 ratio_ndim = 5, truncate = 20, l2penalty = 1.0,
                 target_policy=None, control_policy = None, dim_state = 1, dim_mediator = 1,
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
        self.N = N
        self.T = T
        
        self.dataset = dataset
        self.dim_state = dim_state
        self.dim_mediator = dim_mediator
        self.truncate = truncate
        
        self.target_policy = target_policy
        #control_policy
        self.control_policy = control_policy
        self.a0 = control_policy(get_a = True)
        
        self.qlearner = QLearner
        self.pmlearner = PMLearner(dataset, problearner_parameters, seed, dim_state = dim_state,
                                   dim_mediator = dim_mediator, toy_binary = toy_binary)
        self.pmlearner.train()
        self.rewardlearner = RewardLearner(dataset, problearner_parameters, seed, dim_state = dim_state,
                                           dim_mediator = dim_mediator, toy_binary = toy_binary)
        self.rewardlearner.train()
        self.palearner = PALearner(dataset, problearner_parameters, seed, test = False, dim_state = dim_state,
                                   dim_mediator = dim_mediator)
        self.palearner.train()
        self.ratiolearner = RatioLearner(dataset, target_policy, control_policy, self.palearner, ndim=ratio_ndim,
                                         truncate=truncate, dim_state = dim_state, l2penalty = l2penalty)
        self.ratiolearner.fit()
        
        self.w_pie = self.ratiolearner.get_r_prediction(self.state, policy = 'target', normalize=True)
        self.w_a0 = self.ratiolearner.get_r_prediction(self.state, policy = 'control', normalize=True)
        self.pie_A = self.target_policy(self.state, self.dim_state, action = self.action)
        self.I_A = self.control_policy(self.state, self.dim_state, self.action)
        self.pieb_A = self.palearner.get_pa_prediction(self.state, self.action)
        
        self.unique_action = np.unique(self.action)
        self.unique_mediator = np.unique(self.mediator)
        self.est_DEMESE = None
        
        self.est_Q()
        
        pass
    
    def est_Q(self):
        data_num = self.state.shape[0]
        self.ind_est = np.array([range(data_num)] * 8, dtype=float)
        self.q1 = self.qlearner(self.dataset, self.target_policy, self.pmlearner, self.rewardlearner, self.control_policy, Q_func = "Q1")
    
        self.q2 = self.qlearner(self.dataset, self.target_policy, self.pmlearner, self.rewardlearner, self.control_policy, Q_func = "Q2")
    
        self.q3 = self.qlearner(self.dataset, self.target_policy, self.pmlearner, self.rewardlearner, self.control_policy, Q_func = "Q3")
    
        self.q4 = self.qlearner(self.dataset, self.control_policy, self.pmlearner, self.rewardlearner, self.control_policy, Q_func = "Q4")
        
        
    def estimate_DE_ME_SE(self):    
        data_num = self.N * self.T
        self.Q1_diff, self.eta_pie = self.q1.Q_diff, self.q1.eta
        self.Q2_diff, self.eta_piea0 = self.q2.Q_diff, self.q2.eta
        self.Q3_diff, self.eta_piea0star = self.q3.Q_diff, self.q3.eta
        self.Q4_diff, self.eta_a0 = self.q4.Q_diff, self.q4.eta
        self.Q1_est_beta = self.q1.Q_beta_hat
        self.Q2_est_beta = self.q2.Q_beta_hat
        self.Q3_est_beta = self.q3.Q_beta_hat
        self.Q4_est_beta = self.q4.Q_beta_hat
        
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
        self.ind_est[1] = intercept_ME + termI2 - termI3 # 
        #SE
        self.ind_est[2] = intercept_SE + termI3 - termI4 # 
        
        #DE
        self.ind_est[3] = intercept_DE
        #ME
        self.ind_est[4] = intercept_ME
        #SE
        self.ind_est[5] = intercept_SE
        
        #base DE
        self.ind_est[6] = self.compute_base_DE(data_num, self.state)
        #base ME
        self.ind_est[7] = self.compute_base_ME(data_num, self.state)

        self.est_DEMESE = np.mean(self.ind_est,1)
        if data_num > 100:
            self.se_DEMESE = np.array([np.mean(self.ind_est[:,i*self.T:(i+1)*self.T],1) for i in range(self.N)]).std(0)/np.sqrt(self.N)
        else:
            self.se_DEMESE = np.std(self.ind_est,1)/np.sqrt(data_num)
        pass
        
        
    def compute_termI2(self, data_num, state, action, reward, next_state, mediator):
        pM_S = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pM_Sa = self.pmlearner.get_pm_prediction(state, np.array([a]), mediator)
            pie_a = self.target_policy(state, self.dim_state, a)
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
        for m in self.unique_mediator:
            pm_Sa0 = self.pmlearner.get_pm_prediction(state, self.a0, np.array([m]))
            Er_Sa0 += pm_Sa0 * self.rewardlearner.get_reward_prediction(state, self.a0, np.array([m])) 
            
            pm_SA = self.pmlearner.get_pm_prediction(state, action, np.array([m]))
            Er_SA += pm_SA * self.rewardlearner.get_reward_prediction(state, action, np.array([m]))
        
        termI3 = self.w_pie * self.I_A / self.pieb_A * (reward - Er_SA)
        termI3 += self.w_pie * self.pie_A / self.pieb_A * (Er_Sa0 + self.Q3_diff - self.eta_piea0star)
        
        return termI3
         
    def compute_termI4(self, data_num, state, action, reward, next_state):
        reward = np.copy(reward).flatten()
        termI4 = self.w_a0 * self.I_A / self.pieb_A * (reward + self.Q4_diff - self.eta_a0)
        
        return termI4
    
    
    def compute_base_DE(self, data_num, state):
        intercept_DE = np.zeros(data_num, dtype=float)
        #w_pie = np.copy(self.w_pie)
        for a in self.unique_action:
            pie_a = self.target_policy(state, self.dim_state, a)
            for m in self.unique_mediator:
                Er_sam = self.rewardlearner.get_reward_prediction(state, np.array([a]), np.array([m]))
                Er_sa0m = self.rewardlearner.get_reward_prediction(state, self.a0, np.array([m]))
                pm_a = self.pmlearner.get_pm_prediction(state, np.array([a]), np.array([m]))
                #intercept_DE += w_pie * pie_a * (Er_sam - Er_sa0m) * pm_a
                intercept_DE += pie_a * (Er_sam - Er_sa0m) * pm_a
        return intercept_DE
    
    def compute_base_ME(self, data_num, state):
        intercept_ME = np.zeros(data_num, dtype=float)
        #w_pie = np.copy(self.w_pie)
        for a in self.unique_action:
            pie_a = self.target_policy(state, self.dim_state, a)
            for m in self.unique_mediator:
                Er_sa0m = self.rewardlearner.get_reward_prediction(state, self.a0, np.array([m]))
                pm_a = self.pmlearner.get_pm_prediction(state, np.array([a]), np.array([m]))
                pm_a0 = self.pmlearner.get_pm_prediction(state, self.a0, np.array([m]))
                #intercept_ME += w_pie * pie_a * Er_sa0m * (pm_a - pm_a0)
                intercept_ME += pie_a * Er_sa0m * (pm_a - pm_a0)

        return intercept_ME
    
    def update_exp(self, rep, old_est, new_obs):
        return (rep*old_est + new_obs)/(rep+1)
        