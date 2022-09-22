import numpy as np
#from problearner import PMLearner, PALearner
#from qlearner import Qlearner
#from rll import RatioLinearLearner
#from rnnl import RatioRKHSLearner, train
#from sklearn.model_selection import KFold
#import tensorflow as tf

class evaluator:
    def __init__(self, dataset,
                 QLearner, PMLearner, 
                 RewardLearner, PALearner, RatioLearner,
                 problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,20)},
                 ratio_ndim = 5, truncate = 20, 
                 target_policy=None, control_policy = None, dim_state = 1,
                 dim_mediator = 1, scaler = 'Identity', expectation_MCMC_iter = 100, seed = 0):
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
        self.state = np.copy(dataset['state'])
        self.action = np.copy(dataset['action'])
        self.mediator = np.copy(dataset['mediator'])
        self.reward = np.copy(dataset['reward'])
        self.next_state = np.copy(dataset['next_state'])
        self.s0 = np.copy(dataset['s0'])
        
        self.dataset = dataset
        self.dim_state = dim_state
        self.dim_mediator = dim_mediator
        self.scaler = scaler
        
        self.target_policy = target_policy
        #control_policy
        self.control_policy = control_policy
        self.a0 = control_policy(get_a = True)
        
        self.expectation_MCMC_iter = expectation_MCMC_iter
        
        self.qlearner = QLearner
        self.pmlearner = PMLearner(dataset, problearner_parameters, seed)
        self.pmlearner.train()
        self.rewardlearner = RewardLearner(dataset, problearner_parameters, seed)
        self.rewardlearner.train()
        self.palearner = PALearner(dataset, problearner_parameters, seed, test = False)
        self.palearner.train()
        self.ratiolearner = RatioLearner(dataset, target_policy, control_policy, self.palearner, ndim=ratio_ndim, truncate=truncate)
        self.ratiolearner.fit()
        
        
        self.unique_action = np.unique(self.action)
        self.unique_mediator = np.unique(self.mediator)
        self.est_DEMESE = None
        pass
    
    def estimate_DE_ME_SE(self):
        data_num = self.state.shape[0]
        self.ind_est = np.array([range(data_num),range(data_num),range(data_num),range(data_num),range(data_num),range(data_num)], dtype=float)
        Q_est = self.qlearner(self.dataset, self.target_policy, self.control_policy, self.pmlearner, self.rewardlearner, self.ratiolearner, self.palearner, self.unique_action, self.dim_state, self.dim_mediator, self.scaler, self.expectation_MCMC_iter)
    
        self.Q1_diff, self.eta_pie, self.Q2_diff, self.eta_piea0, self.Q3_diff, self.eta_piea0star, self.Q4_diff, self.eta_a0, self.ratio_target, self.ratio_control = Q_est.Q1_diff, Q_est.eta_pie, Q_est.Q2_diff, Q_est.eta_piea0, Q_est.Q3_diff, Q_est.eta_piea0star, Q_est.Q4_diff, Q_est.eta_a0, Q_est.ratio_target, Q_est.ratio_control
        self.time_rec = Q_est.time_rec
        self.Q1_est_beta, self.Q2_est_beta, self.Q3_est_beta, self.Q4_est_beta = Q_est.Q1_est_beta, Q_est.Q2_est_beta, Q_est.Q3_est_beta, Q_est.Q4_est_beta
        self.bspline = Q_est.bspline
        self.para_dim = Q_est.para_dim
        
        
        intercept_DE = self.eta_pie - self.eta_piea0
        intercept_ME = self.eta_piea0 - self.eta_piea0star
        intercept_SE = self.eta_piea0star - self.eta_a0
        

        termI1 = self.ratio_target * (np.copy(self.reward).flatten() + self.Q1_diff - self.eta_pie)
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

        est_DEMESE = np.mean(self.ind_est,1)
        var_DEMESE = np.var(self.ind_est,1)
        self.est_DEMESE = est_DEMESE
        self.var_DEMESE = var_DEMESE
        pass
        
        
    def compute_termI2(self, data_num, state, action, reward, next_state, mediator):
        pM_S = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pM_Sa = self.pmlearner.get_pm_prediction(state, np.array([a]), mediator)
            pie_a = self.target_policy(state, a, matrix_based = True)
            pM_S += pie_a * pM_Sa
            
        pM_SA = self.pmlearner.get_pm_prediction(state, action, mediator)
        
        reward = np.copy(reward).flatten()
        Er_SAM = self.rewardlearner.get_reward_prediction(state, action, mediator)
        Er_Sa0M = self.rewardlearner.get_reward_prediction(state, np.array([self.a0]), mediator)
        
        pie_A = self.target_policy(state, action, matrix_based = True)
        I_A = self.control_policy(state, action, matrix_based = True)
        termI2 = pM_S / pM_SA * self.ratio_target * (I_A / pie_A) * (reward - Er_SAM)
        print(self.ratio_target.shape, Er_Sa0M.shape, self.Q2_diff.shape, self.eta_piea0.shape)
        termI2 += self.ratio_target * (Er_Sa0M + self.Q2_diff - self.eta_piea0)
        
        return termI2

        
    def compute_termI3(self, data_num, state, action, reward, next_state, mediator):
        reward = np.copy(reward).flatten()
        
        Er_Sa0 = np.zeros(data_num, dtype=float)
        Er_SA = np.zeros(data_num, dtype=float)
        sampled_reward_a0 = []
        sampled_reward_A = []
        for rep in range(self.expectation_MCMC_iter):
            np.random.seed(rep)
            m_i_A = self.pmlearner.sample_m(state, action, random = True)
            m_i_a0 = self.pmlearner.sample_m(state, self.a0, random = True)
            reward_i_A = self.rewardlearner.get_reward_prediction(state, action, m_i_A)
            reward_i_a0 = self.rewardlearner.get_reward_prediction(state, self.a0, m_i_a0)
            sampled_reward_A.append(reward_i_A)
            sampled_reward_a0.append(reward_i_a0)
        Er_SA = np.mean(sampled_reward_A,0)
        Er_Sa0 = np.mean(sampled_reward_a0,0)
        #Em_Sa0 = self.pmlearner.get_pm_prediction(self.state, np.array([self.a0]), mediator = None)
        #Er_Sa0 = self.rewardlearner.get_reward_prediction(self.state, np.array([self.a0]), Em_Sa0)
        #Em_SA = self.pmlearner.get_pm_prediction(self.state, self.action, mediator = None)
        #Er_SA = self.rewardlearner.get_reward_prediction(self.state, self.action, Em_SA)
        
        pie_A = self.target_policy(state, action, matrix_based = True)
        I_A = self.control_policy(state, action, matrix_based = True)
        termI3 = self.ratio_target * (I_A / pie_A) * (reward - Er_SA)
        termI3 += self.ratio_target * (Er_Sa0 + self.Q3_diff - self.eta_piea0star)
        
        return termI3
         
    def compute_termI4(self, data_num, state, action, reward, next_state):
        reward = np.copy(reward).flatten()
        termI4 = self.ratio_control * (reward + self.Q4_diff - self.eta_a0)
        
        return termI4
