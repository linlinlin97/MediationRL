import numpy as np
#from problearner import PMLearner, PALearner
#from qlearner import Qlearner
#from rll import RatioLinearLearner
#from rnnl import RatioRKHSLearner, train
#from sklearn.model_selection import KFold
#import tensorflow as tf

class evaluator:
    def __init__(self, dataset,
                 QLearner1, QLearner2,
                 QLearner3, RatioLearner,
                 PMLearner, RewardLearner,
                 PALearner, EtaLearner,
                 target_policy=None, control_policy = None):
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

        self.target_policy = target_policy
        #control_policy
        self.control_policy = control_policy
        self.a0 = control_policy(get_a = True)

        self.qlearner1 = QLearner1
        self.qlearner2 = QLearner2
        self.qlearner3 = QLearner3
        self.ratiolearner = RatioLearner
        self.pmlearner = PMLearner
        self.rewardlearner = RewardLearner
        self.palearner = PALearner
        self.etalearner = EtaLearner
        
        self.eta_pi = self.etalearner.get_eta_prediction()

        self.unique_action = np.unique(self.action)
        self.unique_mediator = np.unique(self.mediator)
        self.est_DEMESE = None
        pass
    
    def estimate_DE_ME_SE(self):
        data_num = self.state.shape[0]
        self.ind_est = np.array([range(data_num),range(data_num),range(data_num)], dtype=float)
        #self.ind_est = np.array([range(data_num),range(data_num),range(data_num),range(data_num),range(data_num),range(data_num),range(data_num)], dtype=float)
        
        self.w_pie = self.ratiolearner.get_w_prediction(self.state, policy="target")
        self.w_a0 = self.ratiolearner.get_w_prediction(self.state, policy="control")
        
        pie_A = self.target_policy(self.state, self.action, matrix_based = True)
        pib_A = self.palearner.get_pa_prediction(self.state, self.action)
        self.pAS_ratio_eb = pie_A/pib_A
        
        pia0_A = self.control_policy(self.state, self.action, matrix_based = True)
        self.pAS_ratio_a0b = pia0_A/pib_A
        
        self.compute_Q(data_num, self.state, self.action, self.next_state)
        
        intercept_DE = self.compute_DE(data_num, self.state)
        intercept_ME = self.compute_ME(data_num, self.state)
        intercept_SE = self.compute_SE(data_num, self.state)
        
        self.eta_pia0 = self.eta_pi - intercept_DE
        self.eta_pia0star = self.eta_pia0 - intercept_ME
        self.eta_a0 = self.eta_pia0star - intercept_SE
        
        termI1 = self.compute_termI1(data_num, self.state, self.action, self.reward, self.next_state)
        termI2 = self.compute_termI2(data_num, self.state, self.action, self.reward, self.next_state, self.mediator)
        termI3 = self.compute_termI3(data_num, self.state, self.action, self.reward, self.next_state, self.mediator)
        termI4 = self.compute_termI4(data_num, self.state, self.action, self.reward, self.next_state)
        
        #DE
        self.ind_est[0] = intercept_DE + termI1 - termI2 #
        #ME
        self.ind_est[1] = intercept_ME +termI2 - termI3 # 
        #SE
        self.ind_est[2] = intercept_SE +termI3 - termI4 # 
        #self.ind_est[0] = termI1
        #self.ind_est[1] = termI2
        #self.ind_est[2] = termI3
        #self.ind_est[3] = termI4
        #self.ind_est[4] = intercept_DE
        #self.ind_est[5] = intercept_ME
        #self.ind_est[6] = intercept_SE
        
        est_DEMESE = np.mean(self.ind_est,1)
        var_DEMESE = np.var(self.ind_est,1)
        self.est_DEMESE = est_DEMESE
        self.var_DEMESE = var_DEMESE
        pass

    
    def compute_DE(self, data_num, state):
        intercept_DE = np.zeros(data_num, dtype=float)
        w_pie = np.copy(self.w_pie)
        for a in self.unique_action:
            pie_a = self.target_policy(state, a, matrix_based = True)
            for m in self.unique_mediator:
                Er_sam = self.rewardlearner.get_reward_prediction(state, a, m)
                Er_sa0m = self.rewardlearner.get_reward_prediction(state, self.a0, m)
                pm_a = self.pmlearner.get_pm_prediction(state, a, m)
                intercept_DE += w_pie * pie_a * (Er_sam - Er_sa0m) * pm_a
        return intercept_DE
    
    def compute_ME(self, data_num, state):
        intercept_ME = np.zeros(data_num, dtype=float)
        w_pie = np.copy(self.w_pie)
        for a in self.unique_action:
            for m in self.unique_mediator:
                Er_sa0m = self.rewardlearner.get_reward_prediction(state, self.a0, m)
                pm_a = self.pmlearner.get_pm_prediction(state, a, m)
                pm_a0 = self.pmlearner.get_pm_prediction(state, self.a0, m)
                intercept_ME += w_pie * Er_sa0m * (pm_a - pm_a0)

        return intercept_ME

    def compute_SE(self, data_num, state):
        intercept_SE = np.zeros(data_num, dtype=float)
        w_pie = np.copy(self.w_pie)
        w_a0 = np.copy(self.w_a0)
        for m in self.unique_mediator:
            Er_sa0m = self.rewardlearner.get_reward_prediction(state, self.a0, m)
            pm_a0 = self.pmlearner.get_pm_prediction(state, self.a0, m)
            intercept_SE += (w_pie - w_a0) * Er_sa0m * pm_a0 

        return intercept_SE

        
    def compute_termI1(self, data_num, state, action, reward, next_state):
        w_pie = np.copy(self.w_pie)
        pAS_ratio_eb = np.copy(self.pAS_ratio_eb)
        reward = np.copy(reward).flatten()
        Q1_Sprime = np.copy(self.Q1_Sprime)    
        Q1_SA = np.copy(self.Q1_SA)
        
        termI1 = w_pie * pAS_ratio_eb * (reward + Q1_Sprime - Q1_SA - self.eta_pi)
        
        return termI1
        
        
    def compute_termI2(self, data_num, state, action, reward, next_state, mediator):
        w_pie = np.copy(self.w_pie)
        
        pM_S = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pM_Sa = self.pmlearner.get_pm_prediction(state, a, mediator)
            pie_a = self.target_policy(state, a, matrix_based = True)
            pM_S += pie_a * pM_Sa
            
        pM_SA = self.pmlearner.get_pm_prediction(state, action, mediator)
        pAS_ratio_a0b = np.copy(self.pAS_ratio_a0b)
        reward = np.copy(reward).flatten()
        Er_SAM = self.rewardlearner.get_reward_prediction(state, action, mediator)
        pAS_ratio_eb = np.copy(self.pAS_ratio_eb)
        Er_Sa0M = self.rewardlearner.get_reward_prediction(state, self.a0, mediator)
        
        Q2_Sprime = np.copy(self.Q2_Sprime)  
        Q2_SA = np.copy(self.Q2_SA)
        
        termI2 = pM_S / pM_SA * pAS_ratio_a0b * (reward - Er_SAM)
        termI2 += pAS_ratio_eb * (Er_Sa0M + Q2_Sprime - Q2_SA - self.eta_pia0)
        termI2 *= w_pie
        
        return termI2

        
    def compute_termI3(self, data_num, state, action, reward, next_state, mediator):
        w_pie = np.copy(self.w_pie)
        pAS_ratio_a0b = np.copy(self.pAS_ratio_a0b)
        reward = np.copy(reward).flatten()
        Er_SAM = self.rewardlearner.get_reward_prediction(state, action, mediator)
        pAS_ratio_eb = np.copy(self.pAS_ratio_eb)
        
        Er_Sa0 = np.zeros(data_num, dtype=float)
        for m in self.unique_mediator:
            pm_Sa0 = self.pmlearner.get_pm_prediction(state, self.a0, m)
            Er_Sa0 += pm_Sa0 * self.rewardlearner.get_reward_prediction(state, self.a0, m) 
        
        Q3_Sprime = np.copy(self.Q3_Sprime)
        Q3_SA = np.copy(self.Q3_SA)
        
        termI3 = pAS_ratio_a0b * (reward - Er_SAM)
        termI3 += pAS_ratio_eb * (Er_Sa0 + Q3_Sprime - Q3_SA - self.eta_pia0star)
        termI3 *= w_pie
        
        return termI3
         
    def compute_termI4(self, data_num, state, action, reward, next_state):
        w_a0 = np.copy(self.w_a0)
        pAS_ratio_a0b = np.copy(self.pAS_ratio_a0b)
        reward = np.copy(reward).flatten()
        
        Q_Sa0_prime = np.zeros(data_num, dtype=float)
        for m in self.unique_mediator:
            pm_Sa0_prime = self.pmlearner.get_pm_prediction(next_state, self.a0, m)
            Q_Sa0_prime += pm_Sa0_prime * self.qlearner1.get_q_prediction(next_state, self.a0, m)
        
        Q1_SA = np.copy(self.Q1_SA)
        
        termI4 = w_a0 * pAS_ratio_a0b * (reward + Q_Sa0_prime - Q1_SA - self.eta_a0)
        
        return termI4
    
    
    def compute_Q(self, data_num, state, action, next_state):
        self.Q1_Sprime = np.zeros(data_num, dtype=float)
        self.Q2_Sprime = np.zeros(data_num, dtype=float)
        self.Q3_Sprime = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pie_a_Sprime = self.target_policy(next_state, a, matrix_based = True)
            for m in self.unique_mediator:
                pm_a_Sprime = self.pmlearner.get_pm_prediction(next_state, a, m)
                Q1_Sam_prime = self.qlearner1.get_q_prediction(next_state, a, m)
                Q2_Sam_prime = self.qlearner2.get_q_prediction(next_state, a, m)
                Q3_Sam_prime = self.qlearner3.get_q_prediction(next_state, a, m)
                self.Q1_Sprime += Q1_Sam_prime * pm_a_Sprime * pie_a_Sprime
                self.Q2_Sprime += Q2_Sam_prime * pm_a_Sprime * pie_a_Sprime
                self.Q3_Sprime += Q3_Sam_prime * pm_a_Sprime * pie_a_Sprime
                
        self.Q1_SA = np.zeros(data_num, dtype=float)
        self.Q2_SA = np.zeros(data_num, dtype=float)
        self.Q3_SA = np.zeros(data_num, dtype=float)
        for m in self.unique_mediator:
            pm_AS = self.pmlearner.get_pm_prediction(state, action, m)
            Q1_SAm = self.qlearner1.get_q_prediction(state, action, m)
            Q2_SAm = self.qlearner2.get_q_prediction(state, action, m)
            Q3_SAm = self.qlearner3.get_q_prediction(state, action, m)
            self.Q1_SA += Q1_SAm * pm_AS
            self.Q2_SA += Q2_SAm * pm_AS
            self.Q3_SA += Q3_SAm * pm_AS