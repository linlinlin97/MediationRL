import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.linear_model import SGDRegressor, LinearRegression, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import PolynomialFeatures
from sklearn.ensemble import RandomForestRegressor
from sklearn.base import clone
from scipy.special import expit

class Qlearner:
    def __init__(self, data, simulator, target_policy, PMLearner, RewardModel, control_policy, Q_func = "Q1", true_etas = None):          
        self.data = data
        self.action = np.copy(data['action']).flatten()
        self.mediator = np.copy(data['mediator']).flatten()
        self.state = np.copy(data['state']).flatten()
        self.next_state = np.copy(data['next_state']).flatten()
        self.simulator = simulator
        
        self.unique_action = np.unique(data['action'])
        self.unique_mediator = np.unique(data['mediator'])
        self.unique_state = np.unique(data['state'])
        
        self.pmlearner = PMLearner
        self.target_policy = target_policy
        self.control_policy = control_policy
        
        if Q_func == "Q1":
            self.Y = RewardModel(self.state,self.action,self.mediator, matrix_based = True)
        elif Q_func == "Q2":
            self.a0 = control_policy(get_a = True)
            self.Y = RewardModel(self.state,self.a0,self.mediator, matrix_based = True)
        elif Q_func == "Q3" or Q_func == 'Q4':
            self.a0 = control_policy(get_a = True)
            self.Y = np.zeros(shape=self.state.shape).flatten()
            for m in self.unique_mediator:
                Er_Sa0m = RewardModel(self.state,self.a0,m, matrix_based = True)
                pm_Sa0 = self.pmlearner(self.state, self.a0, m)
                self.Y += Er_Sa0m * pm_Sa0  
        elif Q_func == "Q5":
            self.a0 = control_policy(get_a = True)
            self.Y = RewardModel(self.state,self.action,self.mediator, matrix_based = True)
        else:
            "Warning!"
            
        self.phi = self.get_phi(self.state, self.action, self.mediator, include_eta = True)
        self.w = self.get_exp_w(self.next_state, Q_func)

        
    def get_phi(self, state, action, mediator, include_eta = False):
        '''000, 100, 010, 001, 110, 101, 011, 111'''
        state1 = np.copy(state).flatten()
        indicator = np.zeros(shape = (len(state1),8)) #last column saved for eta
        #indicator[(state1==0)*(action==0)*(mediator==0),0] = 1 reference group
        indicator[(state1==1)*(action==0)*(mediator==0),0] = 1
        indicator[(state1==0)*(action==1)*(mediator==0),1] = 1
        indicator[(state1==0)*(action==0)*(mediator==1),2] = 1
        indicator[(state1==1)*(action==1)*(mediator==0),3] = 1
        indicator[(state1==1)*(action==0)*(mediator==1),4] = 1
        indicator[(state1==0)*(action==1)*(mediator==1),5] = 1
        indicator[(state1==1)*(action==1)*(mediator==1),6] = 1
        if include_eta:
            indicator[:,7] = 1
            
        return indicator


    def get_w(self, next_state):
        next_state1 = np.copy(next_state).flatten()
        w = np.zeros(shape = (len(next_state1),8))
        for a in self.unique_action:
            for m in self.unique_mediator:
                pie_a = self.target_policy(next_state1, a, matrix_based = True)
                pm_aSprime = self.pmlearner(next_state1, a, m)
                X_Sprime_am = self.get_phi(next_state1.flatten(), a, m)
                w += (np.array(X_Sprime_am).T * pie_a * pm_aSprime).T
        return w

    def get_exp_w(self,next_state, Q_func):
        next_state1 = np.copy(next_state).flatten()
        w = np.zeros(shape = (len(next_state1),8))
        for s in self.unique_state:
            p_s = self.simulator.sam2nextstate_model(self.state, self.action, self.mediator,random = False)
            p_s = s * p_s + (1-s) * (1-p_s)
            for a in self.unique_action:
                if Q_func in ['Q1','Q2','Q3']:
                    pie_a = self.target_policy(s*np.ones(len(next_state1)), a, matrix_based = True)
                elif Q_func in ['Q4','Q5']:
                    pie_a = self.control_policy(s*np.ones(len(next_state1)), a, matrix_based = True)
                for m in self.unique_mediator:
                    if Q_func == 'Q4':
                        pm_aSprime = np.zeros(len(next_state1))
                        for a_prime in self.unique_action:
                            pie_aprime = self.target_policy(s*np.ones(len(next_state1)), a_prime, matrix_based = True)
                            pm_aSprime_aprime = self.pmlearner(s*np.ones(len(next_state1)), a_prime, m)
                            pm_aSprime += pm_aSprime_aprime * pie_aprime
                    else:
                        pm_aSprime = self.pmlearner(s*np.ones(len(next_state1)), a, m)
                    X_Sprime_am = self.get_phi(s*np.ones(len(next_state1)), a, m)
                    w += (np.array(X_Sprime_am).T * p_s* pie_a * pm_aSprime).T
        return w
    
    def estimate_beta(self, phi, w, reward):
        NT = reward.shape[0]
        p = phi.shape[1]
        A = np.matmul(phi.T, (phi-w))/NT
        B = np.matmul(phi.T, reward)/NT
        beta_hat = np.matmul(np.linalg.inv(A),B)
        return beta_hat
    
    def fit(self):
        return self.estimate_beta(self.phi, self.w, self.Y)
  