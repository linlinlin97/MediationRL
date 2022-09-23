import numpy as np
from multiprocessing import Pool
from tqdm import tqdm
import pickle
from scipy.special import expit
import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd  
import sklearn

from sklearn.tree import DecisionTreeRegressor  
from sklearn.model_selection import cross_val_score  
from sklearn.metrics import mean_squared_error

from scipy.interpolate import BSpline
from scipy.stats import norm

from numpy.linalg import inv
import time

class normcdf():
    def transform(self, S):
        return norm.cdf(S)

class iden():
    def transform(self, S):
        return S
    
class Qlearner():
    def __init__(self, data, target_policy, control_policy, pmlearner, rewardlearner, ratiolearner, palearner, unique_action, dim_state, dim_mediator, scaler = 'Identity', expectation_MCMC_iter_Q3 = 100, expectation_MCMC_iter_Q_diff = 100, seed = 0 ):
        np.random.seed(seed)
        self.unique_action = unique_action
        self.nums_action = len(unique_action)
        self.data = data
        self.dim_state = dim_state
        self.dim_mediator = dim_mediator
        self.NT = len(self.data['action'])
        self.beta = 3/7
        self.L = int(np.sqrt((self.NT)**self.beta))
        if scaler == "NormCdf":
            self.scaler = normcdf()
        elif scaler == "Identity":
            self.scaler = iden()
        self.pmlearner = pmlearner
        self.rewardlearner = rewardlearner
        self.B_spline(L = max(7,self.L+3), d = 3)
        
        self.expectation_MCMC_iter_Q3 = expectation_MCMC_iter_Q3
        self.expectation_MCMC_iter_Q_diff = expectation_MCMC_iter_Q_diff
        
        self.state = self.data['state']
        self.mediator = self.data['mediator']
        self.action = self.data['action']
        self.R = self.data['reward']
        self.next_state = self.data['next_state']
        
        self.target_policy = target_policy
        #control_policy
        self.control_policy = control_policy
        self.a0 = control_policy(get_a = True)
        
        self.tuples_target = self.get_tuples_curr_next(self.data, self.target_policy)
        self.tuples_control = self.get_tuples_curr_next(self.data, self.control_policy)
        
        #Q1
        t0 = time.time()
        self.eta_pie, self.Q1_diff, self.Q1_est_beta = self.Q_diff_eta(self.tuples_target, self.R, self.target_policy)
        #Q2
        Er_Sa0M = self.rewardlearner.get_reward_prediction(self.state, self.a0, self.mediator)
        self.eta_piea0, self.Q2_diff, self.Q2_est_beta = self.Q_diff_eta(self.tuples_target, Er_Sa0M, self.target_policy)
        #Q3 
        Er_Sa0m = []
        for rep in range(self.expectation_MCMC_iter_Q3):
            m_Sa0 = self.pmlearner.sample_m(self.state, self.a0, random = True)
            r_Sa0m = self.rewardlearner.get_reward_prediction(self.state, self.a0, m_Sa0)
            Er_Sa0m.append(r_Sa0m)
        Er_Sa0m = np.mean(Er_Sa0m,0)
        self.eta_piea0star, self.Q3_diff, self.Q3_est_beta = self.Q_diff_eta(self.tuples_target, Er_Sa0m, self.target_policy)
        #Q4
        self.eta_a0, self.Q4_diff, self.Q4_est_beta = self.Q_diff_eta( self.tuples_control, self.R, self.control_policy)
        time_leanr_Q = time.time() - t0
        
        #ratio
        t0 = time.time()
        #with open('best_kde_behavior_type2.txt',"rb") as fp:
        #    stationary_behavior_policy_state_density = pickle.load(fp)
        #with open('best_kde_target_type2.txt',"rb") as fp:
        #    stationary_target_policy_state_density = pickle.load(fp)    
        #with open('best_kde_control_type2.txt',"rb") as fp:
        #    stationary_fixed_policy_state_density = pickle.load(fp)    
     
        self.action = self.data['action']
        pie_A = target_policy(self.state, self.action, matrix_based = True)
        I_A = control_policy(self.state, self.action, matrix_based = True)
        pieb_A = palearner.get_pa_prediction(self.state, self.action)
        #numerator_pie = np.exp(stationary_target_policy_state_density.score_samples(self.state.reshape(-1, 1)))
        #numerator_a0 = np.exp(stationary_fixed_policy_state_density.score_samples(self.state.reshape(-1, 1)))
        #denominator = np.exp(stationary_behavior_policy_state_density.score_samples(self.state.reshape(-1, 1)))
        #numerator_pie = np.clip(numerator_pie, a_min = 1e-4, a_max = .9999)
        #numerator_a0 = np.clip(numerator_a0, a_min = 1e-4, a_max = .9999)
        #denominator = np.clip(denominator, a_min = 1e-4, a_max = .9999)
        #w_pie = numerator_pie/denominator
        #w_a0 = numerator_a0/denominator
        w_pie = ratiolearner.get_r_prediction(self.state, policy = 'target', normalize=True)
        w_a0 = ratiolearner.get_r_prediction(self.state, policy = 'control', normalize=True)
        self.ratio_control = w_a0*I_A/pieb_A
        self.ratio_target = w_pie*pie_A/pieb_A
        time_learn_ratio = time.time() - t0
        
        self.time_rec = {'time_learn_Q': time_leanr_Q, 'time_learn_ratio': time_learn_ratio}
        
    def Q_diff_eta(self, tuples, Y_all, policy):
        est_beta = self._beta_hat(tuples, Y_all)
        eta = self.eta_hat(est_beta)
        Q_diff = self.est_Q_diff(self.state, self.mediator, self.action, self.next_state, est_beta, policy)
        return eta, Q_diff, est_beta
        
    def B_spline(self, L = 3, d = 1):
        tuples = np.array(self.get_tuples(self.data))
        scale_data = self.scaler.transform(tuples[:,[0,2]])
        knot = np.quantile(scale_data, np.linspace(0,1,L + 1), axis=0)
        self.bspline = []
        self.para_dim = 0 # last parameter is for eta estimation
        for i in range(self.dim_state + self.dim_mediator):
            tmp = []
            for j in range(1, L - d): #start from the second column to ignore the intercept
                cof = [0] * (L - d)
                cof[j] = 1
                spf = BSpline(knot.T[i], cof, d)
                tmp.append(spf)
            self.bspline.append(tmp)
            self.para_dim += len(self.bspline[i])
            
    def get_tuples(self, data):
        NT = len(data['state'])
        tuples = []
        for nt in range(NT):
            tuple_t = list(data['state'][nt]) + [data['action'][nt], data['mediator'][nt], data['reward'][nt]] + list(data['next_state'][nt])

            tuples.append(tuple_t) #['state', 'action', 'mediator', 'reward', 'next_state']
        return tuples



    def _Xi(self, S, M, A, include_eta = False):
        x = np.hstack([S, M])
        x = self.scaler.transform(x)
        if A == 0:
            output = list(np.concatenate([np.array([func(s) for func in f]) for f,s in zip(self.bspline, x)])) + [0] * self.para_dim  + [0] #0 for the intercept
        elif A == 1:
            output = [0] * self.para_dim + list(np.concatenate([np.array([func(s) for func in f]) for f,s in zip(self.bspline, x)])) + [1] #1 for the intercept for Q_A1
        else:
            raise ValueError('A is not in [0,1]')

        if include_eta:
            output += [1]
        else:
            output += [0]
        return np.array(output).reshape(-1,1)

    def _Sigma(self, tuples, Y_all):
        #output = np.zeros((self.para_dim * self.nums_action + 1, self.para_dim * self.nums_action + 1))
        #output_2 = np.zeros((self.para_dim * self.nums_action + 1, 1))
        output = [np.matmul(self._Xi(tuple_i[0], tuple_i[2], tuple_i[1], include_eta = True),
                                 (self._Xi(tuple_i[0], tuple_i[2], tuple_i[1], include_eta = True) - 
                                  self._Xi(tuple_i[4], tuple_i[6], tuple_i[5], include_eta = False)).T) for tuple_i in tuples]
        output_2 = [Y_all[i] * self._Xi(tuples[i][0], tuples[i][2], tuples[i][1], include_eta = True) for i in range(len(tuples))]
        output = np.sum(output,axis=0)
        output_2 = np.sum(output_2,axis=0)
        """
        Add ridge can avoid overfitting!
        """
        Sigma_hat =  np.diag([10 ** (-9)] * output.shape[0])  +  output / self.NT
        vector = output_2 / self.NT
        return Sigma_hat, vector

    def _beta_hat(self, tuples, Y_all):
        Sigma_hat, vector = self._Sigma(tuples, Y_all)
        inv_Sigma_hat = inv(Sigma_hat)
        est_beta = np.matmul(inv_Sigma_hat, vector)
        return est_beta


    def eta_hat(self, est_beta):
        return est_beta[-1]

    def Q(self, S, M, A, est_beta):
        output = self._Xi(S, M, A)[:-1]
        return np.dot(output.reshape((1,-1)), est_beta[:-1].reshape((-1,)))
        #return sum(map(operator.mul, output, est_beta[:-1]))

    def est_Q_diff(self, state, mediator, action, next_state, est_beta, policy):
        Q_Snext_am = np.zeros(len(action), dtype=float)
        #MCMC to get the mean over m
        Q_SAm = []
        for rep in range(self.expectation_MCMC_iter_Q_diff):
            m_SA = self.pmlearner.sample_m(state, action, random = True)
            out_Q = [self.Q(state[i], m_SA[i], action[i], est_beta) for i in range(len(action))]
            Q_SAm.append(out_Q)
        for a in self.unique_action:
            pie_a = policy(next_state, a, matrix_based = True)
            Q_Snext_am_MC = []
            for rep in range(self.expectation_MCMC_iter_Q_diff):
                m_Snext_a = self.pmlearner.sample_m(next_state, np.array([a]), random = True)
                out_Q = [self.Q(next_state[i], m_Snext_a[i], a, est_beta) for i in range(len(action))]
                Q_Snext_am_MC.append(out_Q)
            Q_Snext_am += pie_a * np.mean(Q_Snext_am_MC,0).reshape((-1,))
        return Q_Snext_am - np.mean(Q_SAm,0).reshape((-1,))

    def get_tuples_curr_next(self, data, policy):
        NT = len(data['state'])
        tuples = []
        for nt in range(NT):
            tuple_t = [data['state'][nt], data['action'][nt], data['mediator'][nt], data['reward'][nt], data['next_state'][nt]]
            s_t1 = data['next_state'][nt]
            policy_a_t1 = policy(state = s_t1, action = None)
            m_t1 =  self.pmlearner.sample_m(state = s_t1, action = policy_a_t1, random = True)

            tuple_t += [policy_a_t1, m_t1]
            tuples.append(tuple_t) #['state', 'action', 'mediator', 'reward', 'next_state', 'sampled_a_t+1', 'sampled_m_t+1' ]
        return tuples