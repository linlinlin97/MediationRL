import numpy as np
from multiprocessing import Pool
from itertools import product
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
    def __init__(self, data, target_policy, control_policy, pmlearner, rewardlearner, ratiolearner, palearner, unique_action, dim_state, dim_mediator,
                 Q_settings = {'scaler': 'Identity', 'product_tensor': True, 'beta': 3/7, 'include_intercept': False,
                               'expectation_MCMC_iter_Q3': 100, 'expectation_MCMC_iter_Q_diff':100, 'penalty': 10**(-9),
                              'd': 3, 'min_L': 7}, seed = 0):
        np.random.seed(seed)
        
        self.unique_action = unique_action
        self.nums_action = len(unique_action)
        self.data = data
        self.dim_state = dim_state
        self.dim_mediator = dim_mediator 
        self.NT = len(self.data['action'])
        self.beta = Q_settings['beta']
        self.product_tensor = Q_settings['product_tensor']
        self.include_intercept = Q_settings['include_intercept']
        self.L = int(np.sqrt((self.NT)**self.beta))
        self.d = Q_settings['d']
        self.min_L = Q_settings['min_L']
        if Q_settings['scaler'] == "NormCdf":
            self.scaler = normcdf()
        elif Q_settings['scaler'] == "Identity":
            self.scaler = iden()
        self.pmlearner = pmlearner
        self.rewardlearner = rewardlearner
        self.ratiolearner = ratiolearner
        self.palearner = palearner
        self.B_spline(L = max(self.min_L,self.L+self.d), d = self.d)
        
        self.expectation_MCMC_iter_Q3 = Q_settings['expectation_MCMC_iter_Q3']
        self.expectation_MCMC_iter_Q_diff = Q_settings['expectation_MCMC_iter_Q_diff']
        self.l2penalty = Q_settings['penalty']
        
        self.state = self.data['state']
        self.mediator = self.data['mediator']
        self.action = self.data['action']
        self.R = self.data['reward']
        self.next_state = self.data['next_state']
        
        self.target_policy = target_policy
        #control_policy
        self.control_policy = control_policy
        self.a0 = control_policy(get_a = True)
        
        self.tuples = self.get_tuples_curr_next(self.data)
        
    def est_Q1(self):
        #Q1
        #t0 = time.time()
        self.eta_pie, self.Q1_diff, self.Q1_est_beta = self.Q_diff_eta(self.tuples, self.R, self.target_policy)
        
    def est_Q2(self):
        #Q2
        Er_Sa0M = self.rewardlearner.get_reward_prediction(self.state, self.a0, self.mediator)
        self.eta_piea0, self.Q2_diff, self.Q2_est_beta = self.Q_diff_eta(self.tuples, Er_Sa0M, self.target_policy)
        
    def est_Q3(self):
        #Q3 
        Er_Sa0m = []
        for rep in range(self.expectation_MCMC_iter_Q3):
            m_Sa0 = self.pmlearner.sample_m(self.state, self.a0, random = True)
            r_Sa0m = self.rewardlearner.get_reward_prediction(self.state, self.a0, m_Sa0)
            Er_Sa0m.append(r_Sa0m)
        Er_Sa0m = np.mean(Er_Sa0m,0)
        self.eta_piea0star, self.Q3_diff, self.Q3_est_beta = self.Q_diff_eta(self.tuples, Er_Sa0m, self.target_policy)
        
    def est_Q4(self):
        #Q4
        self.eta_a0, self.Q4_diff, self.Q4_est_beta = self.Q_diff_eta( self.tuples, self.R, self.control_policy)
        #time_leanr_Q = time.time() - t0
    
    def est_ratio(self):
        #ratio
        #t0 = time.time()
        #with open('best_kde_behavior_type2.txt',"rb") as fp:
        #    stationary_behavior_policy_state_density = pickle.load(fp)
        #with open('best_kde_target_type2.txt',"rb") as fp:
        #    stationary_target_policy_state_density = pickle.load(fp)    
        #with open('best_kde_control_type2.txt',"rb") as fp:
        #    stationary_fixed_policy_state_density = pickle.load(fp)    
     
        self.action = self.data['action']
        pie_A = self.target_policy(self.state, self.dim_state, self.action, matrix_based = True)
        I_A = self.control_policy(self.state, self.dim_state, self.action, matrix_based = True)
        pieb_A = self.palearner.get_pa_prediction(self.state, self.action)
        #numerator_pie = np.exp(stationary_target_policy_state_density.score_samples(self.state.reshape(-1, 1)))
        #numerator_a0 = np.exp(stationary_fixed_policy_state_density.score_samples(self.state.reshape(-1, 1)))
        #denominator = np.exp(stationary_behavior_policy_state_density.score_samples(self.state.reshape(-1, 1)))
        #numerator_pie = np.clip(numerator_pie, a_min = 1e-4, a_max = .9999)
        #numerator_a0 = np.clip(numerator_a0, a_min = 1e-4, a_max = .9999)
        #denominator = np.clip(denominator, a_min = 1e-4, a_max = .9999)
        #w_pie = numerator_pie/denominator
        #w_a0 = numerator_a0/denominator
        w_pie = self.ratiolearner.get_r_prediction(self.state, policy = 'target', normalize=True)
        w_a0 = self.ratiolearner.get_r_prediction(self.state, policy = 'control', normalize=True)
        self.ratio_control = w_a0*I_A/pieb_A
        self.ratio_target = w_pie*pie_A/pieb_A
        #time_learn_ratio = time.time() - t0
        
        self.time_rec = {'time_learn_Q': np.nan, 'time_learn_ratio': np.nan}
        
    def Q_diff_eta(self, tuples, Y_all, policy):
        est_beta = self._beta_hat(tuples, Y_all, policy)
        eta = self.eta_hat(est_beta)
        Q_diff = self.est_Q_diff(self.state, self.mediator, self.action, self.next_state, est_beta, policy)
        return eta, Q_diff, est_beta
        
    def B_spline(self, L = 3, d = 1):
        tuples = np.array(self.get_tuples(self.data))
        state_mediator_col = list(np.arange(self.dim_state)) + list(np.arange(self.dim_state+1,self.dim_state+1+self.dim_mediator)) 
        scale_data = self.scaler.transform(tuples[:,state_mediator_col])
        knot = np.quantile(scale_data, np.linspace(0,1,L + 1), axis=0) # to avoid the bounded issue
        self.bspline = []
        n_const = 1 - int(self.include_intercept)
        n_basis = L - d - n_const
        
        self.para_dim =  [1 if self.product_tensor else 0][0] # last parameter is for eta estimation
        for i in range(self.dim_state + self.dim_mediator):
            tmp = []
            for j in range(n_basis): 
                cof = [0] * (L - d)
                #start from the second column if not include the intercept
                cof[j + n_const] = 1
                spf = BSpline(knot.T[i], cof, d)
                tmp.append(spf)
            self.bspline.append(tmp)
            
            if self.product_tensor:
                self.para_dim *= len(self.bspline[i])
            else:
                self.para_dim += len(self.bspline[i])
            
            #print("Building %d-th basis spline (total %d state-mediator dimemsion) which has %d basis " %(i, self.dim_state + self.dim_mediator,len(self.bspline[i])))
            
    def get_tuples(self, data):
        NT = len(data['state'])
        tuples = []
        for nt in range(NT):
            tuple_t = list(data['state'][nt]) 
            tuple_t += [data['action'][nt]]
            tuple_t += list(data['mediator'][nt])
            tuple_t += [data['reward'][nt]]
            tuple_t += list(data['next_state'][nt])

            tuples.append(tuple_t) #['state', 'action', 'mediator', 'reward', 'next_state']
        return tuples



    def _U(self, S, M, A, include_eta = False):
        x = np.hstack([S, M])
        x = self.scaler.transform(x)
        
        if self.product_tensor:
            output = list(map(np.prod,(product(*[np.array([func(s) for func in f]) for f,s in zip(self.bspline, x)],repeat=1))))
        else:
            output = list(np.concatenate([np.array([func(s) for func in f]) for f,s in zip(self.bspline, x)]))
        if A == 0:
            output += [0] * self.para_dim  + [0] #0 for the intercept
        elif A == 1:
            output = [0] * self.para_dim + output + [1] #1 for the intercept for Q_A1
        else:
            raise ValueError('A is not in [0,1]')

        if include_eta:
            output += [1]
        else:
            output += [0]
        #print('_u:', np.array(output).reshape(-1,1))
        return np.array(output).reshape(-1,1)
    
    def _Xi(self, S_next, policy, include_eta = False):
        phi = np.zeros((self.para_dim*len(self.unique_action)+1,))
        for a in self.unique_action:
            pa = policy(state = S_next, dim_state = self.dim_state, action = a)
            if pa > 0:
                sample_phi = []
                for rep in range(self.expectation_MCMC_iter_Q3):
                    m = self.pmlearner.sample_m(state = S_next, action = np.array([a]), random = True)
                    m = m.reshape((self.dim_mediator,))
                    sample_phi.append(self._U(S_next, m, a)[:-1])
                phi += (np.mean(np.array(sample_phi),0)*pa).reshape(-1,)  
            else:
                phi += np.zeros((self.para_dim*len(self.unique_action)+1,))
            
        if include_eta:
            phi = list(phi) + [1]
        else:
            phi = list(phi) + [0]
        #print('_Xi:', np.array(phi).reshape(-1,1))
        return np.array(phi).reshape(-1,1)

    def _Sigma(self, tuples, Y_all, policy):
        #output = np.zeros((self.para_dim * self.nums_action + 1, self.para_dim * self.nums_action + 1))
        #output_2 = np.zeros((self.para_dim * self.nums_action + 1, 1))
        output = [np.matmul(self._U(tuple_i[0], tuple_i[2], tuple_i[1], include_eta = True),
                                 (self._U(tuple_i[0], tuple_i[2], tuple_i[1], include_eta = True) - 
                                  self._Xi(tuple_i[4], policy, include_eta = False)).T) for tuple_i in tuples]
        output_2 = [Y_all[i] * self._U(tuples[i][0], tuples[i][2], tuples[i][1], include_eta = True) for i in range(len(tuples))]
        output = np.mean(output,axis=0)
        output_2 = np.mean(output_2,axis=0)
        #print('output:', output)
        #print('output_2:', output_2)
        """
        Add ridge can avoid overfitting!
        """
        Sigma_hat =  np.diag([self.l2penalty] * output.shape[0])  +  output
        vector = output_2
        self.tuples = tuples
        self.Y_all = Y_all
        self.outpolicy=  policy
        return Sigma_hat, vector

    def _beta_hat(self, tuples, Y_all, policy):
        Sigma_hat, vector = self._Sigma(tuples, Y_all, policy)
        #print(Sigma_hat)
        inv_Sigma_hat = inv(Sigma_hat)
        est_beta = np.matmul(inv_Sigma_hat, vector)
        return est_beta


    def eta_hat(self, est_beta):
        return est_beta[-1]

    def Q(self, S, M, A, est_beta):
        output = self._U(S, M, A)[:-1]
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
            pie_a = policy(next_state, self.dim_state, a, matrix_based = True)
            Q_Snext_am_MC = []
            for rep in range(self.expectation_MCMC_iter_Q_diff):
                m_Snext_a = self.pmlearner.sample_m(next_state, np.array([a]), random = True)
                out_Q = [self.Q(next_state[i], m_Snext_a[i], a, est_beta) for i in range(len(action))]
                Q_Snext_am_MC.append(out_Q)
            Q_Snext_am += pie_a * np.mean(Q_Snext_am_MC,0).reshape((-1,))
        return Q_Snext_am - np.mean(Q_SAm,0).reshape((-1,))

    def get_tuples_curr_next(self, data):
        tuples = [[data['state'][nt], data['action'][nt], data['mediator'][nt], data['reward'][nt], data['next_state'][nt]] 
                  for nt in range(self.NT)]
        return tuples