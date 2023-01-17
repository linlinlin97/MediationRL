import numpy as np
from itertools import product  
from scipy.interpolate import BSpline
from scipy.stats import norm
from numpy.linalg import inv
from _util import *

class Qlearner():
    def __init__(self, data, target_policy, control_policy, pmlearner, rewardlearner, dim_state, dim_mediator,
                 Q_settings = {'scaler': 'Identity', 'product_tensor': True, 'beta': 3/7, 'include_intercept': False,
                               'expectation_MCMC_iter_Q3': 100, 'expectation_MCMC_iter_Q_diff':100, 'penalty': 10**(-9),
                              'd': 3, 'min_L': 7, "t_dependent_Q": False}, 
                 seed = 0, t_depend_target = False):
        np.random.seed(seed)
        
        self.state = data['state']
        self.mediator = data['mediator']
        self.action = data['action']
        self.R = data['reward']
        self.next_state = data['next_state']
        self.time_idx = data['time_idx']
        self.unique_action = np.unique(self.action)
        self.tuples = self.get_tuples(data)
        
        self.dim_state = dim_state
        self.dim_mediator = dim_mediator 
        self.NT = len(self.action)
        self.beta = Q_settings['beta']
        self.product_tensor = Q_settings['product_tensor']
        self.include_intercept = Q_settings['include_intercept']
        self.L = int(np.sqrt((self.NT)**self.beta))
        self.d = Q_settings['d']
        self.min_L = Q_settings['min_L']
        
        self.tuples_array = np.array(self.tuples)
        self.state_mediator_col = list(np.arange(self.dim_state)) + list(np.arange(self.dim_state+1,self.dim_state+1+self.dim_mediator)) 
        
        self.scaler_setting = Q_settings['scaler'] 
        if self.scaler_setting == "NormCdf":
            self.scaler = normcdf()
            self.sm_mean = np.mean(self.tuples_array[:,self.state_mediator_col],axis = 0)
            self.sm_std = np.std(self.tuples_array[:,self.state_mediator_col],axis = 0)
        elif self.scaler_setting == "Identity":
            self.scaler = iden()
        elif self.scaler_setting == "Standardize":
            self.scaler = iden()
            self.sm_mean = np.mean(self.tuples_array[:,self.state_mediator_col],axis = 0)
            self.sm_std = np.std(self.tuples_array[:,self.state_mediator_col],axis = 0)
            self.t_mean = np.mean(self.tuples_array[:,-1],axis = 0)
            self.t_std = np.std(self.tuples_array[:,-1],axis = 0)
            
        self.pmlearner = pmlearner
        self.rewardlearner = rewardlearner
        self.t_dependent_Q = Q_settings['t_dependent_Q']
        self.B_spline(L = max(self.min_L,self.L+self.d), d = self.d)
        
        self.expectation_MCMC_iter_Q3 = Q_settings['expectation_MCMC_iter_Q3']
        self.expectation_MCMC_iter_Q_diff = Q_settings['expectation_MCMC_iter_Q_diff']
        self.l2penalty = Q_settings['penalty']
        
        self.target_policy = target_policy
        #control_policy
        self.control_policy = control_policy
        self.a0 = control_policy(get_a = True)
        self.t_depend_target = t_depend_target
        
        self.U = self._U4all(self.tuples)
        self.Sigma_control = self._Sigma(self.tuples, 'control', self.U)
        self.Sigma_target = self._Sigma(self.tuples, 'target', self.U)
        self.Sigma_G = self._Sigma(self.tuples, 'G', self.U)
        
        self.Er_Sa0m = self._Er_Sa0m()
        
    def est_Q1(self):
        #Q1
        #t0 = time.time()
        self.eta_pie, self.Q1_est_beta = self.est_beta_eta(self.R, self.U, self.Sigma_target)
        
    def est_Q2(self):
        #Q2
        Er_Sa0M = self.rewardlearner.get_reward_prediction(self.state, self.a0, self.mediator)
        self.eta_piea0, self.Q2_est_beta = self.est_beta_eta(Er_Sa0M, self.U, self.Sigma_target)
        
    def est_Q3(self):
        #Q3 
        self.eta_piea0star, self.Q3_est_beta = self.est_beta_eta(self.Er_Sa0m, self.U, self.Sigma_target)
        
    def est_Q4(self):
        #Q4
        self.eta_G, self.Q4_est_beta = self.est_beta_eta(self.Er_Sa0m, self.U, self.Sigma_G)        
        
    def est_Q5(self):
        #Q5
        self.eta_a0, self.Q5_est_beta = self.est_beta_eta(self.R, self.U, self.Sigma_control)
    
    def est_Qdiffs(self):
        self.Q1_diff, self.Q2_diff, self.Q3_diff, self.Q4_diff_1, self.Q4_diff_2, self.Q5_diff = self._Q_diff(self.state, self.mediator, self.action, self.next_state)
        
    def _Er_Sa0m(self):
        Er_Sa0m = np.zeros(self.NT, dtype=float)
        for rep in range(self.expectation_MCMC_iter_Q3):
            #np.random.seed(rep)
            m_Sa0 = self.pmlearner.sample_m(self.state, self.a0, random = True)
            r_Sa0m = self.rewardlearner.get_reward_prediction(self.state, self.a0, m_Sa0)
            Er_Sa0m = self.update_exp(rep, Er_Sa0m, r_Sa0m.reshape((-1,)))
        return Er_Sa0m
            
    def est_beta_eta(self, Y_all, U, Sigma_hat):
        est_beta = self._beta_hat(Y_all, U, Sigma_hat)
        eta = self.eta_hat(est_beta)
        return eta, est_beta
        
    def B_spline(self, L = 3, d = 1):
        if self.scaler_setting == "NormCdf":
            knot = np.array([np.linspace(0,1,L + 1),np.linspace(0,1,L + 1)]).T # to avoid the bounded issue
        elif self.scaler_setting == "Identity":
            scale_data = self.scaler.transform(self.tuples_array[:,self.state_mediator_col])
            knot = np.quantile(scale_data, np.linspace(0,1,L + 1), axis=0) # to avoid the bounded issue
        elif self.scaler_setting == "Standardize":
            scale_data = self.scaler.transform((self.tuples_array[:,self.state_mediator_col]-self.sm_mean)/self.sm_std)
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
            
            print("Building %d-th basis spline (total %d state-mediator dimemsion) which has %d basis, in total %d features " %(i, self.dim_state + self.dim_mediator,len(self.bspline[i]),self.para_dim))
            
    def get_tuples(self, data):
        NT = len(data['state'])
        tuples = []
        for nt in range(NT):
            tuple_t = list(data['state'][nt]) 
            tuple_t += [data['action'][nt]]
            tuple_t += list(data['mediator'][nt])
            tuple_t += [data['reward'][nt]]
            tuple_t += list(data['next_state'][nt])
            tuple_t += [data['time_idx'][nt]]

            tuples.append(tuple_t) #['state', 'action', 'mediator', 'reward', 'next_state', 'time_idx']
        return tuples


    def _U(self, S, M, A, include_eta = False, time_idx = None):
        x = np.hstack([S, M])
        if self.scaler_setting == "NormCdf" or self.scaler_setting == "Standardize":
            x = (x-self.sm_mean)/self.sm_std
            if time_idx:
                time_idx = (time_idx - self.t_mean)/self.t_std
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
        
        if self.t_dependent_Q:
            output += [time_idx]

        if include_eta:
            output += [1]
        else:
            output += [0]
        return np.array(output).reshape(-1,1) #[sm_0,sm_1,intercept, t, eta]
    
    def _Xi(self, S_next, t, policy_name, include_eta = False):
        phi = np.zeros((self.para_dim*len(self.unique_action)+1,))
        for a in self.unique_action:
            if policy_name == 'target':
                pa = self.target_pa_next(S_next, a, t)
            elif policy_name == 'control' or policy_name == 'G':
                pa = self.control_policy(state = S_next, dim_state = self.dim_state, action = a)
                
            if pa > 0:
                if policy_name == 'G':
                    sample_phi = np.zeros((self.para_dim*len(self.unique_action)+1,))
                    for a_star in self.unique_action:
                        pa_star = self.target_pa_next(S_next, a_star, t)
                        sample_phi += pa_star * self.MCMC_Em_phi(S_next, a_star, a, t) 
                else:
                    sample_phi = self.MCMC_Em_phi(S_next, a, a, t)
                    
                phi += (np.array(sample_phi)*pa).reshape(-1,)  
                
            else:
                phi += np.zeros((self.para_dim*len(self.unique_action)+1,))
        
        if self.t_dependent_Q:
            if self.scaler_setting == "NormCdf" or self.scaler_setting == "Standardize":
                phi = list(phi) + [(t + 1 - self.t_mean)/self.t_std]
            else:
                phi = list(phi) + [t+1]
            
        if include_eta:
            phi = list(phi) + [1]
        else:
            phi = list(phi) + [0]
        return np.array(phi).reshape(-1,1)
    
    def MCMC_Em_phi(self, S_next, a1, a2, t):
        sample_phi = np.zeros((self.para_dim*len(self.unique_action)+1,))
        for rep in range(self.expectation_MCMC_iter_Q3):
            #np.random.seed(rep)
            m = self.pmlearner.sample_m(state = S_next, action = np.array([a1]), random = True)
            m = m.reshape((self.dim_mediator,))
            if self.t_dependent_Q:
                sample_phi = self.update_exp(rep, sample_phi, self._U(S_next, m, a2, time_idx = t+1)[:-2].reshape(-1,))
            else:
                sample_phi = self.update_exp(rep, sample_phi, self._U(S_next, m, a2)[:-1].reshape(-1,))
        return sample_phi
        
    def target_pa_next(self, S_next, a, t):
        if self.t_depend_target:
            pa = self.target_policy(state = S_next, dim_state = self.dim_state, action = a, time_idx = t + 1)
        else:
            pa = self.target_policy(state = S_next, dim_state = self.dim_state, action = a)
        return pa
    
    def _U4all(self, tuples):
        if self.t_dependent_Q:
            U = [self._U(tuple_i[:self.dim_state], tuple_i[self.dim_state+1:self.dim_state+self.dim_mediator+1],
                                    tuple_i[self.dim_state], include_eta = True, time_idx = tuple_i[-1]) for tuple_i in tuples]
        else:
            U = [self._U(tuple_i[:self.dim_state], tuple_i[self.dim_state+1:self.dim_state+self.dim_mediator+1],
                                    tuple_i[self.dim_state], include_eta = True) for tuple_i in tuples]
        return U
    
    def _Sigma(self, tuples, policy_name, U):
        output = [np.matmul(U[i],
                            (U[i] - self._Xi(np.array(tuples[i][-(self.dim_state+1):-1]), tuples[i][-1], policy_name, include_eta = False)).T) for i in range(len(tuples))]
        output = np.mean(output,axis=0)
        """
        Add ridge can avoid overfitting!
        """
        Sigma_hat =  np.diag([self.l2penalty] * output.shape[0])  +  output
        return Sigma_hat
    
    def _UY(self, Y_all, U):
        output_2 = [Y_all[i] * U[i] for i in range(len(U))]
        output_2 = np.mean(output_2,axis=0)
        vector = output_2
        return vector

    def _beta_hat(self, Y_all, U, Sigma_hat):
        vector = self._UY(Y_all, U)
        #print(Sigma_hat)
        inv_Sigma_hat = inv(Sigma_hat)
        est_beta = np.matmul(inv_Sigma_hat, vector)
        return est_beta


    def eta_hat(self, est_beta):
        return est_beta[-1]

    def Q_1235(self, S, M, A, t):
        output = self._U(S, M, A, time_idx = t)[:-1]
        Q1 = np.dot(output.reshape((1,-1)), self.Q1_est_beta[:-1].reshape((-1,)))
        Q2 = np.dot(output.reshape((1,-1)), self.Q2_est_beta[:-1].reshape((-1,)))
        Q3 = np.dot(output.reshape((1,-1)), self.Q3_est_beta[:-1].reshape((-1,)))
        Q5 = np.dot(output.reshape((1,-1)), self.Q5_est_beta[:-1].reshape((-1,)))
        return Q1, Q2, Q3, Q5
        #return sum(map(operator.mul, output, est_beta[:-1]))

    def _Q_diff(self, state, mediator, action, next_state):
        #MCMC to get the mean over m
        Q1_SAm, Q2_SAm, Q3_SAm, Q5_SAm = self.init_Qs()
        for rep in range(self.expectation_MCMC_iter_Q_diff):
            #np.random.seed(rep)
            m_SA = self.pmlearner.sample_m(state, action, random = True)
            out_Q1,out_Q2,out_Q3,out_Q5 = self.cal_newQ_1235(state, m_SA, action, time_idx = self.time_idx)
            Q1_SAm = self.update_exp(rep, Q1_SAm, out_Q1)
            Q2_SAm = self.update_exp(rep, Q2_SAm, out_Q2)
            Q3_SAm = self.update_exp(rep, Q3_SAm, out_Q3)
            Q5_SAm = self.update_exp(rep, Q5_SAm, out_Q5)
        
        Q1_Snext_am, Q2_Snext_am, Q3_Snext_am, Q5_Snext_am = self.init_Qs()
        Q4_Snext_am = np.zeros(self.NT, dtype=float)
        Q4_S_am = np.zeros(self.NT, dtype=float)
        Q4_SaM = np.zeros(self.NT, dtype=float)
        for a in self.unique_action:
            pie_a_Sprime = self.target_pa_next(next_state, a, self.time_idx)          
            pi0_a_Sprime = self.control_policy(next_state, self.dim_state, a)
            pi0_a_S = self.control_policy(state, self.dim_state, a)
            
            Q1_Snext_am_MC, Q2_Snext_am_MC, Q3_Snext_am_MC, Q4_Snext_am_MC, Q5_Snext_am_MC, Q4_S_am_MC = self.cal_Q_am_MC(next_state, state, action, a)
                    
            Q1_Snext_am += pie_a_Sprime * Q1_Snext_am_MC.reshape((-1,))
            Q2_Snext_am += pie_a_Sprime * Q2_Snext_am_MC.reshape((-1,))
            Q3_Snext_am += pie_a_Sprime * Q3_Snext_am_MC.reshape((-1,))
            Q4_Snext_am += pi0_a_Sprime * Q4_Snext_am_MC.reshape((-1,))
            Q5_Snext_am += pi0_a_Sprime * Q5_Snext_am_MC.reshape((-1,))
            
            Q4_S_am += pi0_a_S * Q4_S_am_MC.reshape((-1,))
            
            action_list = [a]*self.NT
            Q4_SaM += pi0_a_S * self.cal_newQ_4(state, mediator, action_list, time_idx = self.time_idx)
            
        Q4_SAM = self.cal_newQ_4(state, mediator, action, self.time_idx)
        
        Q1_diff = Q1_Snext_am - Q1_SAm.reshape((-1,))
        Q2_diff = Q2_Snext_am - Q2_SAm.reshape((-1,))
        Q3_diff = Q3_Snext_am - Q3_SAm.reshape((-1,))
        Q5_diff = Q5_Snext_am - Q5_SAm.reshape((-1,))
        Q4_diff_1 = Q4_Snext_am -  Q4_SAM.reshape((-1,))
        Q4_diff_2 = Q4_SaM - Q4_S_am
        return Q1_diff, Q2_diff, Q3_diff, Q4_diff_1, Q4_diff_2, Q5_diff
    
    
    def init_Qs(self):
        return np.zeros(self.NT, dtype=float),np.zeros(self.NT, dtype=float),np.zeros(self.NT, dtype=float),np.zeros(self.NT, dtype=float)
    
    def update_exp(self, rep, old_est, new_obs):
        return (rep*old_est + new_obs)/(rep+1)
    
    def cal_newQ_1235(self, s,m,a, time_idx):
        Qs = [self.Q_1235(s[i], m[i], a[i], time_idx[i]) for i in range(self.NT)]
        Q1 = np.array([q[0] for q in Qs]).reshape((-1,))
        Q2 = np.array([q[1] for q in Qs]).reshape((-1,))
        Q3 = np.array([q[2] for q in Qs]).reshape((-1,))
        Q5 = np.array([q[3] for q in Qs]).reshape((-1,))
        return Q1, Q2, Q3, Q5
    
    def Q_4(self, S, M, A, t):
        output = self._U(S, M, A, time_idx = t)[:-1]
        Q4 = np.dot(output.reshape((1,-1)), self.Q4_est_beta[:-1].reshape((-1,)))
        return Q4
    
    def cal_newQ_4(self, s,m,a, time_idx):
        Qs = [self.Q_4(s[i], m[i], a[i], time_idx[i]) for i in range(self.NT)]
        Q4 = np.array(Qs).reshape((-1,))
        return Q4
    
    def cal_Q_am_MC(self,next_state, state, action, a):
        Q1_Snext_am_MC, Q2_Snext_am_MC, Q3_Snext_am_MC, Q5_Snext_am_MC = self.init_Qs()
        Q4_Snext_am_MC_astar = dict()
        Q4_S_am_MC= np.zeros(self.NT, dtype=float)
        pie_a_star = dict()
        for a_star in self.unique_action:
            Q4_Snext_am_MC_astar[a_star] = np.zeros(self.NT, dtype=float)
            pie_a_star[a_star] = self.target_pa_next(next_state, a_star, self.time_idx)
            
        for rep in range(self.expectation_MCMC_iter_Q_diff):
            #np.random.seed(rep)
            Q1_Snext_am_MC, Q2_Snext_am_MC, Q3_Snext_am_MC, Q5_Snext_am_MC = self.update_Q1235_Snext_am_MC(rep, next_state, a, Q1_Snext_am_MC, Q2_Snext_am_MC, Q3_Snext_am_MC, Q5_Snext_am_MC)
            
            Q4_Snext_am_MC_astar = self.update_Q4_Snext_am_MC_astar(rep, next_state, a, Q4_Snext_am_MC_astar)
            
            Q4_S_am_MC = self.update_Q4_S_am_MC(rep, state, action, a, Q4_S_am_MC)
                
        Q4_Snext_am_MC = np.zeros(self.NT, dtype=float)
        for a_star in self.unique_action:
            Q4_Snext_am_MC += pie_a_star[a_star] * Q4_Snext_am_MC_astar[a_star]
            
        return Q1_Snext_am_MC, Q2_Snext_am_MC, Q3_Snext_am_MC, Q4_Snext_am_MC, Q5_Snext_am_MC, Q4_S_am_MC
    
    def update_Q1235_Snext_am_MC(self, rep, next_state, a, Q1_Snext_am_MC, Q2_Snext_am_MC, Q3_Snext_am_MC, Q5_Snext_am_MC):
        m_Snext_a = self.pmlearner.sample_m(next_state, np.array([a]), random = True)
        action_list = [a]*self.NT
        out_Q1, out_Q2, out_Q3, out_Q5 = self.cal_newQ_1235(next_state, m_Snext_a, action_list, time_idx = self.time_idx + 1)
        Q1_Snext_am_MC = self.update_exp(rep, Q1_Snext_am_MC, out_Q1)
        Q2_Snext_am_MC = self.update_exp(rep, Q2_Snext_am_MC, out_Q2)
        Q3_Snext_am_MC = self.update_exp(rep, Q3_Snext_am_MC, out_Q3)
        Q5_Snext_am_MC = self.update_exp(rep, Q5_Snext_am_MC, out_Q5)
        return Q1_Snext_am_MC, Q2_Snext_am_MC, Q3_Snext_am_MC, Q5_Snext_am_MC
    
    def update_Q4_Snext_am_MC_astar(self, rep, next_state, a, Q4_Snext_am_MC_astar):
        for a_star in self.unique_action:
            m_Snext_a_star = self.pmlearner.sample_m(next_state, np.array([a_star]), random = True)
            action_list = [a]*self.NT
            out_Q4_a_star = self.cal_newQ_4(next_state, m_Snext_a_star, action_list, time_idx = self.time_idx + 1)
            Q4_Snext_am_MC_astar[a_star] = self.update_exp(rep, Q4_Snext_am_MC_astar[a_star], out_Q4_a_star)
        return Q4_Snext_am_MC_astar
    
    def update_Q4_S_am_MC(self, rep, state, action, a, Q4_S_am_MC):
        m_S_A = self.pmlearner.sample_m(state, action, random = True)
        action_list = [a]*self.NT
        out_Q4 = self.cal_newQ_4(state, m_S_A, action_list, time_idx = self.time_idx)
        Q4_S_am_MC = self.update_exp(rep, Q4_S_am_MC, out_Q4)
        return Q4_S_am_MC
        
