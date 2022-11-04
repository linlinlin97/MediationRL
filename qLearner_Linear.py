import numpy as np
from itertools import product  
from scipy.interpolate import BSpline
from scipy.stats import norm
from numpy.linalg import inv

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
        self.state = data['state']
        self.mediator = data['mediator']
        self.action = data['action']
        self.R = data['reward']
        self.next_state = data['next_state']
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
        
        self.target_policy = target_policy
        #control_policy
        self.control_policy = control_policy
        self.a0 = control_policy(get_a = True)
        
        self.U = self._U4all(self.tuples)
        self.Sigma_control = self._Sigma(self.tuples, control_policy, self.U)
        self.Sigma_target = self._Sigma(self.tuples, target_policy, self.U)
        
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
        Er_Sa0m = np.zeros(self.NT, dtype=float)
        for rep in range(self.expectation_MCMC_iter_Q3):
            #np.random.seed(rep)
            m_Sa0 = self.pmlearner.sample_m(self.state, self.a0, random = True)
            r_Sa0m = self.rewardlearner.get_reward_prediction(self.state, self.a0, m_Sa0)
            Er_Sa0m = self.update_exp(rep, Er_Sa0m, r_Sa0m.reshape((-1,)))
        self.eta_piea0star, self.Q3_est_beta = self.est_beta_eta(Er_Sa0m, self.U, self.Sigma_target)
        
    def est_Q4(self):
        #Q4
        self.eta_a0, self.Q4_est_beta = self.est_beta_eta(self.R, self.U, self.Sigma_control)
        #time_leanr_Q = time.time() - t0
    
    def est_Qdiffs(self):
        self.Q1_diff, self.Q2_diff, self.Q3_diff, self.Q4_diff = self._Q_diff(self.state, self.mediator, self.action, self.next_state)
        
    def est_ratio(self):    
        pie_A = self.target_policy(self.state, self.dim_state, self.action, matrix_based = True)
        I_A = self.control_policy(self.state, self.dim_state, self.action, matrix_based = True)
        pieb_A = self.palearner.get_pa_prediction(self.state, self.action)
        w_pie = self.ratiolearner.get_r_prediction(self.state, policy = 'target', normalize=True)
        w_a0 = self.ratiolearner.get_r_prediction(self.state, policy = 'control', normalize=True)
        self.ratio_control = w_a0*I_A/pieb_A
        self.ratio_target = w_pie*pie_A/pieb_A
        #time_learn_ratio = time.time() - t0
        
        self.time_rec = {'time_learn_Q': np.nan, 'time_learn_ratio': np.nan}
        
    def est_beta_eta(self, Y_all, U, Sigma_hat):
        est_beta = self._beta_hat(Y_all, U, Sigma_hat)
        eta = self.eta_hat(est_beta)
        #Q_diff = self.est_Q_diff(self.state, self.mediator, self.action, self.next_state, est_beta, policy)
        return eta, est_beta
        
    def B_spline(self, L = 3, d = 1):
        tuples_array = np.array(self.tuples)
        state_mediator_col = list(np.arange(self.dim_state)) + list(np.arange(self.dim_state+1,self.dim_state+1+self.dim_mediator)) 
        scale_data = self.scaler.transform(tuples_array[:,state_mediator_col])
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
                    #np.random.seed(rep)
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
    
    def _U4all(self, tuples):
        U = [self._U(tuple_i[:self.dim_state], tuple_i[self.dim_state+1:self.dim_state+self.dim_mediator+1],
                                    tuple_i[self.dim_state], include_eta = True) for tuple_i in tuples]
        return U
    
    def _Sigma(self, tuples, policy, U):
        output = [np.matmul(U[i],
                            (U[i] - self._Xi(np.array(tuples[i][-self.dim_state:]), policy, include_eta = False)).T) for i in range(len(tuples))]
        output = np.mean(output,axis=0)
        #output = [np.matmul(self._U(tuple_i[0], tuple_i[2], tuple_i[1], include_eta = True),
        #                         (self._U(tuple_i[0], tuple_i[2], tuple_i[1], include_eta = True) - 
        #                          self._Xi(tuple_i[4], policy, include_eta = False)).T) for tuple_i in tuples]
        """
        Add ridge can avoid overfitting!
        """
        Sigma_hat =  np.diag([self.l2penalty] * output.shape[0])  +  output
        return Sigma_hat
    
    def _UY(self, Y_all, U):
        output_2 = [Y_all[i] * U[i] for i in range(len(U))]
        output_2 = np.mean(output_2,axis=0)
        vector = output_2
        #output_2 = [Y_all[i] * self._U(tuples[i][0], tuples[i][2], tuples[i][1], include_eta = True) for i in range(len(tuples))]
        return vector

    def _beta_hat(self, Y_all, U, Sigma_hat):
        vector = self._UY(Y_all, U)
        #print(Sigma_hat)
        inv_Sigma_hat = inv(Sigma_hat)
        est_beta = np.matmul(inv_Sigma_hat, vector)
        return est_beta


    def eta_hat(self, est_beta):
        return est_beta[-1]

    def Q(self, S, M, A):
        output = self._U(S, M, A)[:-1]
        Q1 = np.dot(output.reshape((1,-1)), self.Q1_est_beta[:-1].reshape((-1,)))
        Q2 = np.dot(output.reshape((1,-1)), self.Q2_est_beta[:-1].reshape((-1,)))
        Q3 = np.dot(output.reshape((1,-1)), self.Q3_est_beta[:-1].reshape((-1,)))
        Q4 = np.dot(output.reshape((1,-1)), self.Q4_est_beta[:-1].reshape((-1,)))
        return Q1, Q2, Q3, Q4
        #return sum(map(operator.mul, output, est_beta[:-1]))

    def _Q_diff(self, state, mediator, action, next_state):
        #MCMC to get the mean over m
        Q1_SAm, Q2_SAm, Q3_SAm, Q4_SAm = self.init_Qs()
        for rep in range(self.expectation_MCMC_iter_Q_diff):
            #np.random.seed(rep)
            m_SA = self.pmlearner.sample_m(state, action, random = True)
            out_Q1,out_Q2,out_Q3,out_Q4 = self.cal_newQ(state, m_SA, action)
            Q1_SAm = self.update_exp(rep, Q1_SAm, out_Q1)
            Q2_SAm = self.update_exp(rep, Q2_SAm, out_Q2)
            Q3_SAm = self.update_exp(rep, Q3_SAm, out_Q3)
            Q4_SAm = self.update_exp(rep, Q4_SAm, out_Q4)
        
        Q1_Snext_am, Q2_Snext_am, Q3_Snext_am, Q4_Snext_am = self.init_Qs()
        for a in self.unique_action:
            pie_a = self.target_policy(next_state, self.dim_state, a, matrix_based = True)
            pi0_a = self.control_policy(next_state, self.dim_state, a, matrix_based = True)
            Q1_Snext_am_MC, Q2_Snext_am_MC, Q3_Snext_am_MC, Q4_Snext_am_MC = self.init_Qs()
            for rep in range(self.expectation_MCMC_iter_Q_diff):
                #np.random.seed(rep)
                m_Snext_a = self.pmlearner.sample_m(next_state, np.array([a]), random = True)
                action_list = [a]*self.NT
                out_Q1, out_Q2, out_Q3, out_Q4 = self.cal_newQ(next_state, m_Snext_a, action_list)
                Q1_Snext_am_MC = self.update_exp(rep, Q1_Snext_am_MC, out_Q1)
                Q2_Snext_am_MC = self.update_exp(rep, Q2_Snext_am_MC, out_Q2)
                Q3_Snext_am_MC = self.update_exp(rep, Q3_Snext_am_MC, out_Q3)
                Q4_Snext_am_MC = self.update_exp(rep, Q4_Snext_am_MC, out_Q4)
            Q1_Snext_am += pie_a * Q1_Snext_am_MC.reshape((-1,))
            Q2_Snext_am += pie_a * Q2_Snext_am_MC.reshape((-1,))
            Q3_Snext_am += pie_a * Q3_Snext_am_MC.reshape((-1,))
            Q4_Snext_am += pi0_a * Q4_Snext_am_MC.reshape((-1,))
        
        Q1_diff = Q1_Snext_am - Q1_SAm.reshape((-1,))
        Q2_diff = Q2_Snext_am - Q2_SAm.reshape((-1,))
        Q3_diff = Q3_Snext_am - Q3_SAm.reshape((-1,))
        Q4_diff = Q4_Snext_am - Q4_SAm.reshape((-1,))
        return Q1_diff, Q2_diff, Q3_diff, Q4_diff
    
    
    def init_Qs(self):
        return np.zeros(self.NT, dtype=float),np.zeros(self.NT, dtype=float),np.zeros(self.NT, dtype=float),np.zeros(self.NT, dtype=float)
    
    def update_exp(self, rep, old_est, new_obs):
        return (rep*old_est + new_obs)/(rep+1)
    
    def cal_newQ(self, s,m,a):
        Qs = [self.Q(s[i], m[i], a[i]) for i in range(self.NT)]
        Q1 = np.array([q[0] for q in Qs]).reshape((-1,))
        Q2 = np.array([q[1] for q in Qs]).reshape((-1,))
        Q3 = np.array([q[2] for q in Qs]).reshape((-1,))
        Q4 = np.array([q[3] for q in Qs]).reshape((-1,))
        return Q1, Q2, Q3, Q4
    #def get_tuples_curr_next(self, data):
    #    tuples = [[data['state'][nt], data['action'][nt], data['mediator'][nt], data['reward'][nt], data['next_state'][nt]] 
    #              for nt in range(self.NT)]
    #    return tuples