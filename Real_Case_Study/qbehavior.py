from scipy.interpolate import BSpline
from itertools import product
from numpy.linalg import inv
from probLearner import PMLearner, PALearner
import numpy as np
from _util import *
class Learn_Behavior_Q():
    def __init__(self, data, PMlearner, PAlearner, dim_state, dim_mediator,
                 problearner_parameters = {"splitter":["best","random"], "max_depth" : range(1,20)},
                 Q_settings = {'scaler': 'Identity', 'product_tensor': True, 'beta': 3/7, 'include_intercept': False,
                               'expectation_MCMC_iter_Q3': 100, 'expectation_MCMC_iter_Q_diff': 100, 'penalty': 10**(-9),
                              'd': 3, 'min_L': 7}, seed = 0):
        np.random.seed(seed)
        self.state = data['state']
        self.mediator = data['mediator']
        self.action = data['action']
        self.R = data['reward']
        self.next_state = data['next_state']
        self.s0 = data['s0']
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
        
        if Q_settings['scaler'] == "NormCdf":
            self.scaler = normcdf()
            self.sm_mean = np.mean(self.tuples_array[:,self.state_mediator_col],axis = 0)
            self.sm_std = np.std(self.tuples_array[:,self.state_mediator_col],axis = 0)
        elif Q_settings['scaler'] == "Identity":
            self.scaler = iden()
        elif self.scaler_setting == "Standardize":
            self.scaler = iden()
            self.sm_mean = np.mean(self.tuples_array[:,self.state_mediator_col],axis = 0)
            self.sm_std = np.std(self.tuples_array[:,self.state_mediator_col],axis = 0)
            
        self.B_spline(L = max(self.min_L,self.L+self.d), d = self.d)
        
        self.expectation_MCMC_iter_Q3 = Q_settings['expectation_MCMC_iter_Q3']
        self.expectation_MCMC_iter_Q_diff = Q_settings['expectation_MCMC_iter_Q_diff']
        self.l2penalty = Q_settings['penalty']

        self.pmlearner = PMLearner(data, problearner_parameters, seed, dim_state = dim_state,
                                           dim_mediator = dim_mediator)
        self.pmlearner.train()

        self.palearner = PALearner(data, problearner_parameters, seed, test = False, dim_state = dim_state,
                                           dim_mediator = dim_mediator)
        self.palearner.train()
        
        self.U = self._U4all(self.tuples)
        self.Sigma_behavior = self._Sigma(self.tuples, self.U)
        self.est_Q()

    def est_Q(self):
        self.eta_b, self.est_beta = self.est_beta_eta(self.R, self.U, self.Sigma_behavior)
        
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
        if self.scaler_setting == "NormCdf" or self.scaler_setting == "Standardize":
            x = (x-self.sm_mean)/self.sm_std
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
    
    def _Xi(self, S_next, include_eta = False):
        phi = np.zeros((self.para_dim*len(self.unique_action)+1,))
        for a in self.unique_action:
            pa = self.palearner.get_pa_prediction(state = S_next.reshape((-1,1)), action = np.array([a]))
            if pa > 0:
                sample_phi = np.zeros((self.para_dim*len(self.unique_action)+1,))
                for rep in range(self.expectation_MCMC_iter_Q3):
                    np.random.seed(rep)
                    m = self.pmlearner.sample_m(state = S_next, action = np.array([a]), random = True)
                    m = m.reshape((self.dim_mediator,))
                    sample_phi = self.update_exp(rep, sample_phi, self._U(S_next, m, a)[:-1].reshape(-1,))
                phi += (np.array(sample_phi)*pa).reshape(-1,)   
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
    
    def _Sigma(self, tuples, U):
        output = [np.matmul(U[i],
                            (U[i] - self._Xi(np.array(tuples[i][-self.dim_state:]), include_eta = False)).T) for i in range(len(tuples))]
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
        inv_Sigma_hat = inv(Sigma_hat)
        est_beta = np.matmul(inv_Sigma_hat, vector)
        return est_beta


    def eta_hat(self, est_beta):
        return est_beta[-1]

    def Q(self, S, M, A):
        output = self._U(S, M, A)[:-1]
        return np.dot(output.reshape((1,-1)), self.est_beta[:-1].reshape((-1,)))
    
    def opt_A(self, state):
        #MCMC to get the mean over m
        state = state.reshape((-1,self.dim_state))
        NT = state.shape[0]
        Q_SA = np.zeros((NT,2))
        for a in [0,1]:
            for rep in range(self.expectation_MCMC_iter_Q_diff):
                np.random.seed(rep)
                m_SA = self.pmlearner.sample_m(state, np.array([a]), random = True)
                out_Q = np.array([self.Q(state[i], m_SA[i], a) for i in range(NT)]).reshape((-1,))
                Q_SA[:,a] = self.update_exp(rep, Q_SA[:,a], out_Q)
        opt_A = np.argmax(Q_SA,axis=1)
        return opt_A
    
    def update_exp(self, rep, old_est, new_obs):
        return (rep*old_est + new_obs)/(rep+1)
    