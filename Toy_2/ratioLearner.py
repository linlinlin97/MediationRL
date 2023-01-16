import numpy as np
from numpy.linalg import inv
from sklearn.kernel_approximation import RBFSampler

class RatioLinearLearner:
    def __init__(self, dataset, target_policy, control_policy, palearner, ndim=100, truncate=20, dim_state = 1, l2penalty = 1.0):
        
        self.dim_state = dim_state
        self.state = np.copy(dataset['state']).reshape(-1, self.dim_state)
        self.action = np.copy(dataset['action']).reshape(-1, 1)
        self.unique_action = np.unique(dataset['action'])
        self.next_state = np.copy(dataset['next_state']).reshape(-1, self.dim_state)
        self.s0 = np.copy(dataset['s0']).reshape(-1, self.dim_state)

        self.target_policy = target_policy
        self.control_policy = control_policy
        self.beta_target = None
        self.beta_control = None
        self.truncate = truncate
        self.l2penalty = l2penalty
        
        self.palearner = palearner
        pass

    def feature_engineering(self, feature):
        feature_new = np.hstack([np.repeat(1, feature.shape[0]).reshape(-1, 1), feature])
        return feature_new

    def fit(self):
        psi = self.feature_engineering(self.state)
        psi_next = self.feature_engineering(self.next_state)

        self.estimate_pa = self.palearner.get_pa_prediction(self.state, self.action)
        self.target_pa = self.target_policy(state = self.state, dim_state=self.dim_state, action=self.action).flatten()
        self.control_pa = self.control_policy(state = self.state, dim_state=self.dim_state, action=self.action).flatten()
        
        self.pa_ratio_target = self.target_pa / self.estimate_pa
        self.pa_ratio_control = self.control_pa / self.estimate_pa
        # print(np.mean(ratio)) # close to 1 if behaviour and target are the same
        
        #target_ratio_learning
        self.beta_target = self._beta(psi, psi_next, self.pa_ratio_target)
        #control_ratio_learning
        self.beta_control = self._beta(psi, psi_next, self.pa_ratio_control)
        pass
    
    def _beta(self, psi, psi_next, pa_ratio):
        psi_minus_psi_next = self.rbf_difference(psi, psi_next, pa_ratio)
        design_matrix_up = np.zeros((psi.shape[1], psi.shape[1]))
        design_matrix_down = np.zeros((1, psi.shape[1]))
        for i in range(self.state.shape[0]):
            design_matrix_up += np.matmul(psi_minus_psi_next[i].reshape(-1, 1), psi[i].reshape(1, -1))
            design_matrix_down += psi[i].reshape(1, -1)
        design_matrix_up /= self.state.shape[0]
        design_matrix_down /= self.state.shape[0]
        
        X = np.vstack((design_matrix_up, design_matrix_down))
        XTX = np.matmul(X.T, X)
        #print(XTX)
        if self.l2penalty is not None:
            penalty_matrix = np.diagflat(np.repeat(self.l2penalty, XTX.shape[0]))
            XTX += penalty_matrix
        #print('+',XTX )
        inv_design_matrix = inv(XTX)

        beta_target = np.matmul(inv_design_matrix, design_matrix_down.reshape(-1, 1))
        return beta_target
    
    def rbf_difference(self, psi, psi_next, ratio):
        psi_minus_psi_next = psi - (psi_next.transpose() * ratio).transpose()
        return psi_minus_psi_next

    def get_ratio_prediction(self, state, policy = 'target',normalize=True):
        '''
        Input:
        state: a numpy.array
        Output:
        A 1D numpy array. The probability ratio in certain states.
        '''
        if np.ndim(state) == 0 or np.ndim(state) == 1:
            x_state = np.reshape(state, (1, -1))
        else:
            x_state = np.copy(state).reshape((-1,self.dim_state))
        psi = self.feature_engineering(x_state)
        if policy == 'target':
            ratio = np.matmul(psi, self.beta_target).flatten()
        elif policy == 'control':
            ratio = np.matmul(psi, self.beta_control).flatten()
        ratio_min = 1 / self.truncate
        ratio_max = self.truncate
        ratio = np.clip(ratio, a_min=ratio_min, a_max=ratio_max)
        if state.shape[0] > 1:
            if normalize:
                ratio /= np.mean(ratio)
        return ratio

    def get_r_prediction(self, state, policy = 'target', normalize=True):
        return self.get_ratio_prediction(state, policy, normalize)
    
    def goodness_of_fit(self,test_dataset):
        np.random.seed(1)        
        psi = self.feature_engineering(test_dataset['state'])
        psi_next = self.feature_engineering(test_dataset['next_state'])
        
        estimate_pa = self.palearner.get_pa_prediction(test_dataset['state'], test_dataset['action'])
        target_pa = self.target_policy(state = test_dataset['state'], dim_state = self.dim_state,
                                           action=test_dataset['action']).flatten()
        control_pa = self.control_policy(state = test_dataset['state'], dim_state = self.dim_state, action = test_dataset['action']).flatten()
        pa_ratio_target = target_pa / estimate_pa
        pa_ratio_control = control_pa / estimate_pa
        
        psi_minus_psi_next_target = self.rbf_difference(psi, psi_next, pa_ratio_target)
        psi_minus_psi_next_control = self.rbf_difference(psi, psi_next, pa_ratio_control)

        rmse_target = [np.matmul(np.matmul(psi_minus_psi_next_target[i].reshape(-1, 1), psi[i].reshape(1, -1)),self.beta_target) for i in range(test_dataset['state'].shape[0])]
        rmse_target = [np.vstack([rmse_target[i], np.matmul(psi[i].reshape(1, -1),self.beta_target)-1]) for i in range(test_dataset['state'].shape[0])]
        rmse_target = np.sqrt(np.mean(np.square(np.mean(rmse_target,axis=0)))) 
        
        rmse_control = [np.matmul(np.matmul(psi_minus_psi_next_control[i].reshape(-1, 1), psi[i].reshape(1, -1)),self.beta_control) for i in range(test_dataset['state'].shape[0])]
        rmse_control = [np.vstack([rmse_control[i], np.matmul(psi[i].reshape(1, -1),self.beta_control)-1]) for i in range(test_dataset['state'].shape[0])]
        rmse_control = np.sqrt(np.mean(np.square(np.mean(rmse_control,axis=0)))) 
        return rmse_target, rmse_control