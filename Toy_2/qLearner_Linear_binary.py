import numpy as np

class Qlearner:
    def __init__(self, data, target_policy, PMLearner, RewardLearner, control_policy, Q_func = "Q1"):          
        self.data = data
        self.action = np.copy(data['action']).flatten()
        self.mediator = np.copy(data['mediator']).flatten()
        self.state = np.copy(data['state']).flatten()
        self.next_state = np.copy(data['next_state']).flatten()
        self.data_num = len(data['action'])

        self.unique_action = np.unique(data['action'])
        self.unique_mediator = np.unique(data['mediator'])
        
        self.pmlearner = PMLearner
        self.rewardlearner = RewardLearner
        self.target_policy = target_policy
        
        if Q_func == "Q1":
            self.Y = np.copy(data['reward']).flatten()#-self.true_eta_target
        elif Q_func == "Q2":
            self.a0 = control_policy(get_a = True)
            self.Y = self.rewardlearner.get_reward_prediction(self.state, self.a0, self.mediator)
        elif Q_func == "Q3":
            self.a0 = control_policy(get_a = True)
            self.Y = np.zeros(shape=self.state.shape).flatten()
            for m in self.unique_mediator:
                Er_Sa0m = self.rewardlearner.get_reward_prediction(self.state, self.a0, np.array([m]))
                pm_Sa0 = self.pmlearner.get_pm_prediction(self.state, self.a0, np.array([m]))
                self.Y += Er_Sa0m * pm_Sa0 
            #self.Y -= self.true_eta_target_a0star   
        elif Q_func == "Q4":
            self.a0 = control_policy(get_a = True)
            self.Y = np.copy(data['reward']).flatten()#-self.true_eta_a0
        else:
            "Warning!"
            
        self.phi = self.get_phi(self.state, self.action, self.mediator, include_eta = True)
        self.w = self.get_w(self.next_state)
        
        self.Q_beta_hat, self.eta = self.fit()
        self.Q_diff = self.Q_diff_eta(self.data_num, self.state, self.action, self.next_state, self.target_policy)

        
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
                pie_a = self.target_policy(self.state, dim_state = 1, action = a)
                pm_aSprime = self.pmlearner.get_pm_prediction(next_state1, np.array([a]), np.array([m]))
                X_Sprime_am = self.get_phi(next_state1.flatten(), a, m)
                w += (np.array(X_Sprime_am).T * pie_a * pm_aSprime).T
        return w

    
    def estimate_beta(self, phi, w, reward):
        NT = reward.shape[0]
        p = phi.shape[1]
        A = np.matmul(phi.T, (phi-w))/NT
        B = np.matmul(phi.T, reward)/NT
        beta = np.matmul(np.linalg.inv(A),B)
        self.beta_hat = beta[:-1]
        self.eta = beta[-1]
        return self.beta_hat, self.eta
    
    def fit(self):
        return self.estimate_beta(self.phi, self.w, self.Y)
    
    
    def true_q_function(self, state, action, mediator):
        state1 = np.copy(state).flatten()
        action1 = np.copy(action).flatten()
        mediator1 = np.copy(mediator).flatten()
        model = self.beta_hat
        Indicator = self.get_phi(state, action1, mediator1)
        true_Q = np.matmul(Indicator[:,:7], model)

        return true_Q
        
    def get_q_prediction(self, state, action, mediator):
        return self.true_q_function(state, action, mediator)
  

    def Q_diff_eta(self, data_num, state, action, next_state, policy):

        Sprime, AS = self.compute_Q(data_num, state, action, next_state, policy)
        return Sprime-AS
    
    def compute_Q(self, data_num, state, action, next_state, policy):
        self.Sprime = np.zeros(data_num, dtype=float)
        for a in self.unique_action:
            pie_a_Sprime = policy(next_state, dim_state = 1, action = a)
            for m in self.unique_mediator:
                pm_a_Sprime = self.pmlearner.get_pm_prediction(next_state, np.array([a]), np.array([m]))
                Sam_prime = self.get_q_prediction(next_state, a, m)
                self.Sprime += Sam_prime * pm_a_Sprime * pie_a_Sprime
                
        self.SA = np.zeros(data_num, dtype=float)
        for m in self.unique_mediator:
            pm_AS = self.pmlearner.get_pm_prediction(state, action, np.array([m]))
            SAm = self.get_q_prediction(state, action, m)
            self.SA += SAm * pm_AS
        return self.Sprime, self.SA