import numpy as np
from scipy.special import expit
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
from multiprocessing import Pool
from Policy import control_policy
from Policy import target_policy_Gaussian_toy as target_policy

class Simulator:
#################################################################################3
# define the environment of the toy example
    def toy_init_state(self):
        init_state = np.random.binomial(n=1, p=0.5, size=1).reshape(-1)
        return init_state
    
    def toy_s2action_model(self, state, random):
        pa = 1.0  - 2.0*np.sum(state)
        #pa = 1.0
        pa = expit(pa)
        if random:
            pa = np.random.binomial(n=1, p=pa, size=1)
        else:
            pa = pa
        return pa
    
    def toy_sa2mediator_model(self, state, action, random):
        pm = 1.0 - .5 * state + 2.5 * action
        #pm = 1.0
        pm = expit(pm)
        if random:
            pm = np.random.binomial(n=1, p=pm, size=1)
        else: 
            pm = pm
        return pm
    
    def toy_sam2reward_model(self, state, action, mediator, random, matrix_based = False):
        if matrix_based:
            #print("matrix")
            state1 = np.copy(state).flatten()
            action1 = np.copy(action).flatten()
            mediator1 = np.copy(mediator).flatten()
            rmean = 1.0 + 3.5 * state1 - 3.0 * action1 +  .5 * mediator1
            #rmean = 1.0 * np.ones(shape = state1.shape[0])
            rmean = expit(rmean)
            if random:
                raise ValueError('No random for matrix input')
            else:
                rmean *= 10
                reward = rmean
        else:
            rmean = 1.0 + 3.5 * state - 3.0 * action +  .5 * mediator
            #rmean = 1.0
            rmean = expit(rmean)
            if random:
                reward = np.random.binomial(n=1, p=rmean, size=1) * 10
            else:
                rmean *= 10
                reward = rmean
        return reward
    
    def toy_sam2nextstate_model(self, state, action, mediator, reward):
        next_state = .5 + 3.0 * state - 2.5 * action - .5 * mediator
        #next_state = .5
        next_state = expit(next_state)
        next_state = np.random.binomial(n=1, p=next_state, size=1)
        return next_state

#################################################################################3
# define the environment of the toy --Gaussian
    def Gaussian_init_state(self):
        init_state = np.random.normal(loc=0, scale=1, size=1)
        return init_state
    
    def Gaussian_s2action_model(self, state, random):
        #pa = .1*np.sum(state)
        #pa = expit(pa)
        pa = .3
        if random:
            pa = np.random.binomial(n=1, p=pa, size=1)
        else:
            pa = pa
        return pa
    
    def Gaussian_sa2mediator_model(self, state, action, random):
        state = state.reshape((-1,self.dim_state))
        N = state.shape[0]
        pm =  .1 + .5 * state + .8 * (action - .5)
        pm = pm.reshape((N,))
        if random:
            #pm += np.random.normal(loc=0, scale=1, size=1)
            pm += .5 * np.random.normal(loc=0, scale=1, size=(N,))
        else:
            pm = pm
        return pm
    
    def Gaussian_sam2reward_model(self, state, action,mediator, random, matrix_based = False):#, mediator
        if matrix_based:
            #print("matrix")
            state1 = np.copy(state).flatten()
            action1 = np.copy(action).flatten()
            mediator1 = np.copy(mediator).flatten()
            rmean = .7 * state1 +  .6 * mediator1 + .9 * (action1-.5) + (.4 * np.sqrt(abs(mediator1)) ) * (action1 - .5)
            rmean *= 5
            if random:
                raise ValueError('No random for matrix input')
            else:
                reward = rmean
        else:
            rmean = .7 * state +  .6 * mediator  + .9 * (action-.5) + (.4 * np.sqrt(abs(mediator))) * (action - .5)
            rmean *= 5
            if random:
                #reward = rmean + 1 * np.random.normal(loc=0, scale=1, size=1)
                reward = rmean + .5 * np.random.normal(loc=0, scale=1, size=1)
            else:
                reward = rmean
        return reward
    
    def Gaussian_sam2nextstate_model(self, state, action, mediator, reward):
        state = state.reshape((-1,self.dim_state))
        N = state.shape[0]
        next_state = .3 * state +  .4 * mediator + 1 * (action - .5)
        next_state += .5 * np.random.normal(loc=0, scale=1, size=N)
        return next_state

#################################################################################3
# define the environment of the semi-synthetic --Gaussian
    def semi_init_state(self):
        init_state = np.random.normal(loc=0, scale=1, size=self.dim_state)
        return init_state
    
    def semi_s2action_model(self, state, random):
        pa = .5
        if random:
            pa = np.random.binomial(n=1, p=pa, size=1)
        else:
            pa = pa
        return pa
    
    def semi_sa2mediator_model(self, state, action, random):
        state = state.reshape((-1,self.dim_state))
        N = state.shape[0]
        if self.dim_mediator == 2:
            pm1 = np.sqrt(abs(state.reshape((-1,1)))) + (action.reshape((-1,1)) - .5)
            pm2 = -.5 * state.reshape((-1,1)) + .5 * (action.reshape((-1,1)) - .5) * np.sqrt(abs(state.reshape((-1,1))))
            pm = np.hstack([pm1, pm2])
            
        pm = pm.reshape((N, self.dim_mediator))
        if random:
            #pm += np.random.normal(loc=0, scale=1, size=1)
            pm +=  self.std_M * np.random.normal(loc=0, scale=1, size=(N, self.dim_mediator))
        else:
            pm = pm
        return pm
    
    def semi_sam2reward_model(self, state, action,mediator, random, matrix_based = False):#, mediator
        if matrix_based:
            #print("matrix")
            if self.dim_mediator == 2:
                state1 = state.reshape((-1,self.dim_state))
                mediator1 = mediator.reshape((-1,self.dim_mediator))
                action1 = action.reshape((-1,1))
                rmean = .5 * state1.reshape((-1,1))
                rmean += .5 * np.sqrt(abs(state1.reshape((-1,1))))
                rmean += mediator1[:,0].reshape((-1,1))
                rmean += mediator1[:,1].reshape((-1,1))
                rmean += .5 * np.sqrt(abs(mediator1[:,0]).reshape((-1,1))+ abs(mediator1[:,1]).reshape((-1,1))) * (action1 - .5)
                rmean += .5*(action1 - .5)
                rmean *= 1.5
                
            if random:
                print("wrong")
            else:
                reward = rmean.flatten()
        else:
            if self.dim_mediator == 2:
                rmean = .5 * state 
                rmean += .5 * np.sqrt(abs(state)) 
                rmean += 1 * mediator[0] 
                rmean += 1 * mediator[1] 
                rmean += .5 * np.sqrt(abs(mediator[0])+abs(mediator[1])) * (action - .5)
                rmean += .5*(action - .5)
                rmean *= 1.5
            if random:
                reward = rmean + self.std_S * np.random.normal(loc=0, scale=1, size=1)
            else:
                reward = rmean
        return reward
    
    def semi_sam2nextstate_model(self, state, action, mediator, reward):
        next_state = reward
        return next_state
######################################################################
# Initiate the simulator

    def __init__(self, model_type='toy', dim_state=1, dim_mediator = 1, std_M = None, std_S = None):
        #i.e., if only one component of state, then dim_state=1
        self.dim_state = dim_state
        self.dim_mediator = dim_mediator
        self.std_M = std_M
        self.std_S = std_S
        if model_type == "toy":
            self.model_type = "toy"
            self.init_state_model = self.toy_init_state
            self.s2action_model = self.toy_s2action_model
            self.sa2mediator_model = self.toy_sa2mediator_model
            self.sam2reward_model = self.toy_sam2reward_model
            self.sam2nextstate_model = self.toy_sam2nextstate_model
        elif model_type == "Gaussian_toy":
            self.model_type = "Gaussian_toy"
            self.init_state_model = self.Gaussian_init_state
            self.s2action_model = self.Gaussian_s2action_model
            self.sa2mediator_model = self.Gaussian_sa2mediator_model
            self.sam2reward_model = self.Gaussian_sam2reward_model
            self.sam2nextstate_model = self.Gaussian_sam2nextstate_model
        elif model_type == "Gaussian_semi":
            self.model_type = "Gaussian_semi"
            self.init_state_model = self.semi_init_state
            self.s2action_model = self.semi_s2action_model
            self.sa2mediator_model = self.semi_sa2mediator_model
            self.sam2reward_model = self.semi_sam2reward_model
            self.sam2nextstate_model = self.semi_sam2nextstate_model
            
            
        self.trajectory_list = []
        self.stationary_target_policy_state_density = None
        self.stationary_fixed_policy_state_density = None
        self.stationary_behaviour_policy_state_density = None
        pass

######################################################################
# utility functions used to sample random action/mediator...

    def sample_init_state(self):
        init_state = self.init_state_model()
        return init_state

    def sample_s2action(self, state, random=True):
        '''
        Output: a random action
        '''
        random_action = self.s2action_model(state, random)
        return random_action

    def sample_sa2mediator(self, state, action, random=True):
        '''
        Output: a random mediator
        '''
        random_mediator = self.sa2mediator_model(state, action, random)
        return random_mediator

    def sample_sam2reward(self, state, action, mediator, random=True, matrix_based = False):
        random_reward = self.sam2reward_model(state, action, mediator, random=random, matrix_based = matrix_based)
        return random_reward

    def sample_sam2nextstate(self, state, action, mediator, reward):
        random_next_state = self.sam2nextstate_model(
            state, action, mediator, reward)
        return random_next_state
    
######################################################################
# Sample trajectories

    def sample_one_trajectory(self, num_time, burn_in):
        '''
        Output: A list containing 4 elements: state, action, mediator, reward
        '''
        if burn_in:
            burn_in_time = 50
            num_time += burn_in_time

        init_state = self.sample_init_state()
        random_state = np.zeros((num_time+1, self.dim_state))
        random_action = np.zeros(num_time)
        random_mediator = np.zeros((num_time, self.dim_mediator))
        random_reward = np.zeros(num_time)
        time_idx = np.array(np.arange(num_time))

        random_state[0] = init_state.reshape(-1)
        for i in range(num_time):
            random_action[i] = self.sample_s2action(
                random_state[i])
            random_mediator[i] = self.sample_sa2mediator(
                random_state[i], random_action[i])
            random_reward[i] = self.sample_sam2reward(
                random_state[i], random_action[i], random_mediator[i])
            random_state[i+1] = self.sample_sam2nextstate(
                random_state[i], random_action[i], random_mediator[i], random_reward[i])
            pass
        
        if burn_in:
            valid_index = range(burn_in_time, num_time+1)
            random_state = random_state[valid_index]
            valid_index = range(burn_in_time, num_time)
            random_action = random_action[valid_index]
            random_mediator = random_mediator[valid_index]
            random_reward = random_reward[valid_index]

        random_trajectory = [random_state, random_action, random_mediator, random_reward, time_idx]
        
        return random_trajectory
    
    def sample_trajectory(self, num_trajectory, num_time, seed, burn_in=False, return_trajectory=False):
        tmp_list = self.trajectory_list.copy()
        self.trajectory_list = []
        np.random.seed(4321*seed)
        for i in range(num_trajectory):
            one_trajectory = self.sample_one_trajectory(num_time, burn_in)
            self.trajectory_list.append(one_trajectory)
            pass

        if return_trajectory:
            to_return_list = self.trajectory_list.copy()
            self.trajectory_list = tmp_list
            return to_return_list

    def sample_one_target_policy_trajectory(self, num_time, target_policy, control_policy, burn_in, sample_policy = 'control'):
        '''
        Output: A list containing 4 elements: state, action, mediator, reward
        '''
        if burn_in:
            burn_in_time = 50
            num_time += burn_in_time
            
        init_state = self.sample_init_state()
        random_state = np.zeros((num_time+1, self.dim_state))
        random_action = np.zeros(num_time)
        random_mediator = np.zeros((num_time, self.dim_mediator))
        random_reward = np.zeros(num_time)
        time_idx = np.array(np.arange(num_time))

        random_state[0] = init_state.reshape(-1)
        for i in range(num_time):
            if sample_policy == 'control':
                random_action[i] = control_policy(state = random_state[i], dim_state = self.dim_state)
                action_t = random_action[i]
            elif sample_policy == 'target':
                random_action[i] = target_policy(state = random_state[i], dim_state = self.dim_state)
                action_t = random_action[i]
            else:
                random_action[i] = control_policy(state = random_state[i], dim_state = self.dim_state)
                action_t = target_policy(state = random_state[i], dim_state = self.dim_state)
                
            random_mediator[i] = self.sample_sa2mediator(
                random_state[i], action_t)
            random_reward[i] = self.sample_sam2reward(
                random_state[i], random_action[i], random_mediator[i])
            random_state[i+1] = self.sample_sam2nextstate(
                random_state[i], random_action[i], random_mediator[i], random_reward[i])
            pass
        
        if burn_in:
            valid_index = range(burn_in_time, num_time+1)
            random_state = random_state[valid_index]
            valid_index = range(burn_in_time, num_time)
            random_action = random_action[valid_index]
            random_mediator = random_mediator[valid_index]
            random_reward = random_reward[valid_index]

        random_trajectory = [random_state, random_action, random_mediator, random_reward, time_idx]
        return random_trajectory

    def sample_target_policy_trajectory(self, num_trajectory, num_time, seed, policy = "target", burn_in=False, return_trajectory=False):
        self.target_policy_trajectory_list = []
        for i in range(num_trajectory):
            np.random.seed(i + 7654321*seed)
            one_trajectory = self.sample_one_target_policy_trajectory(
                    num_time, target_policy, control_policy, burn_in, sample_policy = policy)
            self.target_policy_trajectory_list.append(one_trajectory)
            pass
        if return_trajectory:
            return self.target_policy_trajectory_list

    def onetrajectory2iid(self, trajectory):
        num_time = trajectory[1].shape[0]
        s0 = trajectory[0][0]
        state = trajectory[0][range(num_time)]
        next_state = trajectory[0][range(1, num_time+1)]
        trajectory[0] = state
        trajectory.append(trajectory[4])
        trajectory[4] = next_state
        return s0, trajectory

    def trajectory2iid(self, trajectory=None):
        iid_dataset = []
        if trajectory is None:
            trajectory_list = self.trajectory_list.copy()
        else:
            trajectory_list = trajectory.copy()
            pass

        num_trajectory = len(trajectory_list)
        for i in range(num_trajectory):
            s0_data, iid_data = self.onetrajectory2iid(trajectory_list[i])
            if i == 0:
                iid_dataset = iid_data
                s0_dataset = s0_data
            else:
                s0_dataset = np.vstack([s0_dataset, s0_data])
                iid_dataset[0] = np.vstack([iid_dataset[0], iid_data[0]]) #state 3-d
                iid_dataset[4] = np.vstack([iid_dataset[4], iid_data[4]]) #nextstate 3-d
                iid_dataset[1] = np.append(iid_dataset[1], iid_data[1]) #action 1-d
                iid_dataset[2] = np.vstack([iid_dataset[2], iid_data[2]]) #mediator 2-d
                iid_dataset[3] = np.append(iid_dataset[3], iid_data[3]) #reward 1-d
                iid_dataset[5] = np.append(iid_dataset[5], iid_data[5]) #time 1-d
                
                pass
            pass
        if trajectory is None:
            self.iid_dataset = {'s0': s0_dataset, 'state': iid_dataset[0],
                                'action': iid_dataset[1], 'mediator': iid_dataset[2],
                                'reward': iid_dataset[3], 'next_state': iid_dataset[4],
                                'time_idx': iid_dataset[5]}
        else:
            return  {'s0': s0_dataset, 'state': iid_dataset[0],
                     'action': iid_dataset[1], 'mediator': iid_dataset[2],
                     'reward': iid_dataset[3], 'next_state': iid_dataset[4],
                     'time_idx': iid_dataset[5]}
        

    
    