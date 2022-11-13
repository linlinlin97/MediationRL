import numpy as np
from scipy.special import expit
from sklearn.neighbors import KernelDensity
from sklearn.model_selection import RandomizedSearchCV
from multiprocessing import Pool
from Policy import target_policy, control_policy

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
        pm = 1.0 - 1.5 * state + 2.5 * action
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
            rmean = 1.0 + 2 * state1 - 1 * action1 -  2.5 * mediator1
            #rmean = 1.0 * np.ones(shape = state1.shape[0])
            rmean = expit(rmean)
            if random:
                print("wrong")
            else:
                rmean *= 10
                reward = rmean
        else:
            rmean = 1.0 + 2 * state - 1 * action - 2.5 * mediator
            #rmean = 1.0
            rmean = expit(rmean)
            if random:
                reward = np.random.binomial(n=1, p=rmean, size=1) * 10
            else:
                rmean *= 10
                reward = rmean
        return reward
    
    def toy_sam2nextstate_model(self, state, action, mediator,random = True):
        state1 = np.copy(state).flatten()
        action1 = np.copy(action).flatten()
        mediator1 = np.copy(mediator).flatten()
        N = len(action1)
        next_state = .5 + 3.0 * state1 - 2.5 * action1 - .5 * mediator1
        next_state = expit(next_state)
        if random:
            next_state = np.random.binomial(n=1, p=next_state, size=N)

        return next_state
#################################################################################3
# define the environment of the toy example 2, iid
    def toy2_init_state(self):
        init_state = np.random.binomial(n=1, p=0.5, size=1).reshape(-1)
        return init_state
    
    def toy2_s2action_model(self, state, random):
        pa = 1.0  - 2.0*np.sum(state)
        #pa = 1.0
        pa = expit(pa)
        if random:
            pa = np.random.binomial(n=1, p=pa, size=1)
        else:
            pa = pa
        return pa
    
    def toy2_sa2mediator_model(self, state, action, random):
        pm = 1.0 - 1.5 * state + 2.5 * action
        #pm = 1.0
        pm = expit(pm)
        if random:
            pm = np.random.binomial(n=1, p=pm, size=1)
        else: 
            pm = pm
        return pm
    
    def toy2_sam2reward_model(self, state, action, mediator, random, matrix_based = False):
        if matrix_based:
            #print("matrix")
            state1 = np.copy(state).flatten()
            action1 = np.copy(action).flatten()
            mediator1 = np.copy(mediator).flatten()
            rmean = 1.0 + 2 * state1 - 1 * action1 -  2.5 * mediator1
            #rmean = 1.0 * np.ones(shape = state1.shape[0])
            rmean = expit(rmean)
            if random:
                print("wrong")
            else:
                rmean *= 10
                reward = rmean
        else:
            rmean = 1.0 + 2 * state - 1 * action - 2.5 * mediator
            #rmean = 1.0
            rmean = expit(rmean)
            if random:
                reward = np.random.binomial(n=1, p=rmean, size=1) * 10
            else:
                rmean *= 10
                reward = rmean
        return reward
    
    def toy2_sam2nextstate_model(self, state, action, mediator, random = True):
        state1 = np.copy(state).flatten()
        action1 = np.copy(action).flatten()
        mediator1 = np.copy(mediator).flatten()
        N = len(action1)
        if random:
            next_state = .2
            next_state = np.random.binomial(n=1, p=next_state, size=N)
        else:
            next_state = .2 * np.ones(N)
        return next_state
    

#################################################################################3
# define the environment of the toy --Gaussian
    def Gaussian_init_state(self):
        init_state = np.random.normal(loc=0, scale=1, size=1)
        return init_state
    
    def Gaussian_s2action_model(self, state, random):
        pa = .5*np.sum(state)
        pa = expit(pa)
        if random:
            pa = np.random.binomial(n=1, p=pa, size=1)
        else:
            pa = pa
        return pa
    
    def Gaussian_sa2mediator_model(self, state, action, random):
        pm =  .45 * state - .25 * action
        if random:
            pm += .1 * np.random.normal(loc=0, scale=1, size=1)
        else:
            pm = pm
        return pm
    
    def Gaussian_sam2reward_model(self, state, action,mediator, random, matrix_based = False):#, mediator
        if matrix_based:
            #print("matrix")
            state1 = np.copy(state).flatten()
            action1 = np.copy(action).flatten()
            mediator1 = np.copy(mediator).flatten()
            rmean = .5 * state1 - .5 * action1 +  .5 * mediator1
            rmean = expit(rmean)
            if random:
                print("wrong")
            else:
                reward = rmean
        else:
            rmean = .5 * state - .5 * action +  .5 * mediator
            if random:
                reward = rmean + .16 * np.random.normal(loc=0, scale=1, size=1)
            else:
                reward = rmean
        return reward
    
    def Gaussian_sam2nextstate_model(self, state, action, mediator):
        next_state = .9 * state - .5 * action - .25 * mediator
        next_state += .5 * np.random.normal(loc=0, scale=1, size=1)
        return next_state
    
######################################################################
# Initiate the simulator

    def __init__(self, model_type='toy', dim_state=3):
        #i.e., if only one component of state, then dim_state=1
        self.dim_state = dim_state
        if model_type == "toy":
            self.model_type = "toy"
            self.init_state_model = self.toy_init_state
            self.s2action_model = self.toy_s2action_model
            self.sa2mediator_model = self.toy_sa2mediator_model
            self.sam2reward_model = self.toy_sam2reward_model
            self.sam2nextstate_model = self.toy_sam2nextstate_model
        if model_type == "toy_iid":
            self.model_type = "toy_iid"
            self.init_state_model = self.toy2_init_state
            self.s2action_model = self.toy2_s2action_model
            self.sa2mediator_model = self.toy2_sa2mediator_model
            self.sam2reward_model = self.toy2_sam2reward_model
            self.sam2nextstate_model = self.toy2_sam2nextstate_model
        if model_type == "Gaussian_toy":
            self.model_type = "Gaussian_toy"
            self.init_state_model = self.Gaussian_init_state
            self.s2action_model = self.Gaussian_s2action_model
            self.sa2mediator_model = self.Gaussian_sa2mediator_model
            self.sam2reward_model = self.Gaussian_sam2reward_model
            self.sam2nextstate_model = self.Gaussian_sam2nextstate_model
            
            
        self.trajectory_list = []
        self.target_policy_trajectory_list = []
        self.fixed_policy_trajectory_list = []
        self.stationary_target_policy_state_density = None
        self.stationary_fixed_policy_state_density = None
        self.stationary_behaviour_policy_state_density = None
        pass

######################################################################
# utility functions used to sample random action/mediator...

    def sample_init_state(self):
        init_state = self.init_state_model()
        return init_state

#    def logistic_sampler(self, prob):
#        prob_size = np.array(prob).flatten().size
#        if prob_size <= 2:
#            if prob.ndim == 1:
#                prob = prob[0]
#            elif prob.ndim == 2:
#                prob = prob[0][0]
#            prob_arr = np.array([1-prob, prob])
#            random_y = np.random.choice([0, 1], 1, p=prob_arr)
#        else:
#            prob_arr = prob.flatten()
#            random_y = np.random.choice([-1, 0, 1], 1, p=prob_arr)
#        return random_y

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

    def sample_sam2nextstate(self, state, action, mediator):
        random_next_state = self.sam2nextstate_model(
            state, action, mediator)
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
        random_mediator = np.zeros(num_time)
        random_reward = np.zeros(num_time)

        random_state[0] = init_state.reshape(-1)
        for i in range(num_time):
            random_action[i] = self.sample_s2action(
                random_state[i])
            random_mediator[i] = self.sample_sa2mediator(
                random_state[i], random_action[i])
            random_reward[i] = self.sample_sam2reward(
                random_state[i], random_action[i], random_mediator[i])
            random_state[i+1] = self.sample_sam2nextstate(
                random_state[i], random_action[i], random_mediator[i])
            pass
        
        if burn_in:
            valid_index = range(burn_in_time, num_time+1)
            random_state = random_state[valid_index]
            valid_index = range(burn_in_time, num_time)
            random_action = random_action[valid_index]
            random_mediator = random_mediator[valid_index]
            random_reward = random_reward[valid_index]

        random_trajectory = [random_state, random_action, random_mediator, random_reward]
        
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

    def sample_one_target_policy_trajectory(self, num_time, target_policy, burn_in):
        '''
        Output: A list containing 4 elements: state, action, mediator, reward
        '''
        if burn_in:
            burn_in_time = 50
            num_time += burn_in_time
            
        init_state = self.sample_init_state()
        random_state = np.zeros((num_time+1, self.dim_state))
        random_action = np.zeros(num_time)
        random_mediator = np.zeros(num_time)
        random_reward = np.zeros(num_time)

        random_state[0] = init_state.reshape(-1)
        for i in range(num_time):
            random_action[i] = target_policy(random_state[i])
            random_mediator[i] = self.sample_sa2mediator(
                random_state[i], random_action[i])
            random_reward[i] = self.sample_sam2reward(
                random_state[i], random_action[i], random_mediator[i])
            random_state[i+1] = self.sample_sam2nextstate(
                random_state[i], random_action[i], random_mediator[i])
            pass
        
        if burn_in:
            valid_index = range(burn_in_time, num_time+1)
            random_state = random_state[valid_index]
            valid_index = range(burn_in_time, num_time)
            random_action = random_action[valid_index]
            random_mediator = random_mediator[valid_index]
            random_reward = random_reward[valid_index]

        random_trajectory = [random_state, random_action, random_mediator, random_reward]
        return random_trajectory

    def sample_target_policy_trajectory(self, num_trajectory, num_time, seed, policy = "target", burn_in=False, return_trajectory=False):
        if policy == "target":
            tmp_list = self.target_policy_trajectory_list.copy()
            self.target_policy_trajectory_list = []
            for i in range(num_trajectory):
                np.random.seed(i + 7654321*seed)
                one_trajectory = self.sample_one_target_policy_trajectory(
                    num_time, target_policy, burn_in)
                self.target_policy_trajectory_list.append(one_trajectory)
                pass
            if return_trajectory:
                to_return_list = self.target_policy_trajectory_list.copy()
                self.target_policy_trajectory_list = tmp_list
                return to_return_list
        else:
            tmp_list = self.fixed_policy_trajectory_list.copy()
            self.fixed_policy_trajectory_list = []
            for i in range(num_trajectory):
                np.random.seed(i + 7654321*seed)
                one_trajectory = self.sample_one_target_policy_trajectory(
                    num_time, control_policy, burn_in)
                self.fixed_policy_trajectory_list.append(one_trajectory)
                pass

            if return_trajectory:
                to_return_list = self.fixed_policy_trajectory_list.copy()
                self.fixed_policy_trajectory_list = tmp_list
                return to_return_list

    def onetrajectory2iid(self, trajectory):
        num_time = trajectory[1].shape[0]
        s0 = trajectory[0][0]
        state = trajectory[0][range(num_time)]
        next_state = trajectory[0][range(1, num_time+1)]
        trajectory[0] = state
        trajectory.append(next_state)
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
                iid_dataset[0] = np.vstack([iid_dataset[0], iid_data[0]]) #state 2-d
                iid_dataset[4] = np.vstack([iid_dataset[4], iid_data[4]]) #nextstate 2-d
                iid_dataset[1] = np.append(iid_dataset[1], iid_data[1]) #action 1-d
                iid_dataset[2] = np.append(iid_dataset[2], iid_data[2]) #mediator 1-d
                iid_dataset[3] = np.append(iid_dataset[3], iid_data[3]) #reward 1-d
                pass
            pass

        self.iid_dataset = {'s0': s0_dataset, 'state': iid_dataset[0],
                            'action': iid_dataset[1], 'mediator': iid_dataset[2],
                            'reward': iid_dataset[3], 'next_state': iid_dataset[4]}
        if trajectory is not None:
            return  {'s0': s0_dataset, 'state': iid_dataset[0],
                            'action': iid_dataset[1], 'mediator': iid_dataset[2],
                            'reward': iid_dataset[3], 'next_state': iid_dataset[4]}
        
        
######################################################################
# estimating truth
    def predict_discrete_ratio(self, state, policy = "target"):
        if policy == "target":
            numerator = self.stationary_target_policy_state_density * state
            numerator += (1.0 - self.stationary_target_policy_state_density) * (1.0 - state)
        elif policy == "control":
            numerator = self.stationary_fixed_policy_state_density * state
            numerator += (1.0 - self.stationary_fixed_policy_state_density) * (1.0 - state)
            
        denominator = self.stationary_behaviour_policy_state_density * state
        denominator += (1.0 - self.stationary_behaviour_policy_state_density) * (1.0 - state)
        ratio = numerator/denominator
        return ratio
    

    
    