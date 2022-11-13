import numpy as np
from scipy.special import expit

def target_policy(state, dim_state = None, action=None):
    # if action is not None, then return the probability
    state = np.copy(state).reshape((-1, dim_state))
    NT = state.shape[0]
    pa = expit(1.5 + 1.0 * np.sum(state,axis = 1)).flatten()
    if action is None:
        if NT == 1:
            pa = pa[0]
            prob_arr = np.array([1-pa, pa])
            action_value = np.random.choice([0, 1], 1, p=prob_arr)
        else:
            raise ValueError('No random for matrix input')
    else:
        action = np.copy(action).flatten()
        action_value = pa * action + (1-pa) * (1-action)
    return action_value

def control_policy(state = None, dim_state=None, action=None, get_a = False):
    # fixed policy with fixed action 0
    if get_a:
        action_value = np.array([0])
    else:
        state = np.copy(state).reshape(-1,dim_state)
        NT = state.shape[0]
        if action is None:
            action_value = np.array([0]*NT)
        else:
            action = np.copy(action).flatten()
            if len(action) == 1 and NT>1:
                action = action * np.ones(NT)
            action_value = 1-action
    return action_value

