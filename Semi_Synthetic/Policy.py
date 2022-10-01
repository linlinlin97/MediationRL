import numpy as np
from scipy.special import expit

def target_policy(state, action=None, matrix_based=False):
    # if action is not None, then return the probability
    if not matrix_based:
        pa = .3 * np.sum(state)
        pa = expit(pa)
        if pa.ndim == 1:
            pa = pa[0]
        elif pa.ndim == 2:
            pa = pa[0][0]
            pass

        prob_arr = np.array([1-pa, pa])

        if action is None:
            action_value = np.random.choice([0, 1], 1, p=prob_arr)
        else:
            action_value = np.array([prob_arr[int(action)]])
    else:
        state1 = np.copy(state).flatten()
        action1 = np.copy(action).flatten()
        pa = expit(.3 * state1)
        action_value = pa * action1 + (1-pa) * (1-action1)
    return action_value


def control_policy(state = None, action=None, get_a = False, matrix_based=False):
    # fixed policy with fixed action 0
    if get_a:
        action_value = 0
        
    elif matrix_based:
        if len(np.array([action])) == 1:
            action1 = action * np.ones(len(state))
        else:
            action1 = np.copy(action).flatten()
            
        action_value = 1 - action1 ###for a0 = 0 only
        return action_value

        
    else:
        if action is None:
            action_value = 0 
        else:
            if action == 0:
                action_value = 1
            else:
                action_value = 0
    return np.array([action_value])

