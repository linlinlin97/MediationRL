import numpy as np
from scipy.special import expit

def get_phi(state, action, mediator):
    '''000, 100, 010, 001, 110, 101, 011, 111'''
    state1 = np.copy(state).flatten()
    indicator = np.zeros(shape = (len(state1),7))
    #indicator[(state1==0)*(action==0)*(mediator==0),0] = 1 reference group
    indicator[(state1==1)*(action==0)*(mediator==0),0] = 1
    indicator[(state1==0)*(action==1)*(mediator==0),1] = 1
    indicator[(state1==0)*(action==0)*(mediator==1),2] = 1
    indicator[(state1==1)*(action==1)*(mediator==0),3] = 1
    indicator[(state1==1)*(action==0)*(mediator==1),4] = 1
    indicator[(state1==0)*(action==1)*(mediator==1),5] = 1
    indicator[(state1==1)*(action==1)*(mediator==1),6] = 1
    return indicator
    
def true_q_function(state, action, mediator, typ = "Q1"):
    state1 = np.copy(state).flatten()
    action1 = np.copy(action).flatten()
    mediator1 = np.copy(mediator).flatten()
    if typ == "Q1":
        model = np.array([  4.5468457 ,  -5.6803222 ,  -6.30630156,   2.22455077,
         1.12441126, -10.2119709 ,  -3.5351791 ])
    elif typ == "Q2":
        model = np.array([  5.36285339,  -4.54902292,  -6.59326191,   3.19680312,
         1.89797304, -10.42715419,  -1.08599247])
    elif typ == "Q3":
        model = np.array([ 8.44807161, -5.01250534, -1.21971234,  6.06133082,  8.26765716,
       -5.44422513,  4.97966689])
    elif typ == "Q4":
        model = np.array([  8.11246785, -4.527494  , -1.10169265,  5.95666873,  7.94951035,
       -4.91744047,  4.97966689 ])
    elif typ == "Q5":
        model = np.array([  5.2622589 ,  -6.71423051,  -6.55788647,   2.44766102,
         1.80261113, -11.33492822,  -3.5351791 ])
    Indicator = get_phi(state, action1, mediator1)
    true_Q = np.matmul(Indicator, model)
    #true_Q = model[0] + model[1] * action1 + model[2] * state1 + model[3] * mediator1 + model[4] * action1 * state1 + model[5] * action1 * mediator1 + model[6] * state1 * mediator1 + model[7] * action1 * state1 * mediator1
    return true_Q
        
def false_q_function(state, action, mediator, typ = "Q1"):
    #true_Q = true_q_function(action, state, mediator, typ)
    #noise_Q = true_Q + np.random.normal(loc=1, scale=1, size=true_Q.shape[0])
    state1 = np.copy(state).flatten()
    action1 = np.copy(action).flatten()
    mediator1 = np.copy(mediator).flatten()
    true_Q = true_q_function(state, action, mediator, typ)
    
    if typ == "Q1":
        model = np.random.normal(loc=1, scale=1, size=7)
    elif typ == "Q2":
        model = np.random.normal(loc=.5, scale=1, size=7)
    elif typ == "Q3":
        model = np.random.normal(loc=2, scale=1, size=7)
    elif typ == "Q4":
        model = np.random.normal(loc=1.5, scale=1, size=7)
    elif typ == "Q5":
        model = np.random.normal(loc=1.5, scale=1, size=7)
    Indicator = get_phi(state, action1, mediator1)
    noise = np.matmul(Indicator, model)
    #noise = model[0] + model[1] * action1 + model[2] * state1 + model[3] * mediator1 + model[4] * action1 * state1 + model[5] * action1 * mediator1 + model[6] * state1 * mediator1 + model[7] * action1 * state1 * mediator1
    noise_Q = true_Q + noise
    return noise_Q


def true_ratio_function(state, policy = "target", return_all = False):
    state1 = np.copy(state).flatten()
    if return_all:
        if policy == "target":
            ratio = [0.5207450263918821,1.7684599385521451]
        elif policy == "control":
            ratio = [1.5199193877713393,0.16633682949416845]
        elif policy == 'G':
            ratio = [1.4923034101190835, 0.21061758537227568]
    else:
        if policy == "target":
            numerator = 0.3207236581709146 * state1
            numerator += (1.0 - 0.3207236581709146) * (1.0 - state1)
        elif policy == "control":
            numerator = 0.9361090004997501 * state1
            numerator += (1.0 - 0.9361090004997501) * (1.0 - state1)
        elif policy == "G":
            numerator = 0.9191004897551226 * state1
            numerator += (1.0 - 0.9191004897551226) * (1.0 - state1)

        denominator = 0.6158938480759618 * state1
        denominator += (1.0 - 0.6158938480759618) * (1.0 - state1)
        ratio = numerator/denominator
    return ratio

#def false_ratio_function(state, policy = "target"):
#    true_ratio = true_ratio_function(state, policy)
#    noise_ratio = true_ratio + np.random.uniform(low=0.0, high=1.0, size=true_ratio.shape[0])
#    noise_ratio = np.clip(noise_ratio, a_min=0.01, a_max=1.0)
#    return noise_ratio


def false_ratio_function(state, policy = "target"):
    true_ratio = true_ratio_function(state, policy, return_all = True)
    ps_one, ps_0 = true_ratio 
    if policy == "target":
        ps_one += .25
        ps_0 -= .25
    elif policy == "control":
        ps_one -= .3
        ps_0 += .3
    elif policy == "G":
        ps_one -= .3
        ps_0 += .3
    state1 = np.copy(state).flatten()
    noise_ratio = ps_one * state1 + ps_0 * (1.0 - state1)
    return noise_ratio

def true_pm_function(state, action, mediator,return_pm_one = False):
    state1 = np.copy(state).flatten()
    action1 = np.copy(action).flatten()
    mediator1 = np.copy(mediator).flatten()
    pm_one = expit(1.0 - 1.5 * state1 + 2.5 * action1)
    if return_pm_one:
        pm = pm_one
    else:
        pm = mediator1*pm_one + (1.0-mediator1)*(1.0-pm_one)
    return pm

def false_pm_function(state, action, mediator):
    mediator1 = np.copy(mediator).flatten()
    true_pm_one = true_pm_function(state, action, mediator, return_pm_one = True)
    noise_pm_one = true_pm_one * np.random.uniform(low=0.7, high=1.2, size= true_pm_one.shape[0])#1)#
    noise_pm_one = np.clip(noise_pm_one, a_min=0.01, a_max=.99)
    noise_pm = mediator1*noise_pm_one + (1.0-mediator1)*(1.0-noise_pm_one)
    return noise_pm

def true_reward_function(state, action, mediator):
    state1 = np.copy(state).flatten()
    action1 = np.copy(action).flatten()
    mediator1 = np.copy(mediator).flatten()
    true_rmean = 1.0 + 2 * state1 - 1 * action1 - 2.5 * mediator1
    true_rmean = expit(true_rmean)
    true_rmean *= 10
    return true_rmean

def false_reward_function(state, action, mediator):
    state1 = np.copy(state).flatten()
    action1 = np.copy(action).flatten()
    mediator1 = np.copy(mediator).flatten()
    true_rmean = true_reward_function(state, action, mediator)
    model = np.random.normal(loc=-1, scale=1, size=8)
    noise = model[0] + model[1] * action1 + model[2] * state1 + model[3] * mediator1 + model[4] * action1 * state1 + model[5] * action1 * mediator1 + model[6] * state1 * mediator1 + model[7] * action1 * state1 * mediator1
    noise_rmean = true_rmean + noise
    return noise_rmean    

def true_pa_function(state, action, return_true_pa_one = False):
    state1 = np.copy(state).flatten()
    action1 = np.copy(action).flatten()
    true_pa_one = 1.0  - 2.0 * state1
    true_pa_one = expit(true_pa_one)
    if return_true_pa_one:
        true_pa = true_pa_one
    else:
        true_pa = true_pa_one * action1 + (1 - true_pa_one) * (1 - action1)
    return true_pa

def false_pa_function(state, action):
    action1 = np.copy(action).flatten()
    true_pa_one = true_pa_function(state, action, return_true_pa_one = True)
    noise_pa_one = true_pa_one * np.random.uniform(low=0.6, high=1.1, size=true_pa_one.shape[0]) #1
    noise_pa_one = np.clip(noise_pa_one, a_min=0.01, a_max=.99)
    noise_pa = noise_pa_one * action1 + (1 - noise_pa_one) * (1 - action1)
    return noise_pa

def true_eta_values():
    true_eta_pi = 2.39701065
    true_eta_pi_a0 = 3.67369661
    true_eta_pi_a0star = 4.89616138
    true_eta_G = 7.87840782
    true_eta_a0 = 7.96309271
    return true_eta_pi, true_eta_pi_a0, true_eta_pi_a0star, true_eta_G, true_eta_a0

def false_eta_values():
    true_eta = true_eta_values()
    noise = np.random.normal(loc=1.5, scale=1, size=5)
    noise_eta_pi = true_eta + noise
    return noise_eta_pi