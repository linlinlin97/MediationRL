import numpy as np

class QModel:
    def __init__(self, func, typ, seed = 1):
        self.q_model = func
        self.seed = seed
        self.typ = typ
        pass

    def get_q_prediction(self, state, action, mediator):
        np.random.seed(self.seed)
        return self.q_model(state, action, mediator, self.typ)

class RatioModel:
    def __init__(self, func, seed=1):
        self.w_model = func
        self.seed = seed
        pass

    def get_w_prediction(self, state, policy):
        np.random.seed(self.seed)
        return self.w_model(state, policy)

class PMModel:
    def __init__(self, func, seed = 1):
        self.pm_model = func
        self.seed = seed
        pass

    def get_pm_prediction(self, state, action, mediator):
        np.random.seed(self.seed)
        return self.pm_model(state, action, mediator)

class RewardModel:
    def __init__(self, func, seed = 1):
        self.reward_model = func
        self.seed = seed
        pass

    def get_reward_prediction(self, state, action, mediator):
        np.random.seed(self.seed)
        return self.reward_model(state, action, mediator)


class PAModel:
    def __init__(self, func, seed = 1):
        self.pa_model = func
        self.seed = seed
        pass

    def get_pa_prediction(self, state, action):
        np.random.seed(self.seed)
        return self.pa_model(state, action)
    

class EtaModel:
    def __init__(self, func, seed = 1):
        self.eta_model = func
        self.seed = seed
        pass

    def get_eta_prediction(self):
        np.random.seed(self.seed)
        return self.eta_model() #return a list[et_a0, eta_pi, eta_pia0, eta_pia0star]
    
    
