import numpy as np  
import matplotlib.pyplot as plt 
import pandas as pd  
import sklearn

from sklearn.tree import DecisionTreeRegressor  
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score  
from sklearn.metrics import mean_squared_error

from sklearn.model_selection import GridSearchCV
from scipy.stats import norm

class PMLearner():
    def __init__(self, data, parameters = {"splitter":["best","random"], 
                                           "max_depth" : range(1,20)}, 
                 seed = 1, test = False, dim_mediator = 1, dim_state = 1):
        self.data = data
        np.random.seed(seed)
        self.parameters = parameters
        self.seed = seed
        self.test = test
        self.dim_mediator = dim_mediator
        self.dim_state = dim_state

    def train(self):
        state = np.copy(self.data['state']).reshape((-1,self.dim_state))
        action = np.copy(self.data['action']).reshape((-1,1))
        mediator = np.copy(self.data['mediator']).reshape((-1, self.dim_mediator))
        X = np.hstack([state, action])
        y = mediator
    
        if self.test:
            mask = (np.random.rand(len(y))<=.7)
            X_train = X[mask]
            X_test = X[~mask]
            y_train = y[mask]
            y_test = y[~mask]
            
            self.bias_1, self.mse_1, self.sample_var_1, self.std_1 =  self.fit_test(X_train, y_train, X_test, y_test)
            
            self.bias_2, self.mse_2, self.sample_var_2, self.std_2, self.tree_model =  self.fit(X, y)
        else:
            self.bias, self.mse, self.sample_var, self.std, self.tree_model =  self.fit(X, y)
            
    def fit(self, X, y):
        regressor = dict()
        best_params = dict()
        bias = dict()
        mse = dict()
        sample_var = dict()
        std = dict()
        tree_model = dict()
        for i in range(self.dim_mediator):
            regressor[i] = GridSearchCV(DecisionTreeRegressor(random_state=self.seed), self.parameters, n_jobs=-1)
            regressor[i].fit(X=X, y=y[:,i])
            best_params[i] = regressor[i].best_params_
            #print('mediator'+str(i),best_params[i])
            tree_model[i] = DecisionTreeRegressor(random_state=self.seed, max_depth = best_params[i]['max_depth'], splitter = best_params[i]['splitter'])
            tree_model[i].fit(X, y[:,i])
            y_Xb = y[:,i].reshape((X.shape[0], -1)) - tree_model[i].predict(X).reshape((X.shape[0], -1))
            bias[i] = np.mean(y_Xb)
            mse[i] = np.mean((y_Xb)**2)
            sample_var[i] = np.sum(((y_Xb)-bias[i])**2)/(X.shape[0]-1)
            std[i] = np.sqrt(sample_var[i])
        return bias, mse, sample_var, std, tree_model
    
    def fit_test(self, X_train, y_train, X_test, y_test):
        regressor = dict()
        best_params = dict()
        bias = dict()
        mse = dict()
        sample_var = dict()
        std = dict()
        for i in range(self.dim_mediator):
            regressor[i] = GridSearchCV(DecisionTreeRegressor(random_state=self.seed), self.parameters, n_jobs=-1)
            regressor[i].fit(X=X_train, y=y_train[:,i])
            best_params[i] = regressor[i].best_params_
            #print(best_params[i])
            y_Xb = y_test[:,i].reshape((X_test.shape[0], -1)) - regressor[i].best_estimator_.predict(X_test).reshape((X_test.shape[0], -1))
            bias[i] = np.mean(y_Xb)
            mse[i] = np.mean((y_Xb)**2)
            sample_var[i] = np.sum(((y_Xb)-bias[i])**2)/(X_test.shape[0]-1)
            std[i] = np.sqrt(sample_var[i])
        return bias, mse, sample_var, std
                


    def get_pm_prediction(self, state, action, mediator = None, random = False):
        state1 = state.reshape((-1,self.dim_state))
        N = state1.shape[0]
        if len(action) == 1 and N > 1:
            action1 = action * np.ones((N,1))
        else:
            action1 = action.reshape((N,-1))
        x = np.hstack([state1, action1])
        
        pm = dict()
        for i in range(self.dim_mediator):
            pm[i] = self.tree_model[i].predict(x).reshape((N,))

        if mediator is not None:
            if len(mediator) == self.dim_mediator and N > 1:
                mediator = mediator.reshape((1,self.dim_mediator))
                mediator1 = mediator * np.ones((N,self.dim_mediator))
            else:
                mediator1 = mediator.reshape((N,self.dim_mediator))
            pM_SA = np.ones((N,))
            for i in range(self.dim_mediator):
                pM_SA *= np.array([norm.pdf(mediator1[j,i], loc = pm[i][j], scale = self.std[i]) for j in range(N)]).reshape((-1,))
            return np.clip(pM_SA, a_min = 1e-6, a_max = .9999)
        return pm

    def sample_m(self, state, action, random = True):
        state1 = state.reshape((-1,self.dim_state))
        N = state1.shape[0]
        if len(action) == 1 and N > 1:
            action1 = action * np.ones((N,1))
        else:
            action1 = action.reshape((N,-1))
        x = np.hstack([state1, action1])
        pm = []
        for i in range(self.dim_mediator):
            pm_i = self.tree_model[i].predict(x).reshape((N,1))
            if random:
                pm_i += np.random.normal(loc = 0, scale = self.std[i], size = (N,1))
            pm.append(pm_i)
        pm = np.hstack(pm)
        return pm
    

    
    
class RewardLearner():
    def __init__(self, data, parameters = {"splitter":["best","random"], "max_depth" : range(1,20)}, 
                 seed = 1, test = False, dim_mediator = 1, dim_state = 1):
        self.data = data
        np.random.seed(seed)
        self.parameters = parameters
        self.seed = seed
        self.test = test
        self.dim_mediator = dim_mediator
        self.dim_state = dim_state


    def train(self):
        state = np.copy(self.data['state']).reshape((-1,self.dim_state))
        action = np.copy(self.data['action']).reshape((-1,1))
        mediator = np.copy(self.data['mediator']).reshape((-1, self.dim_mediator))
        reward = np.copy(self.data['reward']).reshape((-1, 1))

        X = np.hstack([state, action, mediator])
        y = reward
        
        if self.test:
            mask = (np.random.rand(len(y))<=.7)
            X_train = X[mask]
            X_test = X[~mask]
            y_train = y[mask]
            y_test = y[~mask]

            regressor = GridSearchCV(DecisionTreeRegressor(random_state=self.seed), self.parameters, n_jobs=-1)
            regressor.fit(X=X_train, y=y_train)
            best_params = regressor.best_params_
            #print(best_params)
            y_Xb = y_test.reshape((X_test.shape[0], -1)) - regressor.best_estimator_.predict(X_test).reshape((X_test.shape[0], -1))
            self.bias_1 = np.mean(y_Xb)
            self.mse_1 = np.mean((y_Xb)**2)
            self.sample_var_1 = np.sum(((y_Xb)-self.bias_1)**2)/(X_test.shape[0]-1)
            self.std_1 = np.sqrt(self.sample_var_1)
            
            regressor = GridSearchCV(DecisionTreeRegressor(random_state=self.seed), self.parameters, n_jobs=-1)
            regressor.fit(X=X, y=y)
            best_params = regressor.best_params_
            self.tree_model = DecisionTreeRegressor(random_state=self.seed, max_depth = best_params['max_depth'], splitter = best_params['splitter'])
            self.tree_model.fit(X, y)
            y_Xb = y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1))
            self.bias_2 = np.mean(y_Xb)
            self.mse_2 = np.mean((y_Xb)**2)
            self.sample_var_2 = np.sum(((y_Xb)-self.bias_2)**2)/(X.shape[0]-1)
            self.std_2 = np.sqrt(self.sample_var_2)
        else:
            regressor = GridSearchCV(DecisionTreeRegressor(random_state=self.seed), self.parameters, n_jobs=-1)
            regressor.fit(X=X, y=y)
            best_params = regressor.best_params_
            #print('reward', best_params)

            self.tree_model = DecisionTreeRegressor(random_state=self.seed, max_depth = best_params['max_depth'], splitter = best_params['splitter'])
            self.tree_model.fit(X, y)
            y_Xb = y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1))
            self.bias = np.mean(y_Xb)
            self.sample_var = np.sum(((y_Xb)-self.bias)**2)/(X.shape[0]-1)
            self.std = np.sqrt(self.sample_var)
            #self.std = np.std(y_Xb)


    def get_reward_prediction(self, state, action, mediator):
        state1 = state.reshape((-1,self.dim_state))
        N = state1.shape[0]
        if len(action) == 1 and N > 1:
            action1 = action * np.ones((N,1))
        else:
            action1 = action.reshape((N,-1))
        if len(mediator) == self.dim_mediator and N > 1:
            mediator = mediator.reshape((1,self.dim_mediator))
            mediator1 = mediator * np.ones((N,self.dim_mediator))
        else:
            mediator1 = mediator.reshape((N,self.dim_mediator))
        x = np.hstack([state1, action1, mediator1])
        rmean = self.tree_model.predict(x).reshape((N,))
        return rmean
    
    
class PALearner():
    def __init__(self, data, parameters = {"splitter":["best","random"], 
                                           "max_depth" : range(1,20)}, 
                 seed = 1, test = False, dim_mediator = 1, dim_state = 1):
        self.data = data
        np.random.seed(seed)
        self.parameters = parameters
        self.seed = seed
        self.test = test
        self.dim_mediator = dim_mediator
        self.dim_state = dim_state

    def train(self):
        state = np.copy(self.data['state']).reshape((-1,self.dim_state))
        action = np.copy(self.data['action']).reshape((-1,1))

        X = state
        y = action
    
        if self.test:
            mask = (np.random.rand(len(y))<=.7)
            X_train = X[mask]
            X_test = X[~mask]
            y_train = y[mask]
            y_test = y[~mask]

            regressor = GridSearchCV(DecisionTreeClassifier(random_state=self.seed), self.parameters, n_jobs=-1)
            regressor.fit(X=X_train, y=y_train)
            best_params = regressor.best_params_
            #print(best_params)
            self.score = regressor.best_estimator_.score(X_test, y_test)
  
            
            regressor = GridSearchCV(DecisionTreeClassifier(random_state=self.seed), self.parameters, n_jobs=-1)
            regressor.fit(X=X, y=y)
            best_params = regressor.best_params_
            self.tree_model = DecisionTreeClassifier(random_state=self.seed, max_depth = best_params['max_depth'], splitter = best_params['splitter'])
            self.tree_model.fit(X, y)
            self.prob_A = self.tree_model.predict_proba(X)
            self.bias = np.mean(self.prob_A-.5)
            self.mse = np.mean((self.prob_A-.5)**2)
        else:
            regressor = GridSearchCV(DecisionTreeClassifier(random_state=self.seed), self.parameters, n_jobs=-1)
            regressor.fit(X=X, y=y)
            best_params = regressor.best_params_
            #print('action', best_params)
            self.tree_model = DecisionTreeClassifier(random_state=self.seed, max_depth = best_params['max_depth'], splitter = best_params['splitter'])
            self.tree_model.fit(X, y)
            self.prob_A = self.tree_model.predict_proba(X)
            self.bias = np.mean(self.prob_A-.5)
            self.mse = np.mean((self.prob_A-.5)**2)



    def get_pa_prediction(self, state, action):
        state1 = state.reshape((-1,self.dim_state))
        N = state1.shape[0]
        if len(action) == 1 and N > 1:
            action1 = action * np.ones((N,1))
        else:
            action1 = action.reshape((N,-1))
        action1 = action1.flatten()
        pa_all = self.tree_model.predict_proba(state)
        if self.tree_model.classes_[0] == 0:
            pA = [pa_all[i,int(action1[i])] for i in range(N)]
        else:
            pA = [pa_all[i,int(1-action1[i])] for i in range(N)]
        
        return np.clip(pA, a_min = 1e-6, a_max = .9999)
