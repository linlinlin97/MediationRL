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
                 seed = 1, test = False):
        self.data = data
        np.random.seed(seed)
        self.parameters = parameters
        self.seed = seed
        self.test = test

    def train(self):
        state = np.copy(self.data['state'])
        action = np.copy(self.data['action']).reshape((-1,1))
        mediator = np.copy(self.data['mediator']).reshape((state.shape[0], -1))

        X = np.hstack([state, action])
        y = mediator
    
        if self.test:
            mask = (np.random.rand(len(y))<=.7)
            X_train = X[mask]
            X_test = X[~mask]
            y_train = y[mask]
            y_test = y[~mask]

            regressor = GridSearchCV(DecisionTreeRegressor(), self.parameters, n_jobs=-1)
            regressor.fit(X=X_train, y=y_train)
            best_params = regressor.best_params_
            print(best_params)
            self.bias_1 = np.mean(y_test.reshape((X_test.shape[0], -1)) - regressor.best_estimator_.predict(X_test).reshape((X_test.shape[0], -1)))
            self.mse_1 = np.mean((y_test.reshape((X_test.shape[0], -1)) - regressor.best_estimator_.predict(X_test).reshape((X_test.shape[0], -1)))**2)
            self.sample_var_1 = np.sum(((y_test.reshape((X_test.shape[0], -1)) -  regressor.best_estimator_.predict(X_test).reshape((X_test.shape[0], -1)))-self.bias_1)**2)/(X_test.shape[0]-1)
            self.std_1 = np.sqrt(self.sample_var_1)
            
            regressor = GridSearchCV(DecisionTreeRegressor(), self.parameters, n_jobs=-1)
            regressor.fit(X=X, y=y)
            best_params = regressor.best_params_
            self.tree_model = DecisionTreeRegressor(random_state=self.seed, max_depth = best_params['max_depth'], splitter = best_params['splitter'])
            self.tree_model.fit(X, y)
            self.bias_2 = np.mean(y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))
            self.mse_2 = np.mean((y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))**2)
            self.sample_var_2 = np.sum(((y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))-self.bias_2)**2)/(X.shape[0]-1)
            self.std_2 = np.sqrt(self.sample_var_2)
        else:
            regressor = GridSearchCV(DecisionTreeRegressor(), self.parameters, n_jobs=-1)
            regressor.fit(X=X, y=y)
            best_params = regressor.best_params_
            print(best_params)

            self.tree_model = DecisionTreeRegressor(random_state=self.seed, max_depth = best_params['max_depth'], splitter = best_params['splitter'])
            self.tree_model.fit(X, y)
            self.bias = np.mean(y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))
            self.sample_var = np.sum(((y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))-self.bias)**2)/(X.shape[0]-1)
            self.std = np.sqrt(self.sample_var)
            #self.std = np.std(y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))



    def get_pm_prediction(self, state, action, mediator = None, random = False):
        N = state.shape[0]
        state1 = state.reshape((N,-1))
        if len(action) == 1 and N > 1:
            action1 = action * np.ones((N,1))
        else:
            action1 = action.reshape((N,-1))
        x = np.hstack([state1, action1])
        pm = self.tree_model.predict(x).reshape((N,))

        if mediator is not None:
            if len(mediator) == 1 and N > 1:
                mediator1 = mediator * np.ones((N,1))
            else:
                mediator1 = mediator.reshape((N,-1))
            pM_SA = np.array([norm.pdf(mediator1[i,:], loc = pm[i], scale = self.std) for i in range(N)]).reshape((-1,))
            return np.clip(pM_SA, a_min = 1e-6, a_max = .9999)
        return pm

    def sample_m(self, state, action, random = True):
        N = state.shape[0]
        state1 = state.reshape((N,-1))
        if len(action) == 1 and N > 1:
            action1 = action * np.ones((N,1))
        else:
            action1 = action.reshape((N,-1))

        x = np.hstack([state1, action1])
        pm = self.tree_model.predict(x).reshape((N,))
        if random:
            pm += np.random.normal(loc = 0, scale = self.std, size = (N,))
        return pm
    

    
    
class RewardLearner():
    def __init__(self, data, parameters = {"splitter":["best","random"], "max_depth" : range(1,20)}, seed = 1, test = False):
        self.data = data
        np.random.seed(seed)
        self.parameters = parameters
        self.seed = seed
        self.test = test


    def train(self):
        state = np.copy(self.data['state'])
        action = np.copy(self.data['action']).reshape((-1,1))
        mediator = np.copy(self.data['mediator']).reshape((state.shape[0], -1))
        reward = np.copy(self.data['reward']).reshape((state.shape[0], -1))

        X = np.hstack([state, action, mediator])
        y = reward
        
        if self.test:
            mask = (np.random.rand(len(y))<=.7)
            X_train = X[mask]
            X_test = X[~mask]
            y_train = y[mask]
            y_test = y[~mask]

            regressor = GridSearchCV(DecisionTreeRegressor(), self.parameters, n_jobs=-1)
            regressor.fit(X=X_train, y=y_train)
            best_params = regressor.best_params_
            print(best_params)
            self.bias_1 = np.mean(y_test.reshape((X_test.shape[0], -1)) - regressor.best_estimator_.predict(X_test).reshape((X_test.shape[0], -1)))
            self.mse_1 = np.mean((y_test.reshape((X_test.shape[0], -1)) - regressor.best_estimator_.predict(X_test).reshape((X_test.shape[0], -1)))**2)
            self.sample_var_1 = np.sum(((y_test.reshape((X_test.shape[0], -1)) -  regressor.best_estimator_.predict(X_test).reshape((X_test.shape[0], -1)))-self.bias_1)**2)/(X_test.shape[0]-1)
            self.std_1 = np.sqrt(self.sample_var_1)
            
            regressor = GridSearchCV(DecisionTreeRegressor(), self.parameters, n_jobs=-1)
            regressor.fit(X=X, y=y)
            best_params = regressor.best_params_
            self.tree_model = DecisionTreeRegressor(random_state=self.seed, max_depth = best_params['max_depth'], splitter = best_params['splitter'])
            self.tree_model.fit(X, y)
            self.bias_2 = np.mean(y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))
            self.mse_2 = np.mean((y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))**2)
            self.sample_var_2 = np.sum(((y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))-self.bias_2)**2)/(X.shape[0]-1)
            self.std_2 = np.sqrt(self.sample_var_2)
        else:
            regressor = GridSearchCV(DecisionTreeRegressor(), self.parameters, n_jobs=-1)
            regressor.fit(X=X, y=y)
            best_params = regressor.best_params_
            print(best_params)

            self.tree_model = DecisionTreeRegressor(random_state=self.seed, max_depth = best_params['max_depth'], splitter = best_params['splitter'])
            self.tree_model.fit(X, y)
            
            self.bias = np.mean(y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))
            self.sample_var = np.sum(((y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))-self.bias)**2)/(X.shape[0]-1)
            self.std = np.sqrt(self.sample_var)
            #self.std = np.std(y.reshape((X.shape[0], -1)) - self.tree_model.predict(X).reshape((X.shape[0], -1)))


    def get_reward_prediction(self, state, action, mediator):
        N = state.shape[0]
        state1 = state.reshape((N,-1))
        if len(action) == 1 and N > 1:
            action1 = action * np.ones((N,1))
        else:
            action1 = action.reshape((N,-1))
        if len(mediator) == 1 and N > 1:
            mediator1 = mediator * np.ones((N,1))
        else:
            mediator1 = mediator.reshape((N,-1))
        x = np.hstack([state1, action1, mediator1])
        rmean = self.tree_model.predict(x).reshape((N,))
        return rmean
    
    
class PALearner():
    def __init__(self, data, parameters = {"splitter":["best","random"], 
                                           "max_depth" : range(1,20)}, 
                 seed = 1, test = False):
        self.data = data
        np.random.seed(seed)
        self.parameters = parameters
        self.seed = seed
        self.test = test

    def train(self):
        state = np.copy(self.data['state'])
        action = np.copy(self.data['action']).reshape((-1,1))

        X = state
        y = action
    
        if self.test:
            mask = (np.random.rand(len(y))<=.7)
            X_train = X[mask]
            X_test = X[~mask]
            y_train = y[mask]
            y_test = y[~mask]

            regressor = GridSearchCV(DecisionTreeClassifier(), self.parameters, n_jobs=-1)
            regressor.fit(X=X_train, y=y_train)
            best_params = regressor.best_params_
            print(best_params)
            self.score = regressor.best_estimator_.score(X_test, y_test)
  
            
            regressor = GridSearchCV(DecisionTreeClassifier(), self.parameters, n_jobs=-1)
            regressor.fit(X=X, y=y)
            best_params = regressor.best_params_
            self.tree_model = DecisionTreeClassifier(random_state=self.seed, max_depth = best_params['max_depth'], splitter = best_params['splitter'])
            self.tree_model.fit(X, y)
            self.prob_A = self.tree_model.predict_proba(X)
            self.bias = np.mean(self.prob_A-.5)
            self.mse = np.mean((self.prob_A-.5)**2)
        else:
            regressor = GridSearchCV(DecisionTreeClassifier(), self.parameters, n_jobs=-1)
            regressor.fit(X=X, y=y)
            best_params = regressor.best_params_
            self.tree_model = DecisionTreeClassifier(random_state=self.seed, max_depth = best_params['max_depth'], splitter = best_params['splitter'])
            self.tree_model.fit(X, y)
            self.prob_A = self.tree_model.predict_proba(X)
            self.bias = np.mean(self.prob_A-.5)
            self.mse = np.mean((self.prob_A-.5)**2)



    def get_pa_prediction(self, state, action):
        N = state.shape[0]
        state1 = state.reshape((N,-1))
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
