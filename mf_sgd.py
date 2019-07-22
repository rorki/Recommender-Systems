import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class SGD:
    '''Matrix factorization with gradient descent. The dataset is expected in format <item, user, rating>.'''
    
    def __init__(self, dataset, n_users, n_items, n_factors, biased = True, 
                 lr_all=.005, reg_all=.02, lambda_q = 0.1, init_mean=0, init_std_dev=.1):
        self.biased = biased
        self.dataset = dataset
        self.global_mean = np.mean(dataset[:, -1]) if biased else 0

        self.lr_bu = lr_all
        self.lr_bi = lr_all
        self.lr_pu = lr_all
        self.lr_qi = lr_all
        
        self.reg_bu = reg_all
        self.reg_bi = reg_all
        self.reg_pu = reg_all
        self.reg_qi = reg_all
        
        self.lambda_q = lambda_q

        self.bu = np.zeros(n_users, np.double)
        self.bi = np.zeros(n_items, np.double)
        
        self.pu = np.random.normal(init_mean, init_std_dev, (n_users, n_factors))
        self.qi = np.random.normal(init_mean, init_std_dev, (n_items, n_factors))
       
    @staticmethod
    def predict_dataset_with_params(dataset, mu, bu, bi, qi, pu):
        return np.array([mu + bu[u] + bi[i] + np.dot(qi[i], pu[u]) 
                          for i, u, r in dataset])
        
    def predict_dataset(self, trainset):
        '''Make predictions for given dataset using current U, V matricies and biases.'''
        return np.array([self.global_mean + self.bu[u] + self.bi[i] + np.dot(self.qi[i], self.pu[u]) 
                          for i, u, r in trainset])
    
    def predict_matrix(self):
        '''Predict ratings with current U, V matricies and biases. Calculates full matrix n_users x n_items.'''
        return self.global_mean + np.dot(self.pu, self.qi.T) + self.bi + self.bu.reshape(-1, 1) 
    
    def current_error(self):
        '''Return current RMSE and MAE for training set.'''
        preds = self.predict_dataset(self.dataset)
        rmse = mean_squared_error(self.dataset[:, -1], preds) ** 0.5
        mae = mean_absolute_error(self.dataset[:, -1], preds)
        return rmse, mae
    
    def run_epoch(self, qi_cdl = None):
        '''Run training process for one epoch.'''
        for i, u, r in self.dataset:
            # calc error
            dot = np.dot(self.qi[i], self.pu[u])
            err = r - (self.global_mean + self.bu[u] + self.bi[i] + dot)

            # update biases
            if self.biased:
                self.bu[u] += self.lr_bu * (err - self.reg_bu * self.bu[u])
                self.bi[i] += self.lr_bi * (err - self.reg_bi * self.bi[i])

            # calc item and user vector updates
            delta_q = 0
            if qi_cdl is not None:
                delta_q = self.lambda_q * (qi_cdl[i] - self.qi[i])
                
            pu_temp = self.lr_pu * (err * self.qi[i] - self.reg_pu * self.pu[u])
            qi_temp = self.lr_qi * (err * self.pu[u] + delta_q - self.reg_qi * self.qi[i])

            # sim update
            self.pu[u] += pu_temp
            self.qi[i] += qi_temp
            
        return self.global_mean, self.pu, self.qi, self.bu, self.bi
            
    def train(self, qi_cdl = None, n_epochs = 10):
        '''Run training process for given num of epochs.'''
        for current_epoch in range(1, n_epochs + 1):
            print('Epoch %s\%s' % (current_epoch, n_epochs))
            self.run_epoch()
                    
        return self.global_mean, self.pu, self.qi, self.bu, self.bi