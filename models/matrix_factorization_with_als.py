import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MF():
    
    def __init__(self , rating_matrix, k, lambda_u, lambda_v):        
        self.k = k         
        self.Q = rating_matrix
        self.Q = self.Q - np.mean(self.Q)
        self.non_zero_idx = self.Q > 0
        
        self.W = rating_matrix > 0.5
        self.W = self.W.astype(np.float64, copy=False)
        
        self.lambda_x = lambda_u * np.eye(k + 1)
        self.lambda_y = lambda_v * np.eye(k + 1)
        
        self.n_factors = k
        self.m, self.n = self.Q.shape
        
        self.X = 5 * np.random.rand(self.m, self.n_factors)
        self.Y = 5 * np.random.rand(self.n_factors, self.n)
        
        # biases of users and items
        self.beta_x = np.zeros(self.m)
        self.beta_y = np.zeros(self.n)
        
        #self.U = np.mat(np.random.normal(0 , 1/self.u_lambda , size=(self.k, self.num_u)))
        #self.V = np.mat(np.random.normal(0 , 1/self.v_lambda , size=(self.k, self.num_v)))
            
    
    def ALS_v3_weighted(self, V_sdae):
        Y_ = np.insert(self.Y, 0, 1, axis = 0)
        
        for u in range(self.W.shape[0]):
            Wu = self.W[u]   
            c = Y_ * Wu
            
            a = np.matmul(c, np.transpose(self.Q[u] - self.beta_y))
            d = np.matmul(c, np.transpose(Y_)) + self.lambda_x
            self.beta_x[u], *self.X[u] = np.linalg.solve(d, a).T
        
        X_ = np.insert(self.X, 0, 1, axis = 1)
        for i in range(self.W.shape[1]):
            Wi = self.W.T[i]
            c = X_.T * Wi
            
            a = np.matmul(c, self.Q[:, i] - self.beta_x) 
            #b = np.matmul(self.lambda_y, np.insert(V_sdae[i], 0, 0, axis = 0))
            
            self.beta_y[i], *self.Y[:,i] = np.linalg.solve(np.matmul(c, X_) + self.lambda_y, a) #+ b)
            
        preds = np.dot(self.X, self.Y) 
        preds += self.beta_x.reshape(-1, 1)
        preds += self.beta_y.reshape(1, -1)
        err = mean_squared_error(self.Q[self.non_zero_idx], preds[self.non_zero_idx]) ** 0.5
        
        return self.X, self.Y, self.beta_x, self.beta_y, err