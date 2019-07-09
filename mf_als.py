import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error

class MF():
    
    def __init__(self , rating_matrix, k, lambda_u, lambda_v):        
        self.k = k         
        self.Q = rating_matrix
        self.Q = self.Q - np.mean(self.Q)
        
        self.lambda_x = lambda_u * np.eye(k + 1)
        self.lambda_y = lambda_v * np.eye(k + 1)
        
        self.m, self.n = self.Q.shape
        
        self.X = np.random.rand(self.m, self.k)
        self.Y = np.random.rand(self.n, self.k)
        
        # biases of users and items
        self.beta_x = np.zeros(self.m)
        self.beta_y = np.zeros(self.n)
        
        self.W = rating_matrix > 0
        self.W = self.W.astype(np.float64)

    def ALS(self):
        _Y = np.insert(self.Y, 0, 1, axis = 1)
        for u in range(self.Q.shape[0]):
            Wu = self.W[u]   
            YWu = _Y.T * Wu
            res = np.linalg.solve(np.dot(YWu, _Y) + self.lambda_x, np.dot(YWu, self.Q[u] - self.beta_y))
            self.beta_x[u], self.X[u] = res[0], res[1:]
        
        _X = np.insert(self.X, 0, 1, axis = 1)
        for i in range(self.Q.shape[1]):
            Wi = self.W[:, i]
            XWi = _X.T * Wi
            res = np.linalg.solve(np.dot(XWi,  _X) + self.lambda_y, np.dot(XWi, self.Q[:, i] - self.beta_x))
            self.beta_y[i], self.Y[i] = res[0], res[1:]
            
        pred = np.dot(self.X,  self.Y.T) + self.beta_y + self.beta_x.reshape(-1, 1)
        err = mean_squared_error(self.Q, pred) ** 0.5
        return self.X, self.Y, self.beta_x, self.beta_y, err
            
    
    