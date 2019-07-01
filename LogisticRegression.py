import numpy as np

class LogisticRegression:
    def __init__(self, LR=0.01, iterations=1000):
        self.LR = LR
        self.itr = iterations
        self.theta = None
        
    def hyphothesis(self, z):
        return 1 / (1 + np.exp(-z))
    
    def scaleData(self,X):
        j = 0
        arr = np.zeros(X.shape)
        for i in X.columns:
            mean = 0
            temp = X[i]
            mean = np.mean(temp)
            temp =  (temp - mean) / np.std(temp)
            arr[:,j] = temp
            j += 1
        return arr
    
    def loss(self, X, y):
        # calculate X * theta
        z = np.dot(X,self.theta)
        # calculate hypothesis
        h = self.hyphothesis(z)
        # calculate loss
        loss = ((-y * np.log(h) - (1 - y) * np.log(1 - h))).mean()
        return loss
        
    def gradientDescent(self, X, y):
        n = len(X)
        for i in range(self.itr):
            z = np.dot(X, self.theta)
            h = self.hyphothesis(z)
            gradient = np.dot(X.T, (h - y))/ len(X)
            self.theta -= (self.LR * gradient)
        
    def fit(self, X, Y):
        print('Scalling the data')
        X = self.scaleData(X)
        ones = np.ones([X.shape[0],1])
        X = np.concatenate([ones,X],axis=1)
        self.theta = np.zeros(X.shape[1])
        print('Scalling Done')
        
        print('Initial Loss is {}'.format(self.loss(X,Y)))
        print('Running Gradient Descent')
        self.gradientDescent(X,Y)
        print('Loss after gradient descent is {}'.format(self.loss(X,Y)))
    
    def predict(self, X_test):
        X_test = self.scaleData(X_test)
        ones = np.ones([X_test.shape[0],1])
        X_test = np.concatenate([ones,X_test],axis=1)
        
        z = np.dot(X_test, self.theta)
        return self.hyphothesis(z).round()
    
    def score(self, pred, y_test):
        mean = np.mean(y_test)
        actual = np.sum((y_test - mean)**2)
        estimated = np.sum((pred - mean)**2)
        rsq = 1 - (estimated/actual)
        return rsq