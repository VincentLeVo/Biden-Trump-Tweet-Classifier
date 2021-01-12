import numpy as np

class NaiveBayes:

    def __init__(self):
        pass

    def fit(self, X, y):
        N, D = X.shape

        C = 2 #Because is there is 2 classes (Binary)
        # Compute the probability of each class i.e p(y==c)
        counts = np.bincount(y)
        p_y = counts / N

        #Dataset of mean, variance, sd
        mu = np.zeros((C,D))
        variances = np.zeros((C,D))
        sd = np.zeros((C,D))

        for d in range(D):
            for c in range(C):
                n_c = counts[c] # Number of y values that have class c
                mu[c,d] = np.mean(X[y==c,d]) #Mean of the dth coloum with class c
                variances[c,d] = np.var(X[y==c,d]) # variance ___
                sd[c,d] = np.sqrt(variances[c,d])

        self.variances = variances
        self.mu = mu
        self.sd = sd
        self.p_y = p_y

    def predict(self, X):
        N, D = X.shape
        p_y = self.p_y
        mu = self.mu
        variances = self.variances
        sd = self.sd
        # p_xy = self.p_xy
        # p_y = self.p_y

        y_pred = np.zeros(N)

        for n in range(N):
            probs = p_y.copy()
            for d in range(D):
                if X[n, d] != 0:
                    probs += ((0.5*((X[n,d]-mu[:,d])/sd[:,d])**2)+np.log(sd[:,d]*np.sqrt(2*np.pi)))
                else:
                    probs += 1-(1/2*((X[n,d] - mu[:,d])/(sd[:,d]))**2 + np.log(np.sqrt(sd[:,d])*np.sqrt(2*np.pi)))
            probs = -1 * probs
            y_pred[n] = np.argmax(probs)

        return y_pred
