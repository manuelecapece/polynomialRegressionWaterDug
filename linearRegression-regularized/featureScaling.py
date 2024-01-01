import numpy as np

def normalize(X):
    #FEATURENORMALIZE return a normalized (X_norm) version of X
    X_norm = np.copy(X)
    r = X.shape[0]
    c = X.shape[1]
    mu = np.mean(X,0)
    sigma = np.std(X,0)
    for i in range(0, c):
        X_norm[:,i] = (X[:,i] - mu[i]) / sigma[i]
    return X_norm, mu, sigma