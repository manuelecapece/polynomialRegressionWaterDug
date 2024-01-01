import numpy as np
    
def calculate(X, y, theta, l):
    #costFuntion for linear regression
    m = y.shape[0]
    h = np.dot(X,theta)
    squaredErrors = np.power(h-y,2)
    thetaExcludingZero = np.copy(theta)
    thetaExcludingZero[0] = 0 #set theta0 to 0 in order to exclude theta0 from regularization
    J = (1 / (2 * m)) * np.sum(squaredErrors) + (l/(2*m)) * np.sum(np.power(thetaExcludingZero,2))
    return J