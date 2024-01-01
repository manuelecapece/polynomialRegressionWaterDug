import costFunction as cf
import numpy as np

def train(X, y, alpha, l, theta, numIters):
    #Train a linear regression model whit Gradient Descent
    m = y.shape[0]
    J_history = np.zeros((numIters,1))
    theta_history = np.zeros((numIters,theta.shape[0]))
    for i in range(0, numIters):
        h = X @ theta
        derTheta = (alpha/m) * (np.transpose(X) @ (h-y))
        theta[0] = theta[0] - derTheta[0] #excluding theta0 from regularization
        theta[1:] = theta[1:] * (1 - alpha * (l/m)) - derTheta[1:]
        J_history[i,0] = cf.calculate(X, y, theta, l)
        theta_history[i,:] = np.transpose(theta)
    return theta, J_history, theta_history