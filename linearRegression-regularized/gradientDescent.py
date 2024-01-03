import costFunction as cf
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
figure_formats = ['svg']
mpl.style.use('ggplot') # for ggplot-like style

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

def testConvergence(values):
    dim = np.max(values.shape)
    for i in range(1,dim):
        if values[i] >= values[i-1]:
            test = 0
            return test
    test = 1
    return test

def plotAlphaConvergence(iterations, different_alpha, alpha_possible, X, y, l):
    m = X.shape[0]
    n = X.shape[1]

    for i in range(1,different_alpha):
        alpha_possible[i] = alpha_possible[i-1]*1.3; 

    cost_function_alpha = np.zeros((iterations,different_alpha))
    convergenceTest = np.zeros((different_alpha,1))

    for i in range(0,different_alpha):
        theta_inner = np.zeros((n,1))
        alpha_val = alpha_possible[i]
        result = train(X, y, alpha_val, l, theta_inner, iterations)
        theta = result[0]
        cost_function_alpha[:,i] = result[1].reshape(1,-1)
        cost_val = cost_function_alpha[:,i]
        convergenceTest[i] = testConvergence(cost_val)
    
    #Plot convergence graph
    plt.figure(figsize=(10, 6))
    plt.xlabel('#iterations')
    plt.ylabel('Cost J')
    for i in range(0,different_alpha):
        if convergenceTest[i] == 1:
            alpha_label = str(alpha_possible[i])
            plt.plot(np.arange(1,np.prod(cost_function_alpha[:,i].shape)+1),cost_function_alpha[:,i], label=alpha_label)
    plt.legend()
    plt.show()
    return convergenceTest, alpha_possible