# import library
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
figure_formats = ['svg']
mpl.style.use('ggplot') # for ggplot-like style
import gradientDescent as gd
import costFunction as cf

def plot(X, y, Xval, yval, alpha, l, iterations, plotChart):
    m = X.shape[0]
    n = X.shape[1]
    error_train = np.zeros((m, 1))
    error_val = np.zeros((m, 1))
    for i in range(0, m):
        XSubset = np.copy(X[0:i+1, :]).reshape(i+1, n)
        ySubset = np.copy(y[0:i+1]).reshape(i+1,1)
        initial_theta = np.zeros((n,1)) 
        result = gd.train(XSubset, ySubset, alpha, l, initial_theta, iterations)
        theta = result[0]
        error_train[i] = cf.calculate(XSubset, ySubset, theta, l)
        error_val[i] = cf.calculate(Xval, yval, theta, l)
    
    if plotChart == 1:
        plt.figure(figsize=(10, 6))
        plt.title('Learning curve')
        plt.plot(error_train, label='Train', color='blue')
        plt.plot(error_val, label='Cross validation',color='red')
        plt.xlabel('Number of training examples')
        plt.ylabel('Error')
        plt.legend()
        plt.show()

    return error_train, error_val