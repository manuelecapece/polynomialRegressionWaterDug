# import library
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
figure_formats = ['svg']
mpl.style.use('ggplot') # for ggplot-like style
import gradientDescent as gd
import costFunction as cf

def plot(X, y, Xval, yval, alpha, iterations,plotChart):
    #VALIDATIONCURVE Generate the train and validation errors needed to
    #plot a validation curve that we can use to select lambda
    lambda_vec = np.array([0, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10]).reshape(-1,1)

    error_train = np.zeros((len(lambda_vec), 1))
    error_val = np.zeros((len(lambda_vec), 1))

    for i in range(0,len(lambda_vec)):
        l = lambda_vec[i]
        initial_theta = np.zeros((X.shape[1], 1))
        result = gd.train(X, y, alpha, l, initial_theta, iterations)
        theta = result[0]
        error_train[i] = cf.calculate(X, y, theta, l)
        error_val[i] = cf.calculate(Xval, yval, theta, l)
    if plotChart == 1:
        plt.figure(figsize=(10, 6))
        plt.title('Validation curve')
        plt.plot(lambda_vec, error_train, label='Train', color='blue')
        plt.plot(lambda_vec, error_val, label='Cross validation',color='red')
        plt.xlabel('Lambda')
        plt.ylabel('Error')
        plt.legend()
        plt.show()
    return lambda_vec, error_train, error_val