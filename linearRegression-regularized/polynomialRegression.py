# import library
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
figure_formats = ['svg']
mpl.style.use('ggplot') # for ggplot-like style
import gradientDescent as gd
import costFunction as cf
import learningCurve as lc
import featureScaling as fs

def polyFeatures(X, p):
    # %POLYFEATURES Maps X (1D vector) into the p-th power
    # [X_poly] = POLYFEATURES(X, p) takes a data matrix X (size m x 1) and
    # maps each example into its polynomial features where
    # X_poly(i, :) = [X(i) X(i).^2 X(i).^3 ...  X(i).^p];
    X_poly = np.zeros((np.size(X), p))
    m = X.shape[0]
    powers = np.tile(np.arange(1,p+1),(m,1))
    Xrep = np.tile(X,(1,p))
    Xpoly = X ** powers
    return Xpoly

def polynomialPlot(X, y, Xval, yval, maxDegree , alpha, l, initial_theta, iterations, plotChart):
    #POLYNOMIALPLOT Generates the train and cross validation set errors for different polynomial degrees. 
    error_train = np.zeros((maxDegree,1))
    error_val = np.zeros((maxDegree,1)) 
    m = X.shape[0]
    
    for d in range(0, maxDegree):
        X_poly = polyFeatures(X, d)
        result = fs.normalize(X_poly)
        X_poly = result[0]
        mu = result[1]
        sigma = result[2]
        x0 = np.ones((m,1))
        X_poly = np.hstack((x0,X_poly))

        X_poly_val = polyFeatures(Xval, d)
        X_poly_val = X_poly_val - mu
        X_poly_val = X_poly_val / sigma
        x0 = np.ones((X_poly_val.shape[0],1))
        X_poly_val = np.hstack((x0,X_poly_val))

        initial_theta = np.zeros((X_poly_val.shape[1], 1))
        result = gd.train(X_poly, y, alpha, l, initial_theta, iterations)
        theta = result[0]
        error_train[d] = cf.calculate(X_poly, y, theta, l)
        error_val[d] = cf.calculate(X_poly_val, yval, theta, l)
    if plotChart == 1:
        polyDegrees = np.arange(1 , maxDegree+1)
        plt.figure(figsize=(10, 6))
        plt.title('Learning curve')
        plt.plot(polyDegrees,error_train, label='Train', color='blue')
        plt.plot(polyDegrees,error_val, label='Cross validation',color='red')
        plt.xlabel('Polynomial degree')
        plt.ylim(top=25)
        plt.ylabel('Error')
        plt.legend()
        plt.show()
    return error_train, error_val

def plotFit(min_x, max_x, mu, sigma, theta, p, df_train):
    #plots the learned polynomial fit with power p and feature normalization (mu, sigma).
    x = np.arange(min_x - 15 , max_x + 25 , 0.05).reshape(-1,1)

    X_poly = polyFeatures(x, p)
    X_poly = X_poly - mu
    X_poly = X_poly / sigma
    x0 = np.ones((X_poly.shape[0],1))
    X_poly = np.hstack((x0,X_poly))
    h = X_poly @ theta

    df_train.plot(kind='scatter', x='x', y='y', figsize=(10, 6), marker='x', color='red')
    plt.title('Polynomial Regression Fit (d = '+str(p)+')')
    plt.xlabel('Change in water level (x)')
    plt.ylabel('Water flowing out of the dam (y)')
    plt.plot(x, h, linestyle='--', color='blue')

    plt.show()