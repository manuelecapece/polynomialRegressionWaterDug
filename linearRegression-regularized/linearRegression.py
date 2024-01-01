# import library
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
figure_formats = ['svg']
mpl.style.use('ggplot') # for ggplot-like style
# import modules
import costFunction as cf
import gradientDescent as gd
import learningCurve as lc
import featureScaling as fs
import polynomialRegression as pr
import validationCurve as vc


#LOAD DATA
import os
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "..\\waterdug.csv"))
#Adding header
headers = ['x','y']
df.columns = headers
print(df.head())
#Split the data in Train, Cross Validation and Test set
df_train = df.loc[0:11].copy()
df_cv = df.loc[12:32].copy()
df_test = df.loc[33:].copy()

#plot data
df_train.plot(kind='scatter', x='x', y='y', figsize=(10, 6), marker='x', color='red')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.show()

print("Program paused. Press Enter to continue.")
input()

#TRAINING
m_train = df_train.shape[0]
n = df_train.shape[1]-1
y = np.copy(df_train['y']).reshape(m_train, n) 
Xorig = np.copy(df_train['x']).reshape(m_train, n) 
x0 = np.ones((m_train,1))
X = np.hstack((x0,Xorig))

#Learning parameters
alpha = 0.001
iterations = 3000
l = 0 #regularizzation parameter lambda
initial_theta = np.zeros((n+1,1))
result = gd.train(X, y, alpha, l, initial_theta, iterations)
theta = result[0]
J_history = result[1]
theta_history = result[2]

#Plot Cost Function convergence graph
plt.figure(figsize=(10, 6))
plt.plot(J_history, color='black')
plt.xlabel('#iterations')
plt.ylabel('Cost J')
plt.show()

#Plot fit over the data
h = X @ theta
df_train.plot(kind='scatter', x='x', y='y', figsize=(10, 6), marker='x', color='red')
plt.xlabel('Change in water level (x)')
plt.ylabel('Water flowing out of the dam (y)')
plt.plot(Xorig , h, linestyle='--', color='blue')
plt.show()
print('Theta found:\n',theta)

print("Program paused. Press Enter to continue.")
input()

#LEARNING CURVES
m_cv = df_cv.shape[0]
n = df_cv.shape[1]-1
yval = np.copy(df_cv['y']).reshape(m_cv, n) 
XvalOrig = np.copy(df_cv['x']).reshape(m_cv, n) 
x0 = np.ones((m_cv,n))
Xval = np.hstack((x0,XvalOrig))

result = lc.plot(X, y, Xval, yval, alpha, l, iterations,1)
error_train = result[0]
error_val = result[1]
k = len(error_train)
print('# Training Examples\tTrain Error\t\tCross Validation Error')
for i in range(k):
    print('\t{}\t\t{}\t\t{}'.format(i + 1, error_train[i], error_val[i]))

print("Program paused. Press Enter to continue.")
input()

#The model is underfitting the data, and we have a chart with high bias
#We need to use a complex model
    
#POLYNOMIAL REGRESSION

#Learning parameters
alpha = 0.001
iterations = 3000   
l = 0 #regularizzation parameter lambda
maxPol = 10
initial_theta = np.zeros((n+1,1))
result = pr.polynomialPlot(Xorig, y, XvalOrig, yval, maxPol , alpha, l, initial_theta, iterations,1)
error_train = result[0]
error_val = result[1]

k = len(error_train)
print('# Polynomial Degree\tTrain Error\t\tCross Validation Error')
for i in range(k):
    print('\t{}\t\t{}\t\t{}'.format(i + 1, error_train[i], error_val[i]))

print("Program paused. Press Enter to continue.")
input()
    
#best degree seems to be 5
m_test = df_test.shape[0]
Xtest = df_test['x'].to_numpy().reshape(m_test,n)
X = np.copy(Xorig)
Xval = np.copy(XvalOrig)
d = 4 #means polynomial whit degree of 5

#preparing data with features scaling

X_poly = pr.polyFeatures(X, d)
result = fs.normalize(X_poly)
X_poly = result[0]
mu = result[1]
sigma = result[2]
x0 = np.ones((m_train,1))
X_poly = np.hstack((x0,X_poly))

X_poly_val = pr.polyFeatures(Xval, d)
X_poly_val = X_poly_val - mu
X_poly_val = X_poly_val / sigma
x0 = np.ones((X_poly_val.shape[0],1))
X_poly_val = np.hstack((x0,X_poly_val))

X_poly_test = pr.polyFeatures(Xtest, d)
X_poly_test = X_poly_test - mu
X_poly_test = X_poly_test / sigma
x0 = np.ones((X_poly_test.shape[0],1))
X_poly_test = np.hstack((x0,X_poly_test))

#Learning parameters
alpha = 0.001
iterations = 3000   
l = 0 #regularizzation parameter lambda
initial_theta = np.zeros((X_poly.shape[1], 1))
result = gd.train(X_poly, y, alpha, l, initial_theta, iterations)
theta = result[0]
print('Theta found:\n',theta)

max_x = np.max(X,0)
min_x = np.min(X,0)
pr.plotFit(min_x, max_x, mu, sigma, theta, d,df_train)

print("Program paused. Press Enter to continue.")
input()

#LAMBDA AUTOMATIC SELECTION   
alpha = 0.001
iterations = 3000 
result = vc.plot(X_poly, y, X_poly_val, yval, alpha, iterations,1)
lambda_vec = result[0]
error_train = result[1]
error_val = result[2]
k = len(lambda_vec)
print('# \tlambda\t\tTrain Error\t\tCross Validation Error')
for i in range(k):
    print('\t{}\t\t{}\t\t{}'.format(lambda_vec[i], error_train[i], error_val[i]))


#best lambda seems to be the first
l = 0.001

print("Program paused. Press Enter to continue.")
input()

#TRAIN DEFINITIVE MODEL

#Learning parameters
alpha = 0.001
iterations = 3000   
l = 0.001 #best lambda

#Training
result = gd.train(X_poly, y, alpha, l, initial_theta, iterations)
theta = result[0]

#Plot learning curve
result = lc.plot(X_poly, y, X_poly_val, yval, alpha, l, iterations,1)
error_train = result[0]
error_val = result[1]
k = len(error_train)
print('# Training Examples\tTrain Error\tCross Validation Error')
for i in range(k):
    print('\t{}\t\t{}\t\t{}'.format(i + 1, error_train[i], error_val[i]))

#Compute the test error
ytest = df_test['y'].to_numpy().reshape(m_test,1)
error_test = cf.calculate(X_poly_test, ytest, theta, l)
print("Test error: {:.4f}".format(error_test))