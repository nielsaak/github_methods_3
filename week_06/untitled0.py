#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  3 10:40:46 2021

@author: nielskrogsgaard
"""

#%%
#1) Do a linear regression based on _x_, _X_ and _y_ below (_y_ as the dependent variable) (Exercise 1.1)  
#    i. find $\hat{\beta}$ and $\hat{y}$ (@ is matrix multiplication)
#    ii. plot a scatter plot of _x_, _y_ and add a line based on $\hat{y}$ (use `plt.plot` after running `import matplotlib.pyplot as plt`)  
#2) Create a model matrix, $X$ that estimates, $\hat\beta$ the means of the three sets of observations below, $y_1, y_2, y_3$ (Exercise 1.2)
#    i. find $\hat\beta$ based on this $X$  
#    ii. Then create an $X$ where the resulting $\hat\beta$ indicates: 1) the difference between the mean of $y_1$ and the mean of $y_2$; 2) the mean of $y_2$; 3) the difference between the mean of $y_3$ and the mean of $y_1$  
#3) Finally, find the F-value for this model (from exercise 1.2.ii) and its degrees of freedom. What is the _p_-value associated with it? (You can import the inverse of the cumulative probability density function `ppf` for _F_ using `from scipy.stats import f` and then run `1 - f.ppf`)
#    i. plot the probability density function `f.pdf` for the correct F-distribution and highlight the _F_-value that you found  
#    ii. how great a percentage of the area of the curve is to right of the highlighted point

import numpy as np
np.random.seed(7) # for reproducibility

x = np.arange(10)
y = 2 * x
y = y.astype(float)
n_samples = len(y)
y += np.random.normal(loc=0, scale=1, size=n_samples)

X = np.zeros(shape=(n_samples, 2))
X[:, 0] = x ** 0
X[:, 1] = x ** 1

beta = np.linalg.pinv(X.T@X)@X.T@y

print(beta)

#%%
import matplotlib.pyplot as plt
y_hat = beta[0] + beta[1]*x
plt.scatter(x, y)
plt.plot(x,y_hat)




#%%
X = np.zeros(shape=(15, 3))
#X[:, 0] = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
X[:, 0] = [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0]
X[:, 1] = [0,0,0,0,0,1,1,1,1,1,0,0,0,0,0]
X[:, 2] = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1] 


y1 = np.array([3, 2, 7, 6, 9])
y2 = np.array([10, 4, 2, 1, -3])
y3 = np.array([15, -2, 0, 0, 3])
y = np.concatenate((y1, y2, y3))

beta = np.linalg.pinv(X.T@X)@X.T@y

print(beta)

x = np.array([1,1,1,1,1,2,2,2,2,2,3,3,3,3,3])
x = x.astype(object)
plt.scatter(x,y)

#%%
X = np.zeros(shape=(15, 3))
X[:, 0] = [1,1,1,1,1,0,0,0,0,0,1,1,1,1,1]
X[:, 1] = [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
X[:, 2] = [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1] 


y1 = np.array([3, 2, 7, 6, 9])
y2 = np.array([10, 4, 2, 1, -3])
y3 = np.array([15, -2, 0, 0, 3])
y = np.concatenate((y1, y2, y3))

beta = np.linalg.pinv(X.T@X)@X.T@y

print(beta)
print(X)
#%%

mu1 = np.mean(y1)
mu2 = np.mean(y2)
mu3 = np.mean(y3)
print(mu1,mu2,mu3)






