# -*- coding: utf-8 -*-
"""
Niels Aalund Krogsgaard
16-10-2021
"""

#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt



#%%
#1)

def square(x):
    return x**2

#i.
x = np.arange(0,6,0.1)
y = square(x)

#ii.
np.random.seed(7)
y_noisy = y + np.random.normal(0, 5, len(y))

#iii.
sns.lineplot(x = x,y = y)
sns.scatterplot(x = x,y = y_noisy, palette = 'red')


#%%
#2)
from sklearn.linear_model import LinearRegression
lin_mod = LinearRegression()

x_re = x.reshape(-1,1)
y_noisy_re = y_noisy.reshape(-1,1)

lin_mod.fit(x_re, y_noisy_re)

#i.
mod_pre = lin_mod.predict(x_re)

sns.lineplot(x = x,y = y)
sns.scatterplot(x = x,y = y_noisy)
plt.plot(x, mod_pre)


#%%
#ii.
from sklearn.preprocessing import PolynomialFeatures
quadratic = PolynomialFeatures(degree=2)
X_quadratic = quadratic.fit_transform(x.reshape(-1, 1))
regressor = LinearRegression()
regressor.fit(X_quadratic, y_noisy_re)
y_quadratic_hat = regressor.predict(X_quadratic)

#what does x_quadratic amount to?
#Well, it is just a numpy array of the shape (60,3) with the first colum being all ones, the next being the all the regular x-values, and the last column being quadratic x's.


#%%
#iii.
from sklearn.preprocessing import PolynomialFeatures
fifth_order = PolynomialFeatures(degree=5)
X_fifth = fifth_order.fit_transform(x.reshape(-1, 1))
fifth_regressor = LinearRegression()
fifth_regressor.fit(X_fifth, y_noisy_re)
y_fifth_hat = fifth_regressor.predict(X_fifth)

sns.lineplot(x = x,y = y)
plt.scatter(x = x,y = y_noisy)
plt.plot(x, mod_pre)
plt.plot(x, y_quadratic_hat)
plt.plot(x, y_fifth_hat)

#%%

np.random.seed(7)

#3)
hund_samp = np.array([y + np.random.normal(0, 5, len(x)) for _ in range(100)])

#i.
y_lin_list = np.zeros([100,60])
y_quad_list = np.zeros([100,60])
y_fifth_list = np.zeros([100,60])
lin_pre_three = np.zeros(100)
quad_pre_three = np.zeros(100)
fifth_pre_three = np.zeros(100)

for i, val in enumerate(hund_samp):
    y_noisy = val
    
    lin_mod = LinearRegression()
    x_re = x.reshape(-1,1)
    y_noisy_re = y_noisy.reshape(-1,1)
    lin_mod.fit(x_re, y_noisy_re)
    y_lin_hat = lin_mod.predict(x_re)
    y_lin_list[i,:] = y_lin_hat[:,0]
    lin_pre_three[i] = lin_mod.predict(np.array(3).reshape(-1,1))
    
    quadratic = PolynomialFeatures(degree=2)
    X_quadratic = quadratic.fit_transform(x.reshape(-1, 1))
    regressor = LinearRegression()
    regressor.fit(X_quadratic, y_noisy_re)
    y_quadratic_hat = regressor.predict(X_quadratic)
    y_quad_list[i,:] = y_quadratic_hat[:,0]
    quad_pre_three[i] = regressor.predict(quadratic.fit_transform(np.array(3).reshape(-1, 1)))
    
    fifth_order = PolynomialFeatures(degree=5)
    X_fifth = fifth_order.fit_transform(x.reshape(-1, 1))
    fifth_regressor = LinearRegression()
    fifth_regressor.fit(X_fifth, y_noisy_re)
    y_fifth_hat = fifth_regressor.predict(X_fifth)
    y_fifth_list[i,:] = y_fifth_hat[:,0]
    fifth_pre_three[i] = fifth_regressor.predict(fifth_order.fit_transform(np.array(3).reshape(-1, 1)))

#%%
#ii.
plt.figure()
for i in range(len(y_lin_list)):
    plt.plot(x, y_lin_list[i,:], color = "red", alpha = 0.1, zorder = 1)
for i in range(len(y_quad_list)):
    plt.plot(x, y_quad_list[i,:], color = "blue", alpha = 0.1, zorder = 2)
plt.scatter(x = 3, y = square(3), c = "lightgreen", edgecolors = "lightgreen", zorder = 3)

#It is definitely the linear model that has the highest bias. The highest variance is also the linear model.

#%%
#iii.
plt.figure()
for i in range(len(y_quad_list)):
    plt.plot(x, y_quad_list[i,:], color = "blue", alpha = 0.1, zorder = 2)
for i in range(len(y_fifth_list)):
    plt.plot(x, y_fifth_list[i,:], color = "green", alpha = 0.1, zorder = 1)
plt.scatter(x = 3, y = square(3), c = "lightgreen", edgecolors = "lightgreen", zorder = 3)
    
#Bias looks like sort of the same for both models, however variance is larger for the fifth order polynomial

#%%
#iv.
#bias for linear
lin_bias = np.mean(square(3) - lin_pre_three)

#bias for quadratic
quad_bias = np.mean(square(3) - quad_pre_three)

#bias for fifth
fifth_bias = np.mean(square(3) - fifth_pre_three)

print(lin_bias, quad_bias, fifth_bias)

#variance for linear
lin_var = np.mean((lin_pre_three - square(3))**2)

#variance for quadratic
quad_var = np.mean((quad_pre_three - square(3))**2)

#variance for fifth order
fifth_var = np.mean((fifth_pre_three - square(3))**2)

print(lin_var, quad_var, fifth_var)











