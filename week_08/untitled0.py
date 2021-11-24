#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov 17 10:25:58 2021

@author: nielskrogsgaard
"""

#%%
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os


#%%
"""
# Exercises and objectives

1) Load the magnetoencephalographic recordings and do some initial plots to understand the data  
2) Do logistic regression to classify pairs of PAS-ratings  
3) Do a Support Vector Machine Classification on all four PAS-ratings  

REMEMBER: In your report, make sure to include code that can reproduce the answers requested in the exercises below (__MAKE A KNITTED VERSION__)  
REMEMBER: This is Assignment 3 and will be part of your final portfolio 
"""

"""
# EXERCISE 1 - Load the magnetoencephalographic recordings and do some initial plots to understand the data  

The files `megmag_data.npy` and `pas_vector.npy` can be downloaded here (http://laumollerandersen.org/data_methods_3/megmag_data.npy) and here (http://laumollerandersen.org/data_methods_3/pas_vector.npy)   

1) Load `megmag_data.npy` and call it `data` using `np.load`. You can use `join`, which can be imported from `os.path`, to create paths from different string segments  
"""


#%%
#Exercise 1
#1)

data = np.load("/Users/nielskrogsgaard/Downloads/megmag_data.npy")


#%%
#i.
print(data.shape)

#682 reptitions, 102 sensors and 251 time samples

#%%
#ii.
times = np.arange(-200, 801, 4)

#%%
#iii.
N = len(data[:,0,0])

cov = sum([data[i,:,:] @ data[i,:,:].T for i in range(N)])/N

plt.imshow(cov)

#%%
#cov = []

#for i in range(628):
#    cov.append(data[i,:,:] @ data[i,:,:].T)
#
#cov = sum(cov)/628
#
#plt.imshow(cov)

#%%
#iv
rep_mean = np.mean(data, axis = 0)


#%%
#v.
for i in range(len(rep_mean[:,0])):
    plt.plot(times, rep_mean[i,:])
plt.axvline(0, c = "black")
plt.axhline(0, c = "black")


#%%
#vi.
max_mag = np.unravel_index(np.argmax(rep_mean), rep_mean.shape)
print(max_mag)


#%%
#vii.
for i in range(len(data[:,73,0])):
    plt.plot(times, data[i,73,:])
plt.axvline(times[112], c = "black")


#viii.
"""
We found that the sensor, who had the highest average activation for all repititions was the
sensor 73 at 248 ms.  However, as can be seen in the plot of all repetitions for that particular
sensor, there is a large amount of variation or noise throughout. It would be hard to visually detect a signal or difference between the individual repetitions in this plot.
"""



#%%
#2)
y = np.load("/Users/nielskrogsgaard/Downloads/pas_vector.npy")

#i.
print(y.shape)
print(data.shape)

#it has the same length as the first dimnesion in the data array, which is the repetition dimension.

#%%
#ii.
idx_1 = np.argwhere(y == 1)
data_pas_1 = np.mean(np.squeeze(data[idx_1,73,:]), axis = 0)
idx_2 = np.argwhere(y == 2)
data_pas_2 = np.mean(np.squeeze(data[idx_2,73,:]), axis = 0)
idx_3 = np.argwhere(y == 3)
data_pas_3 = np.mean(np.squeeze(data[idx_3,73,:]), axis = 0)
idx_4 = np.argwhere(y == 4)
data_pas_4 = np.mean(np.squeeze(data[idx_4,73,:]), axis = 0)

plt.plot(times, data_pas_1, label = "pas1")
plt.plot(times, data_pas_2, label = "pas2")
plt.plot(times, data_pas_3, label = "pas3")
plt.plot(times, data_pas_4, label = "pas4")
plt.axvline(0, c = "black")
plt.axhline(0, c = "black")
plt.legend()

#%%
#iii.
'''At a pas score of 1 the subject did not subjectively perceive anything, so it might make sense that the amplitude
is lower, since we would not expect much activation, when "nothing" is there to be perceived or processed.
Pas 2 is the one with the highest average activation around time 250 ms. This is somewhat surprising since this would also entail
that the person had not seen the stimuli clearly. But perhaps this could be rationalised by thinking of it
as the brain have to process the uncertainty of the perceived stimuli, while this is not needed with higher Pas ratings.
Pas 4 has the highest amplitude before 200 ms, which could come from early visual processing which is more activated 
with a more clear perceived stimulus.'''




#%%
#Exercise 2
#1) 
# i.
idx = np.argwhere((y == 1) | (y == 2))
data_1_2 = np.squeeze(data[idx,:,:])
y_1_2 = y[idx]

#%%
#ii.
data_1_2_1 = data_1_2.reshape(214,-1)


#%%
#iii.
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
data_1_2_2 = scaler.fit_transform(data_1_2_1)

#%%
#iv.
from sklearn.linear_model import LogisticRegression

log_model = LogisticRegression(penalty='none', random_state=(12))
log_model.fit(data_1_2_2, y_1_2[:,0])

#%%
#v.
accuracy = log_model.score(data_1_2_2, y_1_2[:,0])
print(accuracy)

'''The accuracy is at 100%, which does seem a bit troubeling. It could indicate overfitting, but
we do not know since the data is not divided into training and test. Also, adding no penalty makes
the risk of overfitting much higher.'''


#%%
#vi.
#for the penalty l1 to work, we need to change solver to be liblinear or saga.
log_model_l1 = LogisticRegression(penalty='l1', solver = 'liblinear', random_state=(12))
log_model_l1.fit(data_1_2_2, y_1_2[:,0])
print(len(log_model_l1.coef_[0]) - sum(log_model_l1.coef_[0] == 0))

#270 coefficients are non-zero after applying the l1 penalty

#%%
#fix stuff
#vii.
#the reduced X that I have to make... Is that the data or the coefficients? 
idx = np.where(log_model_l1.coef_[0] != 0)
data_1_2_3 = np.squeeze(data_1_2_2[:, idx])

#N = len(data_1_2_3[:,0])
#cov = np.sum([data_1_2_3[i,:] @ data_1_2_3[i,:].T for i in range(N)])/N
#cov = np.sum([np.dot(data_1_2_3[i,:].reshape(1,-1), data_1_2_3[i,:].reshape(1,-1).T) for i in range(N)])/N
#N = len(data[:,0,0])
#cov = sum([data[i,:,:] @ data[i,:,:].T for i in range(N)])/N
cov = np.cov(data_1_2_3)
#cov = data_1_2_3.T @ data_1_2_3

plt.imshow(cov)

"""Since there are more darker colours, I would say that there is much less covariance in this plot
compared to the one from 1.1.iii"""

#%%
#2)
#i.
from sklearn.model_selection import cross_val_score, StratifiedKFold

#%%
#ii.
unique, counts = np.unique(y_1_2, return_counts=True)
un_co = dict(zip(unique,counts))
print(un_co)

#%%
def equalize_targets_binary(data, y):
    np.random.seed(7)
    targets = np.unique(y) ## find the number of targets
    if len(targets) > 2:
        raise NameError("can't have more than two targets")
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target)) ## find the number of each target
        indices.append(np.where(y == target)[0]) ## find their indices
    min_count = np.min(counts)
    # randomly choose trials
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count,replace=False)
    
    # create the new data sets
    new_indices = np.concatenate((first_choice, second_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    
    return new_data, new_y
#%%
X_1_2_equal, y_1_2_equal = equalize_targets_binary(data_1_2, y_1_2)

X_1_2_equal = X_1_2_equal.reshape(198, -1)
#scaling
scaler = StandardScaler()
X_1_2_equal_1 = scaler.fit_transform(X_1_2_equal)

#%%
#iii.

log_model_skf = LogisticRegression(penalty='none')

skf = StratifiedKFold(n_splits = 5)
skf_accu = []

for train, test in skf.split(X_1_2_equal_1, y_1_2_equal[:,0]):
    x_train, x_test = X_1_2_equal_1[train,:], X_1_2_equal_1[test,:]
    y_train, y_test = y_1_2_equal[:,0][train], y_1_2_equal[:,0][test]
    log_model_skf.fit(x_train,y_train)
    skf_accu.append(log_model_skf.score(x_test, y_test))

print(skf_accu)

#%%
log_model_cv = LogisticRegression(penalty='none')
cv_score = cross_val_score(log_model_cv, X_1_2_equal_1, y_1_2_equal[:,0], cv=5)
print(cv_score)

#%%
#iv.
log_model_cv_l2 = LogisticRegression(penalty='l2', C = 1e5)
cv_score = cross_val_score(log_model_cv_l2, X_1_2_equal_1, y_1_2_equal[:,0], cv=5)
print(cv_score, np.mean(cv_score))

log_model_cv_l2 = LogisticRegression(penalty='l2', C = 1e1)
cv_score = cross_val_score(log_model_cv_l2, X_1_2_equal_1, y_1_2_equal[:,0], cv=5)
print(cv_score, np.mean(cv_score))

log_model_cv_l2 = LogisticRegression(penalty='l2', C = 1e-5)
cv_score = cross_val_score(log_model_cv_l2, X_1_2_equal_1, y_1_2_equal[:,0], cv=5)
print(cv_score, np.mean(cv_score))

#is cross validated the mean of those? 

#%%
#v.
X_1_2_equal, y_1_2_equal = equalize_targets_binary(data_1_2, y_1_2)
log_model_cv_time_indi = LogisticRegression(penalty='l2', C = 1e-5)

cv_scores = np.zeros((X_1_2_equal.shape[2], 5))
mean_cv_scores = np.zeros(X_1_2_equal.shape[2])

for i in range(X_1_2_equal.shape[2]):
    x_data = X_1_2_equal[:,:,i]
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    cv_scores[i,:] = cross_val_score(log_model_cv_time_indi, x_data[:,:], y_1_2_equal[:,0], cv=5)
    mean_cv_scores[i] = np.mean(cv_scores[i,:])

print("Binary classification of PAS rating (1 or 2) is best at", times[np.argmax(mean_cv_scores)], "ms")
#%%
plt.plot(times, mean_cv_scores)
plt.axhline(y = 0.5, c = "black")



#%%
#vi.
X_1_2_equal, y_1_2_equal = equalize_targets_binary(data_1_2, y_1_2)
log_model_cv_time_indi = LogisticRegression(penalty='l1', C = 1e-1, solver = "liblinear")

cv_scores = np.zeros((X_1_2_equal.shape[2], 5))
mean_cv_scores = np.zeros(X_1_2_equal.shape[2])

for i in range(X_1_2_equal.shape[2]):
    x_data = X_1_2_equal[:,:,i]
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    cv_scores[i,:] = cross_val_score(log_model_cv_time_indi, x_data[:,:], y_1_2_equal[:,0], cv=5)
    mean_cv_scores[i] = np.mean(cv_scores[i,:])

print("Binary classification of PAS rating (1 or 2) is best at", times[np.argmax(mean_cv_scores)], "ms")


#%%
plt.plot(times, mean_cv_scores)
plt.axhline(y = 0.5, c = "black")

#%%
#vii.
idx = np.argwhere((y == 1) | (y == 4))
data_1_4 = np.squeeze(data[idx,:,:])
y_1_4 = y[idx]

X_1_4_equal, y_1_4_equal = equalize_targets_binary(data_1_4, y_1_4)
log_model_cv_time_indi = LogisticRegression(penalty='l1', C = 1e-1, solver = "liblinear")

cv_scores = np.zeros((X_1_4_equal.shape[2], 5))
mean_cv_scores = np.zeros(X_1_4_equal.shape[2])

for i in range(X_1_4_equal.shape[2]):
    x_data = X_1_4_equal[:,:,i]
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    cv_scores[i,:] = cross_val_score(log_model_cv_time_indi, x_data[:,:], y_1_4_equal[:,0], cv=5)
    mean_cv_scores[i] = np.mean(cv_scores[i,:])

print("Binary classification of PAS rating (1 or 4) is best at", times[np.argmax(mean_cv_scores)], "ms")


#%%
plt.plot(times, mean_cv_scores)
plt.axhline(y = 0.5, c = "black")


#%%
#3
"""Well, it is possible to achieve accuracies way above chance, however they are not that high and only at quite specific timestamps (After 200 ms, before 250 ms).
Quite surprising that classifying Pas 1 vs 4 has basically the same maximum accuracy as classifying PAS 1 vs 2."""


#%%
#Exercise 3
#1
#i. 
def equalize_targets(data, y):
    np.random.seed(7)
    targets = np.unique(y)
    counts = list()
    indices = list()
    for target in targets:
        counts.append(np.sum(y == target))
        indices.append(np.where(y == target)[0])
    min_count = np.min(counts)
    first_choice = np.random.choice(indices[0], size=min_count, replace=False)
    second_choice = np.random.choice(indices[1], size=min_count, replace=False)
    third_choice = np.random.choice(indices[2], size=min_count, replace=False)
    fourth_choice = np.random.choice(indices[3], size=min_count, replace=False)
    
    new_indices = np.concatenate((first_choice, second_choice,
                                 third_choice, fourth_choice))
    new_y = y[new_indices]
    new_data = data[new_indices, :, :]
    
    return new_data, new_y


#%%
data_equal, y_equal = equalize_targets(data, y)

#%%
#ii.
# =============================================================================
# from sklearn.svm import SVC
# svc_lin = SVC(kernel = "linear")
# svc_rbf = SVC(kernel = "rbf")
# 
# svc_lin.fit(data_equal.reshape(data_equal.shape[0],-1), y_equal)
# svc_rbf.fit(data_equal.reshape(data_equal.shape[0],-1), y_equal)
# 
# print("Mean accuracy of SVM with linear kernel is",svc_lin.score(data_equal.reshape(data_equal.shape[0],-1), y_equal), ". Mean accuracy of SVM with RBF kernel is",svc_rbf.score(data_equal.reshape(data_equal.shape[0],-1), y_equal))
# 
# =============================================================================

from sklearn.svm import SVC
svc_lin = SVC(kernel = "linear")
svc_rbf = SVC(kernel = "rbf")

x_data = data_equal.reshape(data_equal.shape[0],-1)
scaler = StandardScaler()
x_data = scaler.fit_transform(x_data)

svc_lin.fit(x_data, y_equal)
svc_rbf.fit(x_data, y_equal)

print("Mean accuracy of SVM with linear kernel is",svc_lin.score(x_data, y_equal), ". Mean accuracy of SVM with RBF kernel is",svc_rbf.score(x_data, y_equal))


#%%
#iii.

svc_lin = SVC(kernel = "linear")

cv_scores = np.zeros((data_equal.shape[2], 5))
mean_cv_scores = np.zeros(data_equal.shape[2])

for i in range(data_equal.shape[2]):
    x_data = data_equal[:,:,i]
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    cv_scores[i,:] = cross_val_score(svc_lin, x_data[:,:], y_equal, cv=5)
    mean_cv_scores[i] = np.mean(cv_scores[i,:])


plt.plot(times, mean_cv_scores)
plt.axhline(y = 0.25, c = "black")

#%%
svc_rbf = SVC(kernel = "rbf")

cv_scores = np.zeros((data_equal.shape[2], 5))
mean_cv_scores = np.zeros(data_equal.shape[2])

for i in range(data_equal.shape[2]):
    x_data = data_equal[:,:,i]
    scaler = StandardScaler()
    x_data = scaler.fit_transform(x_data)
    cv_scores[i,:] = cross_val_score(svc_rbf, x_data[:,:], y_equal, cv=5)
    mean_cv_scores[i] = np.mean(cv_scores[i,:])


plt.plot(times, mean_cv_scores)
plt.axhline(y = 0.25, c = "black")


#%%
#iv.
#I am unsure whether to use linear or rbf. Check up on that. However, around 200-250 ms they both have about 34% accuracy. Sooo not exactly possible to classify subjective experience.


#%%
#2

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(data_equal, y_equal, test_size = 0.3, random_state = 12)



#%%
data_equal_reshaped = data_equal.reshape(data_equal.shape[0],-1)
scaler = StandardScaler()
data_equal_re_trans = scaler.fit_transform(data_equal_reshaped)
x_train_trans, x_test_trans, y_train_trans, y_test_trans = train_test_split(data_equal_re_trans, y_equal, test_size = 0.3, random_state = 12)


#%%
#i.
from sklearn.svm import SVC
svc_lin = SVC(kernel = "linear")
svc_rbf = SVC(kernel = "rbf")

x_train = x_train.reshape(x_train.shape[0], -1)
x_test = x_test.reshape(x_test.shape[0], -1)

svc_lin.fit(x_train, y_train)
svc_rbf.fit(x_train, y_train)

y_pred_lin = svc_lin.predict(x_test)
y_pred_rbf = svc_rbf.predict(x_test)

#%%
#with standardised data (linear kernel performs very well with standardised data)
svc_lin = SVC(kernel = "linear")
svc_rbf = SVC(kernel = "rbf")

svc_lin.fit(x_train_trans, y_train_trans)
svc_rbf.fit(x_train_trans, y_train_trans)

y_pred_lin_std = svc_lin.predict(x_test_trans)
y_pred_rbf_std = svc_rbf.predict(x_test_trans)

#%%
#ii.
#for some reason, the linear kernel completely disregards the possibilities of being pas 2 or 3. I do not know why, and I would love to visualise it
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_lin)
plt.show()
#%%
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test, y_pred_rbf)
plt.show()


#%%
#this should be the one that I go with, since the model with a linear kernel performed best on standardised data 
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test_trans, y_pred_lin_std)
plt.show()


#%%
from sklearn.metrics import ConfusionMatrixDisplay
ConfusionMatrixDisplay.from_predictions(y_test_trans, y_pred_rbf_std)
plt.show()



#%%
#iii.
#comments for confusion matrix of SVC with linear kernel trained on standardised data
"""The classifier seems to be biased towards PAS 1 and 2. It predicts 57.14% the test data to be either PAS 1 or 2, 
but the testdata only contain 42.86% PAS 1 and 2. You could also say that the classifier is biased away
from PAS 3, since 15.97% were predicted to be PAS 3, while 30.25% actually were PAS 3."""



#%%

#%%






