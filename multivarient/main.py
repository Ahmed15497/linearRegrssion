# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:03:49 2019

@author: ahmed
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *

file_data = 'ex1data2.txt'
dataset = pd.read_csv(file_data,sep = ',' , header = None)
del file_data

X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
y = np.reshape(y,(np.size(y),1))

m = np.size(y)

# perform feature scaling
(X_scaled,X_avg,X_std) = featureScaling(X)
(y_scaled,y_avg,y_std) = featureScaling(y)

# adding the free term
free_term = np.ones((m,1))
X_scaled = np.concatenate((free_term,X_scaled), axis = 1)
del free_term


theta = np.zeros((np.size(X_scaled,axis=1), 1)) # initialize fitting parameters
iterations = 15000
alpha = 0.01

J = computeCost(X_scaled, y_scaled, theta)

(theta,J) = gradientDescent(X_scaled, y_scaled, theta, alpha, iterations)

target = predict(X_scaled, theta)

target = descaling(target, y_avg, y_std)




