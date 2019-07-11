# -*- coding: utf-8 -*-


"""
Implementing univarient linear regression algorithm
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from functions import *


file_data = 'ex1data1.txt'
dataset = pd.read_csv(file_data,sep = ',' , header = None, names = ['Area', 'Price'])
del file_data


X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
X = np.reshape(X,(np.size(X),1))
y = np.reshape(y,(np.size(y),1))

m = np.size(y)

free_term = np.ones((m,1))

X = np.concatenate((free_term,X), axis = 1)

del free_term

theta = np.zeros((2, 1)); # initialize fitting parameters
iterations = 1500;
alpha = 0.01;
(theta,J) = gradientDescent(X, y, theta, alpha, iterations)

test = predict([1, 3.5],theta)

exact_theta = exactSolution(X, y)
exact_theta_2 = exactSolution_normal(X, y)

