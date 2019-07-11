# -*- coding: utf-8 -*-
"""
Created on Wed Jun 12 13:05:05 2019

@author: ahmed
"""
import numpy as np

def computeCost(X,y,theta):
    m = np.size(y)
    
    h = np.dot(X, theta)
    J = (1/(2*m)) * np.sum(np.power((h-y),2))
    
    return J

def gradientDescent(X, y, theta, alpha, iterations):
    m = np.size(y)
    J = []
    for i in range(0,iterations):
        
       h = np.dot(X, theta)
       temp = computeCost(X,y,theta)
       J.append(temp)
       theta = (theta.T-(alpha/m)*np.sum(np.multiply((h-y),X), axis = 0)).T

        
    return theta,J

def predict(X,theta):
    y = np.dot(X,theta)
    return y

def exactSolution(X,y):
    theta = np.dot(np.linalg.pinv(X), y)
    return theta
    
def exactSolution_normal(X,y):
    theta = np.dot(np.dot(np.linalg.pinv(np.dot(X.T,X)), X.T), y)
    return theta