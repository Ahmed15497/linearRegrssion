# -*- coding: utf-8 -*-
"""
Created on Thu Jun 13 15:04:22 2019

@author: ahmed
"""
import numpy as np


def featureScaling(X):
    # compute average
    avg = np.mean(X, axis=0)
    avg = np.reshape(avg,(1,-1))
    stand = np.std(X, axis=0)
    stand = np.reshape(stand,(1,-1))
    X = (X-avg)/stand
    return X,avg,stand

def descaling(X,avg,std):
    X = (std*X)+avg
    return X


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