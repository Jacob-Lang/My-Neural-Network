"""Utility functions for the jacobnet package."""

import numpy as np

def sigmoid(x):
    """The sigmoid activation function"""
    sig = 1/(1 + np.exp(-x))
    return sig

def sigmoid_prime(x):
    """The derivative of the sigmoid function"""
    sig_prime = sigmoid(x)*(1 - sigmoid(x))
    return sig_prime


def cost(target, output_array):
    """Quadratic cost function"""
    C = 0.5*np.linalg.norm(target - output_array)**2
    return C

def cost_prime(target, output_array):
    """The derivative of the quadratic cost function w.r.t. output_array"""
    C = output_array - target
    return C
    
    
def label_encoder(label_list):
    """One hot encodes the integer class label"""
    target_output = np.zeros((10,len(label_list)))
    
    for li, label in enumerate(label_list):
        target_output[label, li] = 1
        
    return target_output

def label_decoder(output_array):
    """Determines the output array prediction using argmax"""
    return np.argmax(output_array, axis=0)


def train_test_split(X, y, train_split=0.9):
    """Splits a list of training data and labels into train and test sets"""
    N_total = len(X)
    N_train = int(train_split*N_total)
   
    X_train, y_train = X[:N_train], y[:N_train]
    X_test, y_test = X[N_train:], y[N_train:]

    return X_train, y_train, X_test, y_test
