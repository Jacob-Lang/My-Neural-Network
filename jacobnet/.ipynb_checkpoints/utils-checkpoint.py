import numpy as np

def sigmoid(x):
    """The sigmoid actvation function"""
    sig = 1/(1 + np.exp(-x))
    return sig

def sigmoid_prime(x):
    """The derivative of the sigmoid function"""
    sig_prime = sigmoid(x)*(1 - sigmoid(x))
    return sig_prime