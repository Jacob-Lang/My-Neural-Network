"""Utility functions for the jacobnet package."""

import numpy as np

def sigmoid(x):
    """The sigmoid actvation function"""
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