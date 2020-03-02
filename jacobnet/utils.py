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


def cost(self, output_array, target):
        """Quadratic cost function"""
        C = 0.5*np.linalg.norm(target - output_array)**2
        return C
    
def cost_prime(self, output_array, target):
        """The derivative of the quadratic cost function"""
        C = target - output_array
        return C