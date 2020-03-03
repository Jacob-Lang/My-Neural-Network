"""Neuron class."""

import numpy as np

from jacobnet import utils

class Neuron():
    
    def __init__(self, n_inputs, seed=None):
        ''' Constructor for this class. '''
        
        # size of input array
        self.n_inputs = n_inputs
        # for reproducibility
        if seed != None:
            np.random.seed(seed)
        
        # randomly initialised weights
        self.weights = 1 - 2*np.random.random(n_inputs)
        
        # set initial bias to 0
        self.bias = 0
        # activation function is sigmoid
        self.activation_fn = utils.sigmoid
        
    def forward(self, input_array):
        """Forward propagation of an array through the neuron"""        
        # weighted input
        z = sum(input_array*self.weights + self.bias)
        # activation
        a = self.activation_fn(z)
        return a, z
