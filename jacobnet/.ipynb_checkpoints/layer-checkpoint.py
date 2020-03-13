"""Layer class."""

import numpy as np

from jacobnet import utils
from jacobnet import neuron
    
    
class Layer:
    def __init__(self, n_neurons, n_inputs,  seed=None):
        ''' Constructor for this class. '''
        
        # number of neurons in layer
        self.n_neurons = n_neurons
        # size of input array
        self.n_inputs = n_inputs
        # for reproducibility
        if seed != None:
            np.random.seed(seed)  # this sets the seed for np.random.random in the Neuron class. 
        
        # layer is just a list of neurons
        self.neurons = [neuron.Neuron(n_inputs=self.n_inputs) for n in range(n_neurons)]
        
    def weight_matrix(self):
        """Returns the weight matrix for the layer"""
        # has shape (n_neurons, n_inputs) so that W*a_n + b = z_n+1 
        W = np.zeros((self.n_neurons, self.n_inputs))
        # fill weight matrix by querying neurons
        for ni, nrn in enumerate(self.neurons):
            W[ni,:] = nrn.weights
        
        return W
    
    def bias_vector(self):
        """Returns the bias vector for the layer"""
        b = np.zeros((self.n_neurons,))
        # fill bias vector by querying neurons
        for ni, nrn in enumerate(self.neurons):
            b[ni] = nrn.bias
        
        return b
    
    def forward(self, input_array):
        """Forward propagation of an array through the layer"""
        # activations
        a_array = np.zeros(self.n_neurons)
        # weighted inputs
        z_array = np.zeros(self.n_neurons)
        # use neurons forward function to propagate
        for n_i, nrn in enumerate(self.neurons):
            a_array[n_i], z_array[n_i] = nrn.forward(input_array)
        return a_array, z_array
        