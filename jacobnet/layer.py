"""Layer class."""

import numpy as np
import utils
import neuron
#from jacobnet import utils
#from jacobnet import neuron
    
    
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
        
    
   
#%% Testing 
    
import unittest
    
class TestLayer(unittest.TestCase):

    def setUp(self):
        # example parameters for test
        self.n_neurons = 4
        self.n_inputs = 3
        self.seed = 42
        self.layer = Layer(n_neurons=self.n_neurons, 
                           n_inputs=self.n_inputs, 
                           seed=self.seed)
        
        # two more layers to check reproducibility (to test the seed function)
        self.layer_diff = Layer(n_neurons=self.n_neurons, 
                           n_inputs=self.n_inputs, 
                           seed=None)
        
        self.layer_same = Layer(n_neurons=self.n_neurons, 
                           n_inputs=self.n_inputs, 
                           seed=self.seed)
        
        # example input
        self.test_input = np.ones(self.n_inputs)

    def test_init(self):
        # check layer object is constructed
        self.assertIsInstance(self.layer, Layer)
        
    def test_seed(self):
        # weights of a particular neuron
        w = list(self.layer.neurons[-1].weights)
        w_diff = list(self.layer_diff.neurons[-1].weights)
        w_same = list(self.layer_same.neurons[-1].weights)

        # check weights initialised the same when seed is set
        self.assertSequenceEqual(w, w_same)
        # check they are different when seed is not set
        self.assertNotEqual(w, w_diff)
    
    def test_weight_matrix(self):
        W = self.layer.weight_matrix()
        b = self.layer.bias_vector()
        # check weight matrix is properly filled
        z1 = np.matmul(W, self.test_input) + b
        _, z2 = self.layer.forward(self.test_input)
        
        self.assertEqual(list(z1), list(z2))

    def test_forward(self):
        a_array, z_array = self.layer.forward(self.test_input)
        
        # check output is correct shape
        self.assertEqual(a_array.shape, (self.n_neurons,))
        self.assertEqual(z_array.shape, (self.n_neurons,))

        # check output is numpy array
        self.assertIsInstance(a_array, np.ndarray)
        self.assertIsInstance(z_array, np.ndarray)
        
        # check outputs related as expected
        self.assertSequenceEqual(list(utils.sigmoid(z_array)), list(a_array))

if __name__ == '__main__':
    # run tests
    unittest.main()