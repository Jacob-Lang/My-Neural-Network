import unittest
import numpy as np

# for unittesting in spyder IDE. 
import sys
sys.path.append("../") 


from jacobnet.layer import Layer
from jacobnet import utils
    
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
