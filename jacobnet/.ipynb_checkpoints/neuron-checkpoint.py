import numpy as np
from jacobnet import utils

class Neuron():
    
    def __init__(self, n_inputs, seed=None):
        ''' Constructor for this class. '''
        
        self.n_inputs = n_inputs 
        if seed != None:
            np.random.seed(seed)
        self.weights = np.random.random(n_inputs)
        self.bias = 0
        self.activation = utils.sigmoid
        
    def forward(self, input_array):
        """Forward propagator"""        
        # weighted input
        z = sum(input_array*self.weights + self.bias)
        # activation
        a = self.activation(z)
        return a
    
    # methods - set w,b,sigma - get w,b,sigma
    
    # propagate backward (or store parts in training mode)
    # update weights
    
    
    
#%% Testing 
    
import unittest
    
class TestNeuron(unittest.TestCase):

    def setUp(self):
        self.n_inputs = 3
        self.seed = 42
        self.neuron = Neuron(n_inputs=self.n_inputs, seed=self.seed)
        
        np.random.seed(self.seed)
        self.weights = np.random.random(self.n_inputs)
        self.bias = 0
        self.input = np.ones(self.n_inputs)

    def test_init(self):
        self.assertEqual(self.neuron.n_inputs, 3)
        self.assertSequenceEqual(list(self.neuron.weights), list(self.weights))
        self.assertEqual(self.neuron.bias, self.bias)

    def test_forward(self):
        w = self.weights
        b = self.bias
        expected_output = utils.sigmoid(sum(w*self.input + b))

        actual_output = self.neuron.forward(self.input)
        
        self.assertEqual(actual_output, expected_output)


if __name__ == '__main__':
    unittest.main()