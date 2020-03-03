import unittest
import numpy as np

# for unittesting in spyder IDE. 
import sys
sys.path.append("../") 

from jacobnet.neuron import Neuron
from jacobnet import utils
    
class TestNeuron(unittest.TestCase):

    def setUp(self):
        # create instance
        self.n_inputs = 3
        self.seed = 42
        self.neuron = Neuron(n_inputs=self.n_inputs, seed=self.seed)
        
        # to check random seed
        self.neuron_diff = Neuron(n_inputs=self.n_inputs, seed=None)
        self.neuron_same = Neuron(n_inputs=self.n_inputs, seed=self.seed)

        # example input
        self.input = np.ones(self.n_inputs)

    def test_init(self):
        # check neuron  object is constructed
        self.assertIsInstance(self.neuron, Neuron)
        # check n_inputs, weights, bias set correctly
        self.assertEqual(self.neuron.n_inputs, self.n_inputs)
        
        np.random.seed(self.seed)
        w = 1 - 2*np.random.random(self.n_inputs)
        self.assertSequenceEqual(list(self.neuron.weights), list(w))
        self.assertEqual(self.neuron.bias, 0)
        
    def test_seed(self):
        # test reproducibility
        w = list(self.neuron.weights)
        w_diff = list(self.neuron_diff.weights)
        w_same = list(self.neuron_same.weights)
        # check weights initialised the same when seed is set
        self.assertSequenceEqual(w, w_same)
        # and differently when seed=None
        self.assertNotEqual(w, w_diff)

    def test_forward(self):
        # test activation and weighted input.
        # actual output when weights all set to 1
        self.neuron.weights = np.ones(self.n_inputs)
        a, z = self.neuron.forward(self.input)
        
        # for weights = 1, expected_z = sum(input) = n_inputs
        expected_z = self.n_inputs
        expected_a = utils.sigmoid(expected_z)
    
                
        self.assertEqual(z, expected_z)
        self.assertEqual(a, expected_a)


#if __name__ == '__main__':
#    unittest.main()
