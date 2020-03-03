import unittest
import numpy as np

# for unittesting in spyder IDE. 
import sys
sys.path.append("../") 

from jacobnet.network import Network
from jacobnet import utils


class TestNetwork(unittest.TestCase):

    def setUp(self):
        # example parameters for test
        self.input_size = 10
        self.layer_sizes = [20,40,20,5] # final layer must match output size. can test for this. in training.
        self.seed = 42
        
        # create instance of network
        self.network = Network(input_size=self.input_size, 
                               layer_sizes=self.layer_sizes,
                               seed=self.seed)
        
        # two more networks to check reproducibility
        self.network_diff = Network(input_size=self.input_size, 
                               layer_sizes=self.layer_sizes,
                               seed=None)
                
        self.network_same = Network(input_size=self.input_size, 
                               layer_sizes=self.layer_sizes,
                               seed=self.seed)
        
        # example input array
        self.input_array = np.random.random(self.input_size)
        
    def test_init(self):
        # is network instance created? 
        self.assertIsInstance(self.network, Network)
        
    def test_seed(self):
        # check weights of a particular neuron to check if seed works. 
        w = list(self.network.layers[-1].neurons[-1].weights)
        w_diff = list(self.network_diff.layers[-1].neurons[-1].weights)
        w_same = list(self.network_same.layers[-1].neurons[-1].weights)
        
        self.assertSequenceEqual(w, w_same)
        self.assertNotEqual(w, w_diff)
        
    def test_forward(self):
        # propagate inout array
        a = self.network.forward(self.input_array, mode='test')
        # check output correct shape
        self.assertEqual(a.shape, (self.layer_sizes[-1],))
        # check is a numpy array
        self.assertIsInstance(a, np.ndarray)
        
        # same for train mode
        a_store, z_store = self.network.forward(self.input_array, mode='train')
        # check both modes produce same output array
        self.assertEqual(list(a_store[-1]), list(a))
        # check activation related to weighted input correcclty. 
        self.assertEqual(list(utils.sigmoid(z_store[-1])), list(a_store[-1]))
    
    def test_backpropagate_and_update(self):
        # dummy input and output vectors
        input_array = np.random.random(10)
        target_output = np.random.random(5)
        # feedforward
        a_store, z_store = self.network.forward(input_array=input_array, mode='train')
        # backpropagate
        delta = self.network.backpropagate(a_store[-1], z_store, target_output)
        # check backprop has reached first layer and is correct shape
        self.assertIsInstance(delta[0], np.ndarray)
        self.assertEqual(delta[0].shape, (self.network.layers[0].n_neurons,))
        # check last layer too
        self.assertIsInstance(delta[-1], np.ndarray)
        self.assertEqual(delta[-1].shape, (self.network.layers[-1].n_neurons,))   

        
    def test_update_weights(self):
        # dummy input and output vectors
        input_array = self.input_array
        target_output = np.random.random(5)
        # feedforward
        a_store, z_store = self.network.forward(input_array=input_array, mode='train')
        # backpropagate
        deltas = self.network.backpropagate(a_store[-1], z_store, target_output)

        # weights before update
        W = self.network.layers[0].weight_matrix()
        # update
        learning_rate = 0.01
        self.network.update_network(deltas, learning_rate, a_store, input_array)
        # weights after update
        W_updated = self.network.layers[0].weight_matrix()
        # check all weights have changed
        self.assertTrue((W != W_updated).all())
