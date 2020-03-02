"""Network class."""

import numpy as np
#import utils
#import layer
from jacobnet import utils
from jacobnet import layer

class Network:
    def __init__(self, input_size, layer_sizes, seed=None):
        ''' Constructor for this class. '''
        
        # size of input data
        self.input_size = input_size
        # list of layer sizes - last layer must be size of desired output
        self.layer_sizes = layer_sizes # final layer must match output size. can test for this. in training.
        # input sizes for each layer
        self.input_sizes = [input_size] + layer_sizes[:-1]
        
        # for reproducibility
        if seed != None:
            np.random.seed(seed)  # this sets the seed for np.random.random in the Layers and Neuron class. 
        
        # netowrk is just a list of layers
        self.layers = []
        for l_i in range(len(layer_sizes)): # +1 because output layer is a layer too.
            # number of neurons in this layer
            n_neurons = self.layer_sizes[l_i]
            # number of inputs for this layer
            n_inputs = self.input_sizes[l_i]
            
            layer_i = layer.Layer(n_neurons=n_neurons,
                                  n_inputs=n_inputs,
                                  seed=None)
            
            self.layers.append(layer_i)

        # remove seed after setting weights so it doesn't affect future functions
        np.random.seed(None)

    def forward(self, input_array, mode='test'):
        """Forward propagation of an array through the network"""
        
        # if mode=='test' then only return output array 
        if mode == 'test':
            a = input_array
            for layer_i in self.layers:
                a, z = layer_i.forward(a)
            return a
        
        # if mode =='train' then store intermediate activations and weighted inputs. 
        if mode == 'train':
            a_store = []
            z_store = []     
            a = input_array
            for layer_i in self.layers:
                a, z = layer_i.forward(a)
                a_store.append(a)
                z_store.append(z)
            
            return a_store, z_store
    
    ## BACKPROPAGATION
    def backpropagate(self, a_store, z_store):
        # TO DO
        # return layer_deltas
        return 0
    
        
    def update_network(self, layer_deltas):
        # TO DO
        # update neuron weights and biases from layer_deltas
        return 0
    
    
    
#%% Testing 
    
import unittest
    
class TestNetwork(unittest.TestCase):

    def setUp(self):
        # example parameters for test
        self.input_size = 10
        self.layer_sizes = [20,40,20,2] # final layer must match output size. can test for this. in training.
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
        w = list(self.network.layers[0].neurons[0].weights)
        w_diff = list(self.network_diff.layers[0].neurons[0].weights)
        w_same = list(self.network_same.layers[0].neurons[0].weights)
        
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
    

if __name__ == '__main__':
    # run tests
    unittest.main()