import numpy as np
import layer

def relu(x):
    return max([0, x])

class Network:
    def __init__(self, input_size, layer_sizes, seed=None):
        ''' Constructor for this class. '''
        
        self.input_size = input_size
        self.layer_sizes = layer_sizes # final layer must match output size. can test for this. in training.
        self.input_sizes = [input_size] + layer_sizes[:-1]
        
        if seed != None:
            np.random.seed(seed)  # this sets the seed for np.random.random in the Layers and Neuron class. 
        
        
        self.layers = []
        for l_i in range(len(layer_sizes)): # +1 because output layer is a layer too.
            # number of neurons in this layer
            n_neurons = self.layer_sizes[l_i]
            # number of inputs for this layer
            n_inputs = self.input_sizes[l_i]

            activation=relu
            
            layer_i = layer.Layer(n_neurons=n_neurons,
                                  n_inputs=n_inputs,
                                  activation=activation,
                                  seed=None)
            
            self.layers.append(layer_i)
            
    def forward(self, input_array):
        x = input_array
        for layer_i in self.layers:
            x = layer_i.forward(x)
        return x

        
        
    # attributes - n_layers and sizes, input size, output
    
    # methods - forward propagate
    # get info from layers. set info from layers
    
    # train. 
    # predict. 
    
    
#%% Testing 
    
import unittest
    
class TestNetwork(unittest.TestCase):

    def setUp(self):
        self.input_size = 10
        self.layer_sizes = [20,40,20,2] # final layer must match output size. can test for this. in training.
        self.seed = 42
        
        self.network = Network(input_size=self.input_size, 
                               layer_sizes=self.layer_sizes,
                               seed=self.seed)
        
        self.input_array = np.random.random(self.input_size)
        
    def test_init(self):
        self.assertIsInstance(self.network, Network)
        
    def test_forward(self):
        output= self.network.forward(self.input_array)
        self.assertEqual(output.shape, (self.layer_sizes[-1],))
        self.assertIsInstance(output, np.ndarray)

if __name__ == '__main__':
    unittest.main()