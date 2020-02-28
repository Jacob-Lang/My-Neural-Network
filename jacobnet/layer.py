import numpy as np
import neuron

def relu(x):
    return max([0, x])


class Layer:
    def __init__(self, n_neurons, n_inputs, activation=relu, seed=None):
        ''' Constructor for this class. '''
        
        self.n_neurons = n_neurons
        self.n_inputs = n_inputs
        self.activation = activation
        if seed != None:
            np.random.seed(seed)  # this sets the seed for np.random.random in the Neuron class. 
        
        self.neurons = [neuron.Neuron(n_inputs=self.n_inputs, activation=self.activation) for n in range(n_neurons)]
        
        
    def forward(self, input_array):
        output_array = np.zeros(self.n_neurons)
        for n_i, nrn in enumerate(self.neurons):
            output_array[n_i] = nrn.forward(input_array)
        return output_array
        
    # attributes - n_neurons, n_input, activtion function
    
    # methods - set n_neurons, n_input, activation
    # get and set from neurons. 
    # forward propagate
    
    # backward propagate and update. 
    
    
    
   
#%% Testing 
    
import unittest
    
class TestLayer(unittest.TestCase):

    def setUp(self):
        self.n_neurons = 4
        self.n_inputs = 3
        self.activation = relu
        self.seed = 42
        self.layer = Layer(n_neurons=self.n_neurons, 
                           n_inputs=self.n_inputs, 
                           activation=self.activation,
                           seed=self.seed)
        
        self.test_input = np.ones(self.n_inputs)

    def test_init(self):
        self.assertEqual(len(self.layer.neurons), self.n_neurons)
        
    def test_forward(self):
        actual_ouput = self.layer.forward(self.test_input)
        expected_output = np.array([self.layer.neurons[n_i].forward(self.test_input) 
                                    for n_i in range(self.n_neurons)])
        self.assertSequenceEqual(list(actual_ouput), list(expected_output))

if __name__ == '__main__':
    unittest.main()