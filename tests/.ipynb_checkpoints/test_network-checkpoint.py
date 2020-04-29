"""Unit tests for the Network class"""

import pytest
import numpy as np

# for unittesting in spyder IDE. 
import sys
sys.path.append("../") 

from jacobnet.network import Network
from jacobnet import utils

# network architecture
input_size = 10
layer_sizes = [20,40,20,5]

# inputs and outputs
n_inputs = 5
input_array = np.random.random((input_size, n_inputs))
target_output = np.random.random((layer_sizes[-1], n_inputs))

# input and output with wrong shape
input_array_wrong = np.random.random((input_size + 1, n_inputs))
target_output_wrong = np.random.random((layer_sizes[-1] + 1, n_inputs + 1))

# create three networks (two with same seed, one with random seed = None)
@pytest.fixture
def network_fixture():
    seed = 42
    return Network(input_size=input_size, layer_sizes=layer_sizes, seed=seed)
 
@pytest.fixture
def network_fixture_diff():
    seed = None
    return Network(input_size=input_size, layer_sizes=layer_sizes, seed=seed)

@pytest.fixture
def network_fixture_same():
    seed = 42
    return Network(input_size=input_size, layer_sizes=layer_sizes, seed=seed)
    
    
# test Network object has been created    
def test_init(network_fixture):
    assert type(network_fixture) == Network

    
    
# test random initialisation and reproducibility
def test_seed(network_fixture, network_fixture_diff, network_fixture_same):
    
    # get weights and biases for each network fixture
    for l_i in range(len(layer_sizes)):
        W, b = network_fixture.layers[l_i]
        W_diff, b_diff = network_fixture_diff.layers[l_i]
        W_same, b_same = network_fixture_same.layers[l_i]
        
    # only test weights because biases initialised to zero
    assert (W != W_diff).all()
    assert (W == W_same).all()

    

# test forward pass
def test_forward(network_fixture):
    # check exception raised when input_array is wrong type or shape
    with pytest.raises(AssertionError):
        network_fixture.forward(input_array_wrong)
    
    # check train mode produces right arrays at each layer
    store = network_fixture.forward(input_array, mode='train')
    a = input_array
    for l_i, layer in enumerate(network_fixture.layers):
        W, b = layer
        z = np.matmul(W, a) + b
        a = utils.sigmoid(z)
        
        assert (z == store[l_i][0]).all()
        assert (a == store[l_i][1]).all()
    
    
    # check test and train modes produce same output array
    output_test = network_fixture.forward(input_array, mode ='test')
    output_train = store[-1][1]
    assert (output_test == output_train).all()
 

    
# test bakprop aglorithm produces list of deltas
# note: currently this only tests that the algorithm runs not that it runs as expected
# need worked example to check that. 
def test_backpropagation(network_fixture):
    
    store = network_fixture.forward(input_array, mode='train')

    #check exception raised when target_output the wrong shape
    with pytest.raises(AssertionError):
        network_fixture.backpropagate(input_array, store, target_output_wrong)
    
    deltas = network_fixture.backpropagate(input_array, store, target_output)
    
    # check backprop has reached first layer and is the correct shape
    assert deltas[0].shape == (layer_sizes[0], n_inputs)
    
### TESTS MISSING FOR jacobnet.network.Network FROM batch_update ONWARDS.
# If someone wants to tell me the best way to test these complex functions 
# that rely on other complex functions then please let me know.