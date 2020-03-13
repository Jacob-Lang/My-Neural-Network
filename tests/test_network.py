import pytest
import numpy as np

# for unittesting in spyder IDE. 
import sys
sys.path.append("../") 

from jacobnet.network import Network
from jacobnet import utils

input_size = 10
layer_sizes = [20,40,20,5]

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
    # random layer
    ln = np.random.randint(0, len(layer_sizes))
    # random neuron
    nn = np.random.randint(0, layer_sizes[ln])
    # diff neuron in layer
    nm = nn - 1
    # diff weights for diff seeds
    assert (network_fixture.layers[ln].neurons[nn].weights != network_fixture_diff.layers[ln].neurons[nn].weights).all()
    # same weights for same seeds
    assert (network_fixture.layers[ln].neurons[nn].weights == network_fixture_same.layers[ln].neurons[nn].weights).all()
    # diff weights for diff neurons
    assert (network_fixture.layers[ln].neurons[nn].weights != network_fixture.layers[ln].neurons[nm].weights).all()


# have already tested forward pass through layer so just check that the forward function
# succesfully generates a numpy array output of the correct shape.
def test_forward(network_fixture):
    input_array = np.zeros(input_size)
    # mode = test
    output = network_fixture.forward(input_array, mode='test')
    assert (type(output) == np.ndarray)
    assert (output.shape == (layer_sizes[-1],))
    # mode = train
    a_list, z_list = network_fixture.forward(input_array, mode='train')
    # test random layer
    ln = np.random.randint(0, len(layer_sizes))
    for output_list in [a_list, z_list]:
        assert (type(output_list[ln]) == np.ndarray)
        assert (output_list[ln].shape == (layer_sizes[ln],))
    
    
# test bakprop aglorithm produces list of deltas
# note: currently this only tests that the algorithm runs not that it runs as expected
# need worked example to check that. 
def test_backpropagation(network_fixture):
    # dummy input
    input_array = np.random.random(input_size)
    # dummy target
    target_output = np.random.random(layer_sizes[-1])
    # forward
    a, z = network_fixture.forward(input_array, mode='train')
    # then backward
    deltas = network_fixture.backpropagate(a[-1], z, target_output)
    
    # check backprop has reached first layer and is the correct shape
    assert deltas[0].shape == (layer_sizes[0],)
    
# # test netowrk update. again this only tests that the weights have changed
# rather than if they have been updated correctly. 
def test_update_network(network_fixture):
    # dummy input
    input_array = np.random.random(input_size)
    # dummy target
    target_output = np.random.random(layer_sizes[-1])
    # forward
    a, z = network_fixture.forward(input_array, mode='train')
    # then backward
    deltas = network_fixture.backpropagate(a[-1], z, target_output)
    
    # get weight matrix for a neuron
    weights_before = network_fixture.layers[0].weight_matrix()
    
    # update weights
    learning_rate = 0.1
    network_fixture.update_network(deltas, learning_rate, a, input_array)
    
    # get weights again
    weights_after = network_fixture.layers[0].weight_matrix()
    
    # check weights have changed
    assert (weights_before != weights_after).all()
    
