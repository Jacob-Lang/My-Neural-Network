import pytest
import numpy as np

# for unittesting in spyder IDE. 
import sys
sys.path.append("../") 

from jacobnet.layer import Layer
from jacobnet import utils
    
    
n_inputs = 3
n_neurons = 4

# create three layers (two with same seed, one with random seed = None)
@pytest.fixture
def layer_fixture():
    seed = 42
    return Layer(n_neurons=n_neurons, n_inputs=n_inputs, seed=seed)  
 
@pytest.fixture
def layer_fixture_diff():
    seed = None
    return Layer(n_neurons=n_neurons, n_inputs=n_inputs, seed=seed)  

@pytest.fixture
def layer_fixture_same():
    seed = 42
    return Layer(n_neurons=n_neurons, n_inputs=n_inputs, seed=seed)  
    
# test Layer object has been created    
def test_init(layer_fixture):
    assert type(layer_fixture) == Layer

# test random initialisation and reproducibility
def test_seed(layer_fixture, layer_fixture_diff, layer_fixture_same):
    # check two neurons in layer
    n = np.random.randint(0, n_neurons)
    m = n-1
    # diff weights for diff seeds
    assert (layer_fixture.neurons[n].weights != layer_fixture_diff.neurons[n].weights).all()
    # same weights for same seeds
    assert (layer_fixture.neurons[n].weights == layer_fixture_same.neurons[n].weights).all()
    # diff weights for diff neurons
    assert (layer_fixture.neurons[n].weights != layer_fixture.neurons[m].weights).all()

# test weight matrix by setting all weights to one
def test_weight_matrix(layer_fixture):
    for nrn in layer_fixture.neurons:
        nrn.weights = np.ones(n_inputs)
    assert (layer_fixture.weight_matrix() == np.ones((n_neurons, n_inputs))).all()
    
# test bias vector by setting all biases to one
def test_bias_vector(layer_fixture):
    for nrn in layer_fixture.neurons:
        nrn.bias = 1
    assert (layer_fixture.bias_vector() == np.ones(n_neurons)).all()

# test forward propagation by comparing with matrix multiplication. 
def test_forward(layer_fixture):
    # test input all ones
    input_array = np.ones(n_inputs)
    # propagate via forward function
    a_array, z_array = layer_fixture.forward(input_array)
    # propagate using weight matrix (and bias vector = 0)
    W = layer_fixture.weight_matrix()
    expected_z_array = np.matmul(W, input_array)
    expected_a_array = utils.sigmoid(expected_z_array)
    
    assert (a_array == expected_a_array).all()
    assert (z_array == expected_z_array).all()

    