import pytest
import numpy as np

# for unittesting in spyder IDE. 
import sys
sys.path.append("../") 

from jacobnet.neuron import Neuron
from jacobnet import utils

n_inputs = 3

# create 3 neurons. Two with seed = 42, one with random seed=None
@pytest.fixture
def neuron_fixture():
    seed = 42
    return Neuron(n_inputs=n_inputs, seed=seed)

@pytest.fixture
def neuron_fixture_diff():
    seed = None
    return Neuron(n_inputs=n_inputs, seed=seed)

@pytest.fixture
def neuron_fixture_same():
    seed = 42
    return Neuron(n_inputs=n_inputs, seed=seed)

# test weights and bias as expected
def test_init(neuron_fixture):
    seed = 42
    np.random.seed(seed)
    expected_weights = 1 - 2*np.random.random(n_inputs) 

    assert (neuron_fixture.weights == expected_weights).all()
    assert neuron_fixture.bias == 0

# test reproducibility and random initialisation via seed. 
def test_seed(neuron_fixture, neuron_fixture_same, neuron_fixture_diff):
    assert (neuron_fixture.weights == neuron_fixture_same.weights).all()
    assert (neuron_fixture.weights != neuron_fixture_diff.weights).all()

# test forward propagation through neuron. 
# for input = [1,1,1,...] and bias = 0 the weighted input z = sum(weights) and a = sigmoid(z).
def test_forward(neuron_fixture):
    input_array = np.ones(n_inputs)
    a, z = neuron_fixture.forward(input_array)
    assert z == sum(neuron_fixture.weights)
    assert a == utils.sigmoid(z)


    
