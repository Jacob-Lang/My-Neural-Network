"""Unit tests for the package utility functions"""

import pytest
import numpy as np

# for unittesting in spyder IDE. 
import sys
sys.path.append("../") 

from jacobnet import utils


def test_sigmoid():
    # test on 3 known values
    assert utils.sigmoid(0) ==  0.5
    assert utils.sigmoid(100) == pytest.approx(1)
    assert utils.sigmoid(-100) == pytest.approx(0)
    
def test_sigmoid_prime():
    # test on 3 known values
    assert utils.sigmoid_prime(0) == 0.25
    assert utils.sigmoid_prime(100) == pytest.approx(0)
    assert utils.sigmoid_prime(-100) == pytest.approx(0)

def test_cost():
    # dummy target and output
    target = np.zeros((10,1))
    output_array = np.ones((10,1))
    calculated_cost = utils.cost(output_array, target)
    assert calculated_cost == pytest.approx(5)
    
def test_cost_prime():
    # dummy target and output
    target = np.zeros((10,1))
    output_array = np.ones((10,1))
    calculated_cost_prime = utils.cost_prime(target, output_array)
    assert (calculated_cost_prime == np.ones((10,1))).all()
    
def test_label_encoder():
    # dummy labels
    labels = [0,4,9]
    # encoding function
    encoded_labels = utils.label_encoder(labels)
    # expected answer
    expected = np.zeros((10,3))
    
    for li, label in enumerate(labels):
        expected[label, li] = 1  
    
    assert (encoded_labels == expected).all()
    
def test_label_decoder():
     # dummy labels
    labels = [0,4,9]
    # encoded labels
    encoded = np.zeros((10,3))
    
    for li, label in enumerate(labels):
        encoded[label, li] = 1
    
    #decode
    decoded_labels = utils.label_decoder(encoded)
    
    assert (decoded_labels == labels).all()

def test_train_test_split():
    # dummy variables
    X = list(range(10))
    y = list(range(10,20))
    # use function
    X_train, y_train, X_test, y_test = utils.train_test_split(X, y, train_split=0.7)
    
    assert X_train == list(range(7))
    assert y_train == list(range(10,17))
    assert X_test == list(range(7,10))
    assert y_test == list(range(17,20))
