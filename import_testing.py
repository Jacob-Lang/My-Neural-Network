# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 13:48:59 2020

@author: Jacob
"""

import numpy as np
import jacobnet as jn

net = jn.network.Network(input_size=784, layer_sizes=[100,50,10])
net2 = jn.network.Network(input_size=784, layer_sizes=[100,50,10], seed=42)
net3 = jn.network.Network(input_size=784, layer_sizes=[100,50,10], seed=42)

input_array = np.zeros(784)

output = net.forward(input_array)
output2 = net2.forward(input_array)
output3 = net3.forward(input_array)


print(output)
print(output2)
print(output3)