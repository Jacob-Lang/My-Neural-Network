import unittest
from jacobnet.neuron import Neuron

class TestNeuron(unittest.TestCase):

    def setUp(self):
        self.neuron = Neuron()

    def test_init(self):
        self.assertEqual(self.neuron.name == 'Neuron')





if __name__ == '__main__':
    unittest.main()