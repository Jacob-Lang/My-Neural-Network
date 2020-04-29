"""The Network class for the jacobnet package"""

import numpy as np
from jacobnet import utils

class Network:
    """
    Neural network class.
    
    This is a vanilla feedforward, multi-layer perceptron (MLP) neural network. The input data is assumed to be flat, shape=(input_size, 1), but mulitple training examples can be passed through simultaneously, shape=(input_size, n_inputs). The activtion for all layers is set as the sigmoid function. 
    """
    def __init__(self, input_size, layer_sizes, seed=None):
        """
        Network constructor. Initialises the weights and biases for a desired MLP network architecture. 

        Parameters
        ----------
        input_size : int
            Number of datapoints in training data (assumed flat data).
        layer_sizes : list, elements 
            Number of neurons in each layer. Including output layer.
        seed : int
            Random seed for reproducibility. (Default = None.)
        """
        
        # size of input data
        self.input_size = input_size
        # list of layer sizes - last layer must be size of desired output
        self.layer_sizes = layer_sizes # final layer must match output size. can test for this. in training.
        # input sizes for each layer
        self.input_sizes = [input_size] + layer_sizes[:-1]
        
        # for reproducibility
        if seed != None:
            np.random.seed(seed)  # this sets the seed for np.random.random in the Layers and Neuron class. 
        
        # network is just a list of layers
        # each layer contains a weight matrix and a bias vector
        self.layers = []
        for l_i in range(len(layer_sizes)):
            
            # number of neurons in this layer
            n_neurons = self.layer_sizes[l_i]
            # number of inputs for this layer
            n_inputs = self.input_sizes[l_i]
            
            # initialise weight matrix for each layer. 
            W = np.random.random((n_neurons, n_inputs))*2 - 1  
            
            # initialise bias vector to zero
            b = np.zeros((n_neurons, 1))
            
            # store weight matrix and bias vector
            self.layers.append([W, b])

        # remove seed after setting weights so it doesn't affect future functions
        np.random.seed(None)

    def forward(self, input_array, mode='test'):
        """
        Forward propagation of data through the network.

        Parameters
        ----------
        input_array : numpy array,  shape=(input_size, n_inputs)
            The input array to be fed forward. 
        mode : str
        Whether to store ('train') or not ('test') the output of each layer (default 'test').

        Returns
        -------
        If mode=='test'
        output_array : numpy array, shape=(output_size, n_inputs)
            The output from the network. 
            
        If mode=='train':
        store: list, elements are [z, a] for each layer
            A list of outputs for each layer where z is the weighted input and a = sigmoid(z).
        """

        # assert that input_array is correct shape
        assert len(input_array.shape) == 2 and input_array.shape[0] == self.input_size
        
        a = input_array
        
        # if mode=='test' then only return output array 
        if mode == 'test':
            for layer in self.layers:
                W, b = layer
                z = np.matmul(W, a) + b
                a = utils.sigmoid(z)
            
            output_array = a
            return output_array
        
        # if mode =='train' then store intermediate activations and weighted inputs. 
        if mode == 'train':
            store = []
            for layer in self.layers:
                W, b = layer
                z = np.matmul(W, a) + b
                a = utils.sigmoid(z)
                store.append([z, a])                
            return store


    def backpropagate(self, input_array, store, target_output):
        """
        Backward propagation of the errors for each layer.

        Parameters
        ----------
        input_array : numpy array,  shape=(input_size, n_inputs)
            The input array to be fed forward. 
        store : list
            The output of self.forward when mode='train'.
        target_output : numpy_array, shape=(output_size, n_inputs)
            The training labels in array format.

        Returns
        -------
        deltas : list, elements numpy arrays shape=(n_neurons, n_inputs) for each layer
            The layer errors obtained by backpropagation
        """

        # assert that target_output is correct shape
        assert target_output.shape == (self.layer_sizes[-1], input_array.shape[1])
        
        # empty list to fill layer errors
        n_layers = len(self.layers)
        deltas =[0]*n_layers

        # error of final layer
        final_z = store[-1][0]
        network_output = store[-1][1]
        deltas[-1] = utils.cost_prime(target_output, network_output)*utils.sigmoid_prime(final_z)
        
        # gradient clipping
        deltas[-1] = np.clip(deltas[-1], -1.0, 1.0)
        
        # error of preceding layers
        for l_i in range(-2, -(n_layers+1),-1):
            # weight matrix for l_i + 1 layer
            W = self.layers[l_i + 1][0]
            deltas[l_i] = np.matmul(np.transpose(W), deltas[l_i + 1])*store[l_i][0]
            
            # gradient clipping
            deltas[l_i] = np.clip(deltas[l_i], -1.0, 1.0)
        
        return deltas

        
    def batch_update(self, input_array, target_output, learning_rate):
        """
        Update the network weights with the average of a batch of training data.

        Parameters
        ----------
        input_array : numpy array,  shape=(input_size, n_inputs)
            The input array to be fed forward. 
        target_output : numpy_array, shape=(output_size, n_inputs)
            The training labels in array format.
        learning_rate: float
            Controls step size in gradient descent
            
        Returns
        -------
        None
        """

        n_inputs = input_array.shape[1]
        
        # forward pass
        store = self.forward(input_array, mode='train')
        deltas = self.backpropagate(input_array, store, target_output)

        # trick: add to store so that store[-1][1] is input
        store.append([0, input_array])
        
        
        for l_i, layer in enumerate(self.layers):
            W, b = layer
            delta = deltas[l_i]
            a_in = store[l_i - 1][1] # above trick used here when l_i = 0
            
            # biases
            dCdb = delta.mean(axis=1, keepdims=True)
            db = -learning_rate*dCdb
            
            # weights
            dCdW = np.matmul(delta, a_in.T)/n_inputs
            dW = -learning_rate*dCdW
            
            # update
            self.layers[l_i][0] += dW
            self.layers[l_i][1] += db
            
            
    def train(self, images, labels, learning_rate=0.01, batch_size=32, epochs=1, verbose=False, validation_split=0.1):
        """
        Train the network.

        Parameters
        ----------
        images : list 
            List of mnist images. Each image is flat a list of grayscale data, length 784.
        labels : list
            List of training labels. Each label is an integer 0 - 9.
        learning_rate : float, (default=0.01)
            Controls the step size in batch_update(). 
        batch_size : 
            Batch size for batch update.
        epochs : int, (default=1)
            Number of epochs to train the network. (Data shuffled between each epoch.)
        verbose : bool, (default=False)
            Whether to print training progress and accuracy to screen.
        validation_split : float, in range [0,1]
            How much data to split off as validation set.
    
        Returns
        -------
        None   
        """
        
        # hold out a valdiation set
        X_train, y_train, X_val, y_val = utils.train_test_split(images, labels, train_split=(1-validation_split))

        # list to store accuracy after each epoch
        self.history = np.zeros(epochs)
        
        N_training = len(X_train)
        
        # loop over epochs
        for n_epoch in range(epochs):
            
            if verbose:
                print('Training epoch', n_epoch + 1, '/', epochs)

            # shuffle training set by indices
            shuffle_order = list(range(N_training))
            np.random.shuffle(shuffle_order)

            m = 0
            # loop over mini batches
            while m <= N_training:

                # select batches of training data
                index_low = m
                index_high = min(m + batch_size, N_training)

                batch_indices = shuffle_order[index_low:index_high]

                X_batch = [X_train[i] for i in batch_indices]
                y_batch = [y_train[i] for i in batch_indices]

                # reshape and normalise X_batch
                X_batch = np.array(X_batch).T/255.0   
                # encode labels
                y_batch = utils.label_encoder(y_batch)

                # update on this batch
                self.batch_update(X_batch, y_batch, learning_rate)

                m+=batch_size
                
            # evaluate network accuracy on holdout validation set
            accuracy = self.accuracy_score(X_val, y_val)
            
            if verbose:
                print('Validation accuracy:', accuracy, '%')
    
            self.history[n_epoch] = accuracy
                        
                
    def accuracy_score(self, images, labels):
        """
        Calculate accuracy of the network.

        Parameters
        ----------
        images : list  
            List of mnist images. Each image is flat a list of grayscale data, length 784.
        labels : list
            List of test labels. Each label is an integer 0 - 9.
    
        Returns
        -------
        accuracy: float
            The accuracy (= N_correct/N_total*100 %) of the network evaluated on the given dataset.
        """
        
        N_test = len(labels)
        N_correct = 0
        
        # run in batches of size 32
        batch_size = 32
        m = 0
        while m <= N_test:

            # select batches of training data
            index_low = m
            index_high = min(m + batch_size, N_test)

            images_batch = images[index_low:index_high]
            labels_batch = labels[index_low:index_high]

            # reshape and normalise X_batch
            X_batch = np.array(images_batch).T/255.0 
            # convert labels_batch to list
            labels_batch = np.array(labels_batch)
            
            # feedforward the input data and decode the output to predictions
            output = self.forward(X_batch)
            predictions = utils.label_decoder(output) 
            
            # count how many predictions match labels
            N_correct_this_batch = sum(predictions == labels_batch)

            # add to count
            N_correct += N_correct_this_batch
            
            m+=batch_size
        
        # calculate accuracy as percentage
        accuracy = N_correct/N_test*100 
        
        return accuracy


        
    
        
        