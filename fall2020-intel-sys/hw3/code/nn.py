#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 14:15:46 2020

@author: liu
"""
import numpy as np
from preprocess import get_rand_list, stratify_split
import json
from tqdm import trange


class DenseLayer(object): # fully connected layer
    def __init__(self, n_input, n_neurons, activation=None, trainable=True, 
                 weights=None):
        """
        Initialize a fully-connected layer

        Parameters
        ----------
        n_input : uint
            number of input nodes.
        n_neurons : uint
            number of neurons / output nodes.
        activation : str, optional
            activation function name. The default is None.
        weights : np array, optional
            matrix for weights. The default is None.

        Returns
        -------
        None.

        """
        
        if weights is None:
            a = np.sqrt(6/(n_input+n_neurons))
            self.weights = np.random.uniform(low=-a, high=+a, size=(n_input+1, n_neurons)) #Xavier initialization
        else:
            weights = np.array(weights)
            if weights.shape == (n_input+1, n_neurons):
                self.weights = weights
            else:
                raise ValueError("Given weights does not match given dimensions")
        self.trainable = trainable
            
        self.last_dweights = np.zeros((n_input+1, n_neurons))
        
        self.activation = activation
        self.last_activation = None
        self.error = None
        self.delta = None
    
    def set_trainable(self, trainable):
        """
        Configure if the layer is trainable

        Parameters
        ----------
        trainable : bool
            whether the layer is trainable.

        Returns
        -------
        None.

        """
        self.trainable = trainable
    
    def call(self,x):
        """
        Calculate the output given input

        Parameters
        ----------
        x : np array or list
            array or list of input to the layer.

        Returns
        -------
        np array
            array of output from the layer.

        """
        x = np.append([1],x)
        s = x @ self.weights
        self.last_activation = self._apply_activation(s) 
        return self.last_activation

    
    def _apply_activation(self, s):
        """
        calcualte activated output

        Parameters
        ----------
        s : np array
            array of the inner product between input and weights.

        Returns
        -------
        np array
            activated output.

        """
        if self.activation == 'relu':
            return np.maximum(s,0)
        elif self.activation == 'tanh':
            return np.tanh(s)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + np.exp(-s))
        else:
            return s # if no or unkown activation, f = s
        
    def apply_activation_derivative(self, s):
        """
        calculate the derivative of activation function

        Parameters
        ----------
        s : np array
            array of the inner product between input and weights.

        Returns
        -------
        np array
            calcualted output after activation dirivative.

        """
        if self.activation == 'relu':
            grad = np.copy(s)
            grad[s>0] = 1.0
            grad[s<=0] = 0.0
            return grad
        elif self.activation == 'tanh':
            return 1 - s ** 2
        elif self.activation == 'sigmoid':
            return s * (1-s)
        else:
            return np.ones_like(s) # if no or unkown activation, f' = 1
        
    def update_weights(self, dweights):
        """
        Used for gradient descent to change weight values

        Parameters
        ----------
        dweights : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        if self.trainable:
            self.weights += dweights
            self.last_dweights = dweights
        
    def io(self):
        """
        Get the input and output number of the layer

        Returns
        -------
        int
            input dimension.
        TYPE
            output dimension.

        """
        return (self.weights.shape[0]-1, self.weights.shape[1])

        








class NeuralNetwork(object): # neural network model
    def __init__(self):
        """
        Initialize model with a buffer for layers

        Returns
        -------
        None.

        """
        self._layers = []
        self.learning_rate = None
        self.momentum = None
        self.train_errors = []
    
    def add_layer(self,layer):
        """
        Append a layer at the end

        Parameters
        ----------
        layer : XXLayer type
            a nn layer.

        Returns
        -------
        None.

        """
        self._layers.append(layer)
    
    
    def _feed_forward(self,x):
        """
        Calculate output from an input

        Parameters
        ----------
        x : np array
            input vector into the network.

        Returns
        -------
        x : np array
            output vector out of the network.

        """
        for layer in self._layers:
            x = layer.call(x)
        return x
    
    def _back_prop(self, x, y, learning_rate, momentum=0.0, threshold=0.0, weight_decay=0.0):
        """
        Implement back propagation with momentum gradient descent and 
        thresholded output

        Parameters
        ----------
        x : np array
            input vector to the network.
        y : np array
            target output vector for the network.
        learning_rate : float
            learning rate.
        momentum : float, optional
            alpha value to control the momentum gradient descent. The default is 0.
        threshold : float, optional
            threshold window to be considered 0 or 1, detail see self.train(). The default is 0.

        Returns
        -------
        None.

        """
        output = self._feed_forward(x)
        # Calculate gradients
        for i in reversed(range(len(self._layers))): # start from the last layer
            layer = self._layers[i]
            if i == len(self._layers) -1: # for output layer
                raw_error = y - output
                raw_error = [0 if np.abs(e)<threshold else e for e in raw_error]# implement thresholding
                layer.error = raw_error
                layer.delta = layer.apply_activation_derivative(output) * layer.error
            else: # for hidden layers
                next_layer = self._layers[i+1]
                layer.error = next_layer.weights[1:,:] @ next_layer.delta
                layer.delta =  layer.apply_activation_derivative(layer.last_activation) * layer.error
        # Update weights
        for i,layer in enumerate(self._layers):
            pre_synaptic = (x if i == 0 else self._layers[i-1].last_activation)
            pre_synaptic = np.append([1], pre_synaptic)
            pre_synaptic = np.atleast_2d(pre_synaptic)
            delta_weights = pre_synaptic.T @ np.atleast_2d(layer.delta) * learning_rate # basic gradient descent
            delta_weights -= 2*weight_decay*learning_rate*layer.weights # implement weight decay 
            delta_weights += momentum * layer.last_dweights # implement momentum
            layer.update_weights(delta_weights)
    
            
    def train(self, X_train, Y_train, learning_rate, max_epochs, classify=False,
              momentum=0, threshold=0, validation_ratio=0.0, weight_decay=0.0,
              earlystop=None):
        """
        Train the network with given input, output, and hyper-parameters

        Parameters
        ----------
        X_train : list or np array
            a batch of input vector to the network.
        Y_train : list or np array
            a batch of target output for the network.
        learning_rate : float
            specifies the learning rate of gradient descent.
        max_epochs : int
            specifies the max amount of epochs to train.
        momentum : float, optional
            specifies the alpha value for gradient descent. The default is 0.
        threshold : float, optional
            specified the threshold windows for the output to consider 0 or 1.
            Output is 0 if 0<= output < threshold; output is 1 if 
            1-threshold < output <=1. 
            The default is 0.
        stochastic_ratio : float, optional
            specifies how much of the input batch is selected. 
            The default is 1.0.
        earlystop : set of 2 elements, optional
            specifies earlystop. [0] represents the max value for the output
            to be the 'same'. [1] represents the patience. The default is None.

        Returns
        -------
        errors : np array
            errors every 10 epochs of training.

        """
        if earlystop is None:
            earlystop = (0, max_epochs//10)
        
        
        X_train = np.array(X_train)
        Y_train = np.array(Y_train)
        
        if X_train.shape[1] != self._layers[0].weights.shape[0]-1:
            raise ValueError("Input data does not match layer dimension")
        if Y_train.shape[1] != self._layers[-1].weights.shape[1]:
            raise ValueError("Output data does not match layer dimension")
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        if classify:
            validation_i, realtrain_i = stratify_split(Y_train, validation_ratio)
        else:
            shuffle_i =  get_rand_list(len(X_train))
            realtrain_i = shuffle_i[int(len(X_train)*validation_ratio):]
            validation_i = shuffle_i[:int(len(X_train)*validation_ratio)]
            
        X_vali = [X_train[i] for i in validation_i]
        Y_vali = [Y_train[i] for i in validation_i]
        good_layers = self._layers
        errors = []
        earlystop_counter = 0
        
        for epoch in trange(max_epochs+1, ncols=75, unit='epochs'):
            if epoch % 10 == 0:
                
                if classify:
                    error = self.classify_test(X_vali, Y_vali)
                else:
                    error = self.raw_test(X_vali, Y_vali)
                errors.append(error)
                
                
                if epoch == 0:
                    print("\nLoss = {} at epoch {}".format(errors[-1], epoch))
                    continue
                
                if (np.min(errors[:-1]) - errors[-1]) < earlystop[0]:
                    earlystop_counter += 1
                    if earlystop_counter == earlystop[1]:
                        print("\nEarly stop triggered at epoch {}, restored to epoch {}"
                              .format(epoch, epoch-earlystop[1]*10))
                        self._layers = good_layers
                        self.train_errors = np.array(errors)
                        return self.train_errors
                else: 
                    good_layers = self._layers
                    earlystop_counter = 0
            
                print("\nLoss = {} at epoch {}, training stops in {} epochs".format(errors[-1], epoch, (earlystop[1]-earlystop_counter)*10))
                
            for i in realtrain_i:
                self._back_prop(X_train[i], Y_train[i], learning_rate, momentum, threshold, weight_decay)
                    
            
        self.train_errors = np.array(errors)
        return np.array(errors)
                    
    def raw_test(self, X_test, Y_test):
        """
        Test the model with given data, calculate J2 loss

        Parameters
        ----------
        X_test : 2D list or np array
            input data to the network.
        Y_test : 2D list or np array
            true output data.

        Returns
        -------
        float
            average J2 loss.

        """
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        error = 0
        for i in range(len(X_test)):
            pred = self._feed_forward(X_test[i])
            error += np.sum((pred-Y_test[i])**2)
        return 0.5*error/len(X_test)
    
    
    def classify_test(self, X_test, Y_test):
        """
        Test the network with given input, output and accuracy

        Parameters
        ----------
        X_test : list or np array
            input vector to the network.
        Y_test : list or np array
            ground truth for the testing.

        Returns
        -------
        float
            test accuracy

        """
        errors = []
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        for i in range(len(X_test)):
            pred = self._feed_forward(X_test[i])
            pred = np.argmax(pred)
            truth = np.argmax(Y_test[i])
            errors.append(pred==truth)
        return 1-np.sum(errors)/len(errors)

    def get_cm(self, X_test, Y_test):
        """
        Give the confusion matrix for classification problems

        Parameters
        ----------
        X_test : list or np array
            input vector to the network.
        Y_test : list or np array
            ground truth for the testing.

        Returns
        -------
        cm : np array
            confusion matrix.

        """
        X_test = np.array(X_test)
        Y_test = np.array(Y_test)
        n_classes = Y_test.shape[1]
        cm = np.zeros((n_classes, n_classes))
        for i in range(len(X_test)):
            pred = self._feed_forward(X_test[i])
            max_pred = np.max(pred)
            pred_bin = np.atleast_2d([p==max_pred for p in pred]).T
            truth = np.atleast_2d(Y_test[i])
            cm += pred_bin @ truth
        return cm
    
    def save(self, file_name):
        """
        Save the model in a json file
        First line of json is meta data
        Following line includes layer info

        Parameters
        ----------
        file_name : str
            string to save data into.

        Returns
        -------
        None.

        """
        with open(file_name,'w') as f:
            meta_dict = {}
            meta_dict['learning_rate'] = self.learning_rate
            meta_dict['momentum'] = self.momentum
            meta_dict['weight_decay'] = self.weight_decay
            meta_dict['train_errors'] = self.train_errors.tolist()
            json.dump(meta_dict, f)
            f.write("\n")
            for layer in self._layers:
                layer_dict = {}
                layer_dict['n_input'], layer_dict['n_neurons'] = layer.io()
                layer_dict['activation'] = layer.activation
                layer_dict['trainable'] = layer.trainable
                layer_dict['weights'] = layer.weights.tolist()
                json.dump(layer_dict, f)
                f.write("\n")
        
    def load(self, file_name):
        """
        Loads the json file for a model

        Parameters
        ----------
        file_name : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        with open(file_name,'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    meta = json.loads(line)
                    self.learning_rate = meta['learning_rate']
                    self.momentum = meta['momentum']
                    self.weight_decay = meta['weight_decay']
                    self.train_errors = np.array(meta['train_errors'])
                else:
                    layer = json.loads(line)
                    self.add_layer(DenseLayer(n_input=layer['n_input'], 
                                              n_neurons=layer['n_neurons'],
                                              activation=layer['activation'], 
                                              weights=layer['weights'],
                                              trainable=layer['trainable']))
    def info(self):
        """
        Print the model information

        Returns
        -------
        None.

        """
        print('Learning rate: {}\nMomentum: {}'.format(self.learning_rate, self.momentum))
        for i,layer in enumerate(self._layers):
            print("Layer {} = input: {}, output: {}, activation: {}, trainability: {}".format(i, layer.io()[0], layer.io()[1], layer.activation, layer.trainable))
                
    def layers(self, n):
        """
        Get layers of a model

        Parameters
        ----------
        n : int
            index of the layer starting from 0.

        Returns
        -------
        Layer object
            the n-th layer of the model.

        """
        return self._layers[n]
    
    def predict(self, X):
        """
        Make prediction from the given input

        Parameters
        ----------
        X : 2D list or np array
            Input data to the network.

        Returns
        -------
        pred : list
            predicted output.

        """
        pred = []
        for x in X:
            for layer in self._layers:
                x = layer.call(x)
            pred.append(x)
        return pred