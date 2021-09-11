import numpy as np
from preprocess import get_rand_list, stratify_split
import json
from tqdm import trange
import cupy as cp
from sofm import SOFM


class DenseLayer(object): # fully connected layer
    def __init__(self, n_input, n_neurons, activation=None, trainable=True, 
                 weights=None, zero_bias=False):
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
            weights = cp.random.uniform(low=-a, high=+a, size=(n_input+1, n_neurons)) #Xavier initialization
            if zero_bias:
                weights[0] = 0.
            self.weights = weights
        else:
            if type(weights) != cp.core.core.ndarray:
                weights = cp.asarray(weights)
            
            if zero_bias:
                weights = cp.concatenate((cp.asarray([0], weights)))
            
            if weights.shape != (n_input+1, n_neurons):
                raise ValueError('Given weights {} does not match given '
                                 'dimensions {}'.format(weights.shape, (n_input+1, n_neurons)))
            self.weights = weights

        self.trainable = trainable
            
        self.last_dweights = cp.zeros((n_input+1, n_neurons))
        
        self.activation = activation
        self.last_activation = None
        self.error = cp.zeros(n_neurons)
        self.delta = cp.zeros(n_neurons)
        self.layer_type = "Dense"
    
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
        if type(x) != cp.core.core.ndarray:
                x = cp.asarray(x)
        x = cp.concatenate((cp.asarray([1]),x))
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
            return cp.maximum(s,0)
        elif self.activation == 'tanh':
            return cp.tanh(s)
        elif self.activation == 'sigmoid':
            return 1.0 / (1.0 + cp.exp(-s))
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
            grad = cp.copy(s)
            grad[s>0] = 1.0
            grad[s<=0] = 0.0
            return grad
        elif self.activation == 'tanh':
            return 1 - cp.power(s,2)
        elif self.activation == 'sigmoid':
            return s * (1-s)
        else:
            return cp.ones_like(s) # if no or unkown activation, f' = 1
        
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
        self.weights += dweights
        self.last_dweights = dweights
    
    def get_weights(self):
        return cp.asnumpy(self.weights.copy())
    
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

    def save_dict(self):
        layer_dict = {}
        layer_dict['n_input'], layer_dict['n_neurons'] = self.io()
        layer_dict['activation'] = self.activation
        layer_dict['trainable'] = self.trainable
        layer_dict['weights'] = self.weights.tolist()
        return layer_dict
        
    def info(self):
        return 'Dense, input: {}, output: {}, activation: {}, trainable: {}'.format(self.io()[0], self.io()[1], self.activation, self.trainable)








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
        self.weight_decay = None
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
            if layer.trainable:
                if i == len(self._layers) -1: # for output layer
                    raw_error = y - output
                    raw_error[abs(raw_error)<threshold] = 0 # implement thresholding
                    layer.error = raw_error
                    layer.delta = layer.apply_activation_derivative(output) * layer.error
                else: # for hidden layers
                    next_layer = self._layers[i+1]
                    layer.error = next_layer.weights[1:,:] @ next_layer.delta
                    layer.delta = layer.apply_activation_derivative(layer.last_activation) * layer.error
        # Update weights
        for i,layer in enumerate(self._layers):
            if layer.trainable:
                pre_synaptic = (x if i == 0 else self._layers[i-1].last_activation)
                pre_synaptic = cp.concatenate((cp.asarray([1]),pre_synaptic))
                delta_weights = cp.atleast_2d(pre_synaptic).T @ np.atleast_2d(layer.delta) * learning_rate # basic gradient descent
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
        
        if X_train != cp.core.core.ndarray:
            X_train = cp.asarray(X_train)
        if Y_train != cp.core.core.ndarray:
            Y_train = cp.asarray(Y_train)

        
        
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        
        if classify:
            validation_i, realtrain_i = stratify_split(Y_train.get(), validation_ratio)
        else:
            shuffle_i =  get_rand_list(len(X_train))
            realtrain_i = shuffle_i[int(len(X_train)*validation_ratio):]
            validation_i = shuffle_i[:int(len(X_train)*validation_ratio)]
            
        X_vali = [X_train[i] for i in validation_i]
        Y_vali = [Y_train[i] for i in validation_i]
        good_layers = self._layers
        errors = []
        earlystop_counter = 0
        
        self.info()
        
        for epoch in trange(max_epochs+1, ncols=75, unit='epoch'):
            
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
                        self.train_errors = cp.asarray(errors)
                        return errors
                else: 
                    good_layers = self._layers
                    earlystop_counter = 0
            
                print("\nLoss = {} at epoch {}, training stops in {} epochs".format(errors[-1], epoch, (earlystop[1]-earlystop_counter)*10))
                
            np.random.shuffle(realtrain_i)
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
        if X_test != cp.core.core.ndarray:
            X_test = cp.asarray(X_test)
        if Y_test != cp.core.core.ndarray:
            Y_test = cp.asarray(Y_test)
        
        error = 0
        for i in range(len(X_test)):
            pred = self._feed_forward(X_test[i])
            error += cp.sum(cp.power(pred-Y_test[i],2))
        return cp.asnumpy(0.5*error/len(X_test))
    
    
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
        
        if X_test != cp.core.core.ndarray:
            X_test = cp.asarray(X_test)
        if Y_test != cp.core.core.ndarray:
            Y_test = cp.asarray(Y_test)
        for i in range(len(X_test)):
            pred = self._feed_forward(X_test[i])
            pred = cp.argmax(pred)
            truth = cp.argmax(Y_test[i])
            errors.append(pred==truth)
        return cp.asnumpy(1-sum(errors)/len(errors))

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
            pred = cp.asnumpy(self._feed_forward(X_test[i]))
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
        if type(file_name) is not str:
            file_name = str(file_name)
        
        with open(file_name,'w') as f:
            meta_dict = {}
            meta_dict['learning_rate'] = self.learning_rate
            meta_dict['momentum'] = self.momentum
            meta_dict['weight_decay'] = self.weight_decay
            meta_dict['layers'] = [layer.layer_type for layer in self._layers]
            meta_dict['train_errors'] = self.train_errors.tolist()
            json.dump(meta_dict, f)
            f.write("\n")
            for layer in self._layers:
                layer_dict = layer.save_dict()
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
        if type(file_name) is not str:
            file_name = str(file_name)
        
        with open(file_name,'r') as f:
            for i, line in enumerate(f):
                if i == 0:
                    meta = json.loads(line)
                    self.learning_rate = meta['learning_rate']
                    self.momentum = meta['momentum']
                    self.weight_decay = meta['weight_decay']
                    self.train_errors = np.array(meta['train_errors'])
                    layers = meta['layers']
                else:
                    layer = json.loads(line)
                    if layers[i-1] == "Dense":
                        self.add_layer(DenseLayer(n_input=layer['n_input'], 
                                              n_neurons=layer['n_neurons'],
                                              activation=layer['activation'], 
                                              weights=layer['weights'],
                                              trainable=layer['trainable']))
                    elif layers[i-1] == "SOFM":
                        self.add_layer(SOFM(mapshape=layer['mapshape'],
                                            weights=layer['weights']))
                        
                        
                        
    def info(self):
        """
        Print the model information

        Returns
        -------
        None.

        """
        print("{} layer neural network".format(len(self._layers)))
        print('Learning rate: {}\nMomentum: {}\nWeight Decay: {}'.format(self.learning_rate, self.momentum, self.weight_decay))
        for i,layer in enumerate(self._layers):
            print('[Layer {}] {}'.format(i, layer.info()))
                
    def layers(self, n=None):
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
        if n is None:
            return self._layers
        else:
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
            pred.append(cp.asnumpy(x))
        return np.array(pred)
    
    def pop_layer(self):
        self._layers.pop()


