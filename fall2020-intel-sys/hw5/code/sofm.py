#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 21 17:02:34 2020

@author: liu
"""

import numpy as np
import cupy as cp
from tqdm import trange
import json

class SOFM(object):
    def __init__(self, mapshape, weights=None):
        self.mapshape = mapshape
        
        if weights is not None:
            weights = cp.asarray(weights)
            if weights.shape[1] != np.prod(mapshape):
                raise Exception("Given weights does not match dimension")
            self.weights = weights
        else:
            self.weights = None
        
        indxmap = []
        for i in np.arange(np.prod(mapshape)):
            indx = np.unravel_index(i, mapshape, order='F')
            indxmap.append(indx)
        self.indxmap = cp.asarray(indxmap)

        self.sigma = 0
        self.eta = 0
        self.sigma_tau = 0
        
        self.trainable = False
        self.activation = "SOFM"
        self.zero_bias = True
        self.last_activation = cp.zeros(int(np.prod(mapshape)))
        self.error = cp.zeros(int(np.prod(mapshape)))
        self.delta = cp.zeros(int(np.prod(mapshape)))
        
        self.layer_type = "SOFM"
        
        
        
        
    def initialize(self, data, pca=6):
        if type(data) != cp.core.core.ndarray:
            data = cp.asarray(data)
        
        if type(pca) != int:
            pca = int(pca)
        
        dmin = cp.min(data)
        dmax = cp.max(data)
        
        if pca > 0:
            data = cp.asarray(data)
            _,_,pc = cp.linalg.svd(data)
            pc = pc[:pca].T
            combo = np.random.dirichlet(np.ones(pca), 
                                        size=np.prod(self.mapshape)).T
            combo = cp.asarray(combo)
            
            w = pc @ combo
            self.weights = dmin + (w - cp.min(w)) * (dmax-dmin) / (cp.max(w) - cp.min(w))
        else:
            data = cp.asarray(data)
            self.weights = cp.random.uniform(dmin, dmax, size=
                                         (data.shape[-1],np.prod(self.mapshape)))
    
    
    
    def winner(self, vector, flat=True):
        if type(vector) != cp.core.core.ndarray:
            vector = cp.asarray(vector)
        
        if vector.shape != self.weights.shape:
            vector = cp.broadcast_to(vector, self.weights.shape[::-1]).T
        
        indx_win = cp.argmin(cp.linalg.norm(self.weights - vector, axis=0))
        
        if flat:
            return indx_win
        else:
            return cp.asanyarray(cp.unravel_index(indx_win, self.mapshape, order='F'))



    
    def cycle(self, vector, eta_t, epoch):
        if type(vector) != cp.core.core.ndarray:
            vector = cp.asarray(vector)
        if vector.shape != self.weights.shape:
            vector = cp.broadcast_to(vector, self.weights.shape[::-1]).T
        
        indx_win = self.winner(vector, flat=False)
        dists = cp.linalg.norm(self.indxmap - indx_win, axis=1)
        
        sigma_t = self.sigma * cp.exp( -1 * epoch / self.sigma_tau)
        delta = cp.exp(-1*cp.power(dists, 2) / 2 / cp.power(sigma_t,2))
        delta = cp.broadcast_to(delta, self.weights.shape)
        self.weights += eta_t * delta * (vector - self.weights)

    def train(self, data, max_epochs, learning_rate, learning_rate_decay, 
                 learning_width, learning_width_decay, online_test=True):
        self.sigma = learning_width
        self.sigma_tau = learning_width_decay
        
        if type(data) != cp.core.core.ndarray:
            data = cp.asarray(data)
            
        if self.weights is None:
            self.initialize(data)
        
        for epoch in trange(max_epochs+1, ncols=75, unit='epoch'):
            cp.random.shuffle(data)
            
            if epoch % 10 == 0 and online_test:
                print("\nLoss = {} at epoch {}".format(self.test(data), epoch))
            
            for vector in data:   
                self.cycle(vector, learning_rate*cp.exp(-epoch/learning_rate_decay), epoch)
                
            
    def test(self, data):
        errors = []
        
        if type(data) != cp.core.core.ndarray:
            data = cp.asarray(data)
        
        for vector in data:
            vector = cp.broadcast_to(vector, self.weights.shape[::-1]).T
            dist_win = cp.min(cp.linalg.norm(self.weights - vector, axis=0))
            errors.append(cp.asnumpy(dist_win))
        
        return np.mean(errors)
    
    def save(self, file_name):
        if type(file_name) is not str:
            file_name = str(file_name)
            
        with open(file_name,'w') as f:
            json.dump(self.weights.tolist(), f)

    def save_dict(self):
        layer_dict = {}
        layer_dict['mapshape'] = self.mapshape
        layer_dict['weights'] = self.weights.tolist()
        return layer_dict


    def load(self, file_name):
        if type(file_name) is not str:
            file_name = str(file_name)
            
        with open(file_name, 'r') as f:
            weights = json.load(f)
            weights = cp.asarray(weights)
            if weights.shape[-1] != np.prod(self.mapshape):
                raise Exception("Loaded weights do not match dimension")
            self.weights = weights
        
            
    def get_weights(self):
        return cp.asnumpy(self.weights.copy())
    
    
    def call(self, x):
        if type(x) != cp.core.core.ndarray:
            x = cp.asarray(x)
        
        if x.shape != self.weights.shape:
            x = cp.broadcast_to(x, self.weights.shape[::-1]).T
        
        
        indx_win = self.winner(x, flat=True)
        output = cp.zeros(self.weights.shape[1])
        output[indx_win] = 1.0
        
        # output = cp.linalg.norm(self.weights - x, axis=0)
        # output = 1.0 - (output - cp.min(output)) / (cp.max(output) - cp.min(output))
        
        self.last_activation = output
        
        return output
    
    def apply_activation_derivative(self, s):
        return cp.zeros(self.weights.shape[1])
    
    def update_weights(self, dweights):
        pass
    
    def io(self):
        return (self.weights.shape[0], self.weights.shape[1])
    
    def info(self):
        return 'SOFM, mapshape: {}'.format(self.mapshape)
    

