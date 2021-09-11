#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 25 21:27:27 2020

@author: liu
"""

from preprocess import get_train, get_test
from settings import SIZES, H3P1_NN, HIDDEN_NEURONS, MAX_EPOCHS, VALI_R
from nn import NeuralNetwork, DenseLayer

train_db = get_train()
test_db = get_test()
network = NeuralNetwork()
network.add_layer(DenseLayer(n_input=SIZES['x'][1], n_neurons=HIDDEN_NEURONS, 
                             activation='sigmoid'))
network.add_layer(DenseLayer(n_input=HIDDEN_NEURONS, n_neurons=SIZES['classes'], 
                             activation='sigmoid')) 
network.train(train_db['x'],train_db['y'],max_epochs=MAX_EPOCHS, 
              classify=True, threshold=0.25, 
              validation_ratio=VALI_R, earlystop=(0,5),
              learning_rate = 0.04, 
              momentum=0.8, 
               weight_decay=1E-4,
              )
network.save(str(H3P1_NN))
