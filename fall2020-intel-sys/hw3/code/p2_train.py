#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from preprocess import get_train, get_test
from settings import SIZES, H3P2_NN, HIDDEN_NEURONS, MAX_EPOCHS, VALI_R
from nn import NeuralNetwork, DenseLayer


train_db = get_train()
test_db = get_test()
network = NeuralNetwork()
network.add_layer(DenseLayer(n_input=SIZES['x'][1], n_neurons=HIDDEN_NEURONS, 
                             activation='sigmoid'))
network.add_layer(DenseLayer(n_input=HIDDEN_NEURONS, n_neurons=784, 
                             activation='sigmoid')) 
network.train(train_db['x'], train_db['x'],learning_rate = 0.01, 
              max_epochs=MAX_EPOCHS, classify=False, momentum=0.8, 
              threshold=0, validation_ratio=VALI_R, earlystop=(1E-3,5),
              # weight_decay=1E-5,
              )
network.save(str(H3P2_NN))
