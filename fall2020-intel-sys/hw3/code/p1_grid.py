#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 09:35:16 2020

@author: liu
"""

import numpy as np
import itertools
from preprocess import get_train
from settings import SIZES, HIDDEN_NEURONS, VALI_R, H3P1_GRID
from nn import NeuralNetwork, DenseLayer

train_db = get_train()

if H3P1_GRID.is_file():
    H3P1_GRID.unlink()
with open(str(H3P1_GRID), 'w') as f:
    f.write("eta,alpha,lambda,error\n")

etas = [0.01,0.02,0.03,0.04,0.05,0.06,0.07,0.08,0.09]
alphas = [0.8]
lambdas = [1E-4]

hparams = list(itertools.product(etas,alphas,lambdas))

grid_errors = []
for e,a,l in hparams:
    network = NeuralNetwork()
    network.add_layer(DenseLayer(n_input=SIZES['x'][1], n_neurons=HIDDEN_NEURONS, 
                                 activation='sigmoid'))
    network.add_layer(DenseLayer(n_input=HIDDEN_NEURONS, n_neurons=SIZES['classes'], 
                                 activation='sigmoid')) 
    errors = network.train(train_db['x'], train_db['y'],max_epochs=50, 
                  classify=True, threshold=0.25, 
                  validation_ratio=VALI_R,
                  learning_rate = e, 
                  momentum=a, 
                  weight_decay=l,
                  )
    grid_errors.append([e, a, l, np.min(errors[1:])])
    print("\n-------- (Search {}/{}): eta={:.3f}, alpha={:.1f}, lamda={:.5f}, min error={:.4f}--------".format(len(grid_errors), len(hparams), e,a,l,np.min(errors[1:])))
    with open(str(H3P1_GRID), 'a') as f:
        np.savetxt(f, [grid_errors[-1]], delimiter=',')

for l in grid_errors:
    print(l)
    