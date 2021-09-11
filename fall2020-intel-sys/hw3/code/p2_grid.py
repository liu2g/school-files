# -*- coding: utf-8 -*-
"""
Created on Mon Nov  9 20:00:04 2020

@author: Liu
"""

import numpy as np
import itertools
from preprocess import get_train
from settings import SIZES, HIDDEN_NEURONS, VALI_R, H3P2_GRID
from nn import NeuralNetwork, DenseLayer

train_db = get_train()

if not H3P2_GRID.is_file():
    with open(str(H3P2_GRID), 'w') as f:
        f.write("eta,alpha,lamda,error\n")



etas = [0.02,0.03,0.04,0.05]
alphas = [0.8]
lamdas = [1E-4]

hparams = list(itertools.product(etas,alphas,lamdas))

grid_errors = []
for eta,alpha,lamda in hparams:
    network = NeuralNetwork()
    network.add_layer(DenseLayer(n_input=SIZES['x'][1], n_neurons=HIDDEN_NEURONS, 
                                 activation='sigmoid'))
    network.add_layer(DenseLayer(n_input=HIDDEN_NEURONS, n_neurons=784, 
                                 activation='sigmoid')) 
    errors = network.train(train_db['x'], train_db['x'], max_epochs=50, 
                  classify=False,
                  validation_ratio=VALI_R,
                  learning_rate = eta, 
                  momentum=alpha, 
                  weight_decay=lamda,
                  )
    grid_errors.append([eta, alpha, lamda, np.min(errors[1:])])
    print("\n-------- Search {}/{}: eta={:.3f}, alpha={:.1f}, lamda={:.5f}, min error={:.4f}--------".format(len(grid_errors), len(hparams), eta,alpha,lamda,np.min(errors[1:])))
    with open(str(H3P2_GRID), 'a') as f:
        np.savetxt(f, [grid_errors[-1]], delimiter=',')

for l in grid_errors:
    print(l)
    