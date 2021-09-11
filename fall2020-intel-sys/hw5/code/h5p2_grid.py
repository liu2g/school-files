#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 24 18:54:17 2020

@author: liu
"""

import numpy as np
import itertools
from preprocess import get_train
from settings import SIZES, HIDDEN_NEURONS, VALI_R, H5P2_GRID, H5P1_SOFM
from nn import NeuralNetwork, DenseLayer
from sofm import SOFM

train_db = get_train()
sofm_layer = SOFM((12,12))
sofm_layer.load(H5P1_SOFM)

if not H5P2_GRID.is_file():
    with open(str(H5P2_GRID), 'w') as f:
        f.write("eta,alpha,lamda,error\n")



etas = np.outer(np.arange(1,5,0.25), [0.1]).flatten(order='F')
alphas = [0.8]
lamdas = np.power(0.1,[4,5])

hparams = list(itertools.product(etas,alphas,lamdas))

for i,(e,a,l) in enumerate(hparams):
    network = NeuralNetwork()
    network.add_layer(sofm_layer)
    network.add_layer(DenseLayer(n_input=HIDDEN_NEURONS, n_neurons=SIZES['classes'], 
                                  activation='sigmoid',
                                  zero_bias=True,
                                  )) 
    errors = network.train(train_db['x'], train_db['y'], max_epochs=20, 
                classify=True, 
                threshold=0.25, 
                validation_ratio=VALI_R, 
                # earlystop=(0,PATIENCE),
                learning_rate = e, 
                momentum=a, 
                weight_decay=l,
                )
    error = errors[-1]
    print('Search {}/{}: eta={:.3e}, alpha={:.1e}, lamda={:.5e}, min error='
          '{:.3e}'.format(i+1, len(hparams), e,a,l,error))
    
    with open(str(H5P2_GRID), 'a') as f:
        np.savetxt(f, [[e,a,l,error]], delimiter=',')

