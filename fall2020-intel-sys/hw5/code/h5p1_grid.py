#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 22:09:20 2020

@author: liu
"""

from preprocess import get_train, grid_combo
from settings import H5P1_GRID
from sofm import SOFM
import numpy as np


x = get_train()['x']

if H5P1_GRID.is_file():
    H5P1_GRID.unlink()
with open(str(H5P1_GRID), 'w') as f:
    f.write("eta,sigma,pca,error\n")

# etas = np.outer(np.arange(1,10,2), [0.1,0.01]).flatten(order='F')
etas = [0.1]
# sigmas = np.arange(1,10,1)
sigmas = [8.5]
pcas = np.outer(np.arange(1,10,1), [1,10]).flatten(order='F')
pcas = np.insert(pcas, 0, 0)

hparams = grid_combo(etas, sigmas, pcas)

for i, (e, s, p) in enumerate(hparams):
    model = SOFM((12,12))
    model.initialize(x, pca=p)  
    model.train(x, 
                max_epochs=10, 
                learning_rate = e,
                learning_rate_decay= 1000,
                learning_width = s,
                learning_width_decay= 1000,
                online_test = False
                )
    
    error = model.test(x)
    print('Search {}/{}: eta={:.3e}, sigma={:.3e}, pca={:.3e}, '
          'min error={:.4e}'.format(i+1, len(hparams), e,s,p,error))
    
    with open(str(H5P1_GRID), 'a') as f:
        np.savetxt(f, [[e,s,p,error]], delimiter=',')
