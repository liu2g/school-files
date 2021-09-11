#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 14:24:38 2020

@author: liu
"""

from preprocess import get_train
from settings import H5P1_SOFM
from sofm import SOFM
import numpy as np

ORG_EPOCHS = 1000
CONV_EPOCHS = 500

x = get_train()['x']
model = SOFM((12,12))

print("Self organizing phase")
model.train(x, 
            max_epochs=ORG_EPOCHS, 
            learning_rate = 0.1,
            learning_rate_decay= ORG_EPOCHS,
            learning_width = 8.5,
            learning_width_decay= ORG_EPOCHS / np.log(8.5)
            )

print("Convergence phase")
model.train(x, 
            max_epochs = CONV_EPOCHS, 
            learning_rate = 0.01,
            learning_rate_decay = CONV_EPOCHS*100,
            learning_width = 0.1,
            learning_width_decay = CONV_EPOCHS*100
            )

model.save(H5P1_SOFM)

