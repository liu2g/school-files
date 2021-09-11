#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov 23 15:30:43 2020

@author: liu
"""

from sofm import SOFM
from settings import H5P1_SOFM, H5P1_HEAT_MAP, H5P1_FEATURE_MAP
from preprocess import get_test, prepare_img, subplt_size, int_to_roman
import matplotlib.pyplot as plt
import numpy as np

imgs = get_test()['x']
labels = get_test()['y']
model = SOFM((12,12))
model.load(H5P1_SOFM)

# Heat map
fig1, ax1 = plt.subplots(2,5,figsize=subplt_size((2,5),(2,2)))
heatmaps = np.zeros((10,144))
for i, img in enumerate(imgs):
    indx_win = model.winner(img).get()
    label = int(np.argmax(labels[i])-1)
    if label == -1:
        label = 9
    heatmaps[label,indx_win] += 1
for i, heatmap in enumerate(heatmaps):
    heatmap = heatmap / np.sum(heatmap)
    ax1[i//5,i%5].imshow(prepare_img(heatmap), cmap='binary')
    ax1[i//5,i%5].set_xticks([])
    ax1[i//5,i%5].set_yticks([])
    ax1[i//5,i%5].set_xlabel('('+int_to_roman(i+1)+')', fontsize=14)
fig1.tight_layout()
fig1.savefig(H5P1_HEAT_MAP)

# Feature map
fig2, ax2 = plt.subplots(12,12, figsize=subplt_size((12,12),(2,2)))
for i in range(144):
    wi = np.unravel_index(i, (12,12), order='F')
    weight = model.get_weights()[:,i]
    ax2[wi].imshow(prepare_img(weight), cmap='binary')
    ax2[wi].set_xticks([])
    ax2[wi].set_yticks([])
fig2.tight_layout()
plt.subplots_adjust(wspace=0, hspace=0.02)
fig2.savefig(H5P1_FEATURE_MAP)



plt.show()
plt.close()
