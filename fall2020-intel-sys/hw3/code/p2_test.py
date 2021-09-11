#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:36:30 2020

@author: liu
"""

from preprocess import get_train, get_test, get_rand_list, prepare_img, find_ES
from settings import CLASSES, SIZES, MAX_EPOCHS
from settings import H3P2_NN, H3P2_TRAIN_PLOT, H3P2_TEST_PLOT, H3P2_FEATURE_MAP, H3P2_OUTPUT_MAP, H3P1_NN, H3P1_FEATURE_MAP, HIDDEN_NEURONS
from nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# Load the network
train_db = get_train()
test_db = get_test()
autoenc = NeuralNetwork()
autoenc.load(str(H3P2_NN))

# Plot training error vs epoch
train_errors = autoenc.train_errors
epochs = 10*np.arange(len(train_errors))
fig1, ax1 = plt.subplots(figsize=(8,6))
ax1.plot(epochs, train_errors)
ax1.set_xticks(np.append(np.arange(0,epochs[-1],20),epochs[-1]))
ax1.set_title("2-Layer NN Autoencoder: Error vs Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train Error")
notetxt = find_ES(train_errors, MAX_EPOCHS)
fig1.text(0.02, 0.01, "* Note:"+notetxt, ha='left')
fig1.tight_layout(rect=[0, 0.02, 1, 1])
fig1.savefig(H3P2_TRAIN_PLOT)

# Plot testing errors
test_errors = [[] for _ in CLASSES]
train_errors = [[] for _ in CLASSES]
for i,x in enumerate(train_db['x']):
    c = np.argmax(train_db['y'][i])
    train_errors[c].append(autoenc.raw_test([x],[x])) 
for i,x in enumerate(test_db['x']):
    c = np.argmax(test_db['y'][i])
    test_errors[c].append(autoenc.raw_test([x],[x])) 
test_errors = np.mean(test_errors,axis=1)
test_errors = np.insert(test_errors, 0, np.mean(test_errors))
train_errors = np.mean(train_errors,axis=1)
train_errors = np.insert(train_errors, 0, np.mean(train_errors))
fig2, ax2 = plt.subplots(figsize=(8,6))
width = 0.35
ticks = [str(c) for c in CLASSES]
ticks.insert(0,'Overall')
ax2.bar(np.arange(len(ticks)) - width/2, train_errors, width, label='Train Errors')
ax2.bar(np.arange(len(ticks)) + width/2, test_errors, width, label='Test Errors')
ax2.set_xticks(np.arange(len(ticks)))
ax2.set_xticklabels(ticks)
ax2.set_ylabel('Test Error')
ax2.set_title("2-Layer NN Autoencoder: Performance on Each Class")
ax2.legend(loc='lower right')
ax2.grid(axis='y')
fig2.tight_layout()
fig2.savefig(H3P2_TEST_PLOT)

# Plot feature maps
fig3, ax3 = plt.subplots(5,4, figsize=(8,10))
neuron_i = get_rand_list(HIDDEN_NEURONS)[:20]
for i,ni in enumerate(neuron_i):
    features = autoenc.layers(0).weights[:,ni][1:]
    features = features / np.linalg.norm(features)
    ax3[i//4][i%4].imshow(prepare_img(features), cmap='binary')
    ax3[i//4][i%4].set_xticks([])
    ax3[i//4][i%4].set_yticks([])
    ax3[i//4][i%4].set_xlabel('({})'.format(chr(ord('a')+i)), fontsize=14)
fig3.suptitle("2-Layer NN Autoencoder: Hidden Neuron Features")
fig3.tight_layout(rect=[0, 0, 1, 0.95])
fig3.savefig(H3P2_FEATURE_MAP)

classifier = NeuralNetwork() 
classifier.load(str(H3P1_NN))
fig4, ax4 = plt.subplots(5,4, figsize=(8,10))
for i,ni in enumerate(neuron_i):
    features = classifier.layers(0).weights[:,ni][1:]
    ax4[i//4][i%4].imshow(prepare_img(features), cmap='binary')
    ax4[i//4][i%4].set_xticks([])
    ax4[i//4][i%4].set_yticks([])
    ax4[i//4][i%4].set_xlabel('({})'.format(chr(ord('a')+i)), fontsize=14)
fig4.suptitle("2-Layer NN Classifier: Hidden Neuron Features")
fig4.tight_layout(rect=[0, 0, 1, 0.95])
fig4.savefig(H3P1_FEATURE_MAP)

# Plot sample output
img_i = get_rand_list(SIZES['test'])[:8]
fig5, ax5 = plt.subplots(2,8, figsize=(16,4))
for i, ii in enumerate(img_i):
    original = test_db['x'][ii]
    ax5[0][i].imshow(prepare_img(original), cmap='binary')
    reconstructed = autoenc.predict([test_db['x'][ii]])
    ax5[1][i].imshow(prepare_img(reconstructed), cmap='binary')
    ax5[0][i].set_xticks([])
    ax5[0][i].set_yticks([])
    ax5[0][i].set_xlabel('({})'.format(chr(ord('a')+i)), fontsize=14)
    ax5[1][i].set_xticks([])
    ax5[1][i].set_yticks([])
    ax5[1][i].set_xlabel('({})'.format(chr(ord('i')+i)), fontsize=14)
ax5[0][0].set_ylabel("Original")
ax5[1][0].set_ylabel("Reconstructed")
fig5.suptitle("2-Layer NN Autoencoder: Sample Outputs")
fig5.tight_layout(rect=[0, 0, 1, 0.95])
fig5.savefig(H3P2_OUTPUT_MAP)
    

plt.show()
plt.close('all')