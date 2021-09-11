#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 27 18:07:34 2020

@author: liu
"""
from preprocess import get_train, get_test, find_ES
from settings import CLASSES, H3P1_NN, H3P1_CM_PLOT, H3P1_TRAIN_PLOT, MAX_EPOCHS
from nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# Load the network
train_db = get_train()
test_db = get_test()
network = NeuralNetwork()
network.load(str(H3P1_NN))

# Plot training error vs epoch
train_errors = network.train_errors
epochs = 10*np.arange(len(train_errors))
fig1, ax1 = plt.subplots(figsize=(8,6))
ax1.plot(epochs, train_errors)
ax1.set_xticks(np.append(np.arange(0,epochs[-1],20),epochs[-1]))
ax1.set_title("2-Layer NN Classifier: Error vs Epoch")
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train Error")
notetxt = find_ES(train_errors, MAX_EPOCHS)
fig1.text(0.02, 0.01, "* Note:"+notetxt, ha='left')
fig1.tight_layout(rect=[0, 0.02, 1, 1])
fig1.savefig(H3P1_TRAIN_PLOT)

# Plot confusion metrix
cm = []
cm.append(network.get_cm(train_db['x'], train_db['y']))
cm.append(network.get_cm(test_db['x'], test_db['y']))
errors =[]
errors.append(network.classify_test(train_db['x'], train_db['y']))
errors.append(network.classify_test(test_db['x'], test_db['y']))
fig2, ax2 = plt.subplots(1,2, figsize=(12,6))
for n in range(2):
    ax2[n].imshow(cm[n], cmap='Blues')
    ax2[n].set_xticks(CLASSES)
    ax2[n].set_yticks(CLASSES)
    ax2[n].set_xticklabels(CLASSES)
    ax2[n].set_yticklabels(CLASSES)
    ax2[n].tick_params(axis=u'both', which=u'both',length=0)
    for i in range(len(CLASSES)):
        for j in range(len(CLASSES)):
            c = 'w' if cm[n][i,j]>=50 else 'k'
            text = ax2[n].text(j, i, int(cm[n][i, j]), ha="center", va="center", color=c, fontsize=12)
    ax2[n].set_xlabel("True Class\n({})".format(chr(ord('a')+n)), fontsize=14)
    ax2[n].set_ylabel("Predicted Class", fontsize=14)
    for num in CLASSES:
        ax2[n].axvline(num-0.5, c='cornflowerblue', lw=1.5, alpha=0.3)
        ax2[n].axhline(num-0.5, c='cornflowerblue', lw=1.5, alpha=0.3)
ax2[0].set_title("Result on Train Data (Overall Accuracy = {:.3f})".format(1-errors[0]))
ax2[1].set_title("Result on Test Data Result (Overall Accuracy = {:.3f})".format(1-errors[1]))
fig2.suptitle("2-Layer NN Classifier: Confusion Matrix")
fig2.tight_layout(rect=[0, 0, 1, 0.92])

fig2.savefig(H3P1_CM_PLOT)

plt.show()
plt.close('all')