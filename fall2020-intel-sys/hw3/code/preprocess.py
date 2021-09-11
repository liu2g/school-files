#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 24 16:13:36 2020

@author: liu
"""

import numpy as np
import csv
import settings
import json
import pandas as pd

def prepare_img(img_a):
    img_a = np.array(img_a).flatten()
    img_a = 1 - img_a / np.linalg.norm(img_a)
    return np.reshape(img_a, (-1, int(len(img_a)**0.5)), order='F')

def get_rand_list(length):
    length = int(length)
    return np.random.choice(length,length,replace=False)

def prepare_data():
    x_db = []
    with open(str(settings.X_FILE)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            x_db.append([float(x) for x in row])
    
    
    y_db = []
    with open(str(settings.Y_FILE)) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter='\t')
        for row in csv_reader:
            y_db.append(int(row[0]))
    
    print('Distribution of original dataset:',np.bincount(y_db))
    
    train_i, test_i = stratify_split(y_db, 
                                     settings.SIZES['train']/settings.SIZES['y'])
    
    train_db = {'x':[x_db[i] for i in train_i],
                'y':[y_db[i] for i in train_i]}
    
    test_db = {'x':[x_db[i] for i in test_i],
            'y':[y_db[i] for i in test_i]}

    
    print('Distribution of train dataset:',np.bincount(train_db['y']))
    print('Distribution of test dataset:',np.bincount(test_db['y']))
    
    test_db['y'] = np.eye(settings.SIZES['classes'])[test_db['y']].tolist()
    train_db['y'] = np.eye(settings.SIZES['classes'])[train_db['y']].tolist()
    
    with open(str(settings.TRAIN_FILE),'w') as f:
        json.dump(train_db, f)
    
    print("Saved train data in", settings.TRAIN_FILE)
        
    with open(str(settings.TEST_FILE),'w') as f:
        json.dump(test_db, f)
        
    print("Saved test data in", settings.TEST_FILE)
    
def stratify_split(y, ratio):
    if len(np.array(y).shape) > 1: # collapse for one hot
        y = np.argmax(y,axis=1)
    df = pd.DataFrame(y).groupby(0) # Sort data by class
    indxs = [] # buffer for indexes
    for _,g in df:
        indxs.append(g.index.to_numpy()) # indexes of each class take a row
    indxs = np.array(indxs)
    p1_indx = indxs[:, :int(indxs.shape[1]*ratio)].flatten() # partition 1
    np.random.shuffle(p1_indx) # mix index 
    p2_indx = indxs[:, int(indxs.shape[1]*ratio):].flatten() # partition 2
    np.random.shuffle(p2_indx) # mix index 
    return p1_indx, p2_indx
    

def get_test():
    with open(str(settings.TEST_FILE),'r') as f:
        return json.load(f)

def get_train():
    with open(str(settings.TRAIN_FILE),'r') as f:
        return json.load(f)
 
def find_ES(train_errors, max_epochs):
    epochs = 10*np.arange(len(train_errors))
    epochs = np.flip(epochs)
    train_errors = np.flip(train_errors)
    if len(train_errors) < max_epochs:
        return "Training early stopped at epoch {}, then restored to epoch {}, when error is {:.3f}.".format(epochs[0], epochs[np.argmin(train_errors)], np.min(train_errors))
    else:
        return "Last training error is {:.3f}".format(train_errors[0])
    
 
if __name__ == '__main__':
    prepare_data()