import numpy as np
import csv
import settings
import json
import pandas as pd
import itertools


def prepare_img(img_a):
    img_a = 1- np.array(img_a).flatten()
    img_a = (img_a - np.min(img_a)) / (np.max(img_a) - np.min(img_a))
    # img_a = img_a / np.linalg.norm(img_a)
    return np.reshape(img_a, (-1, int(len(img_a)**0.5)), order='F')

def get_rand_list(length):
    return np.random.choice(length,length,replace=False).astype(int)

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
    
def add_noise(img, noise_type):
    img = np.array(img)
    noise_type = noise_type.lower()
    if noise_type == "gaussian":
        mu = 0.5
        sigma = np.sqrt(0.001)
        return img + np.random.normal(mu,sigma, len(img))
    elif noise_type == "s&p":
        density = 0.5
        svp = 1/8
        rand_i = get_rand_list(len(img))[:int(len(img)*density)]
        salt_i = rand_i[:int(len(rand_i)*svp/(svp+1.0))]
        pepper_i = rand_i[int(len(rand_i)/(svp+1.0)):]
        img[salt_i] = 0.0
        img[pepper_i] = 1.0
        return img
    elif noise_type == "poisson":
        vals = len(np.unique(img))
        vals = 2 ** np.ceil(np.log2(vals))
        return np.random.poisson(img * vals) / float(vals)
    elif noise_type == "speckle":
        return img + img*np.random.uniform(0,1)

def int_to_roman(num):
    result = ''
    mapping = {1000:'M', 900:'CM', 500:'D', 400:'CD', 100:'C', 90:'XC', 50:'L', 40:'XL', 10:'X', 9:'IX', 5:'V', 4:'IV', 1:'I'}
 
    while num != 0:
        for k, v in mapping.items():
            if num >= k:
                dividend = int(num/k)
                num %= k
                result += dividend*v
    return result.lower()

def subplt_size(pane_shape, plt_shape):
    return tuple(np.flip(np.multiply(pane_shape,plt_shape)))

def grid_combo(*args):
    return list(itertools.product(*args))

if __name__ == '__main__':
    prepare_data()
    
    # x = get_train()['x']
    # i = int(get_rand_list(len(x))[0])
    # fig, ax = plt.subplots(1,2,figsize=(8,6))
    # ax[0].imshow(prepare_img(x[i]),cmap='binary')
    # ax[1].imshow(prepare_img(add_noise(x[i],"s&p")), cmap='binary')
    
    # print(int_to_roman(15))
    pass
