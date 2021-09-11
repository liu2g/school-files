# EECE-6036 Intelligent Systems HW3

## Directory

```
.
├── HW4_report.pdf
├── code
│  ├── requirements.txt --- Package info
│  ├── nn.py --- Module to implement nn
│  ├── h3p1_train.py --- To train nn for HW3 Problem 1
│  ├── h3p2_train.py -- To train nn for HW4 Problem 2
│  ├── h4p1_train.py --- To train nn for Problem 1
│  ├── h4p1_test.py --- To test nn for Problem 1
│  ├── h4p2_train.py --- To train nn for Problem 2
│  ├── h4p2_test.py --- To test nn for Problem 2
│  ├── preprocess.py --- To partition dataset
│  └── settings.py --- Contains directory info
├── data --- Contains the dataset
│  ├── MNISTnumImages5000_balanced.txt --- Original dataset (images)
│  ├── MNISTnumLabels5000_balanced.txt --- Original dataset (labels)
│  ├── MNIST_test.json --- Partitioned data for testing
│  ├── MNIST_train.json --- Partitioned data for training
│  └── showMNISTnum_balanced100.m --- Example MATLAB script
├── model --- Contains info for the model
│  ├── h3p1.json --- Model info for HW3 Problem 1
│  ├── h3p2.json --- Model info for HW3 Problem 2
│  ├── h4p1.json --- Model info for Problem 1
│  ├── h4p2c1.json --- Model info for Problem 2 Case 1
│  └── h4p2c2.json --- Model info for Problem 2 Case 2
├── img
│   ├── first100_balanced.jpg --- Example image file
│   ├── h4p1_feature.png --- Feature map for autoencoders
│   ├── h4p1_outputs.png --- Sample output for autoencoders
│   ├── h4p1_test.png --- Bar plot for autoencoders
│   ├── h4p1_train.png --- Epoch plot for autoencoders
│   ├── h4p2_cm.png --- Confusion matrix for classifiers
│   └── h4p2_train.png --- Epoch plot for autoencoders
└── README.md
```

## Dependencies

The source code is written in Python 3.7.7 with extra packages listed below. 

```
numpy==1.19.1
matplotlib==3.2.2
pathlib==1.0.1
tqdm==4.50.2
```

The exact version of the packages does not need to match. However, in case of version conflicts, the packages are listed in `code/requirements.txt`. To quickly install the packages, run

```
pip install -r code/requirements.txt
```

or 

```
conda install --file code/requirements.txt
```

in the root directory to resolve the conflicts. 

## Usage

Since the parameters are written within code, simply running the sources renders with standard Python call will execute their respective commands. For example,

```
python code/preprocess.py
python code/h4p1_train.py
python code/h4p1_test
python code/h4p1_train.py
python code/h4p2_test.py
```

Running `preprocess.py` as main reads the dataset, partitions it and saves it under `MNIST_train.json` and `MNIST_test.json`. 

Running `h4p1_train.py` or `h4p2_train.py `as main trains the corresponding models, then saves the network in `model/xxx.json` .

The json file follows json syntax but uses custom format. In the model json files, the 1st line specifies the meta info of the network (eg. learning rate, momentum, etc.), the 2nd to (n+1)-th line details the layers of a n-layer network, where each line stores one layer. **One line in the json file could contain huge amount of data, so open with caution! Some text editor/IDE may try to buffer the whole line and hang your program!!!**

Running `p1_test.py` or `p2_test.py` as main tests the resulting models and saves the result to plots in `img/xxx.png`