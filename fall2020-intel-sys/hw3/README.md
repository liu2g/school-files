# EECE-6036 Intelligent Systems HW3

## Directory
```
├── HW3_report.pdf --- Homework report
├── code --- Contains Python code to implement homework
│  ├── requirements.txt --- Specifies the packages needed to run the code
│  ├── nn.py ---Neural netork library that contains necessery modules for implementations
│  ├── p1_train.py --- Script to train network for Problem 1 (classifier)
│  ├── p1_test.py --- Script to test network for Problem 1 (classifier)
│  ├── p2_train.py --- Script to train network for Problem 2 (autoencoder)
│  ├── p2_test.py --- Script to test network for Problem 2 (autoencoder)
│  ├── preprocess.py --- Script to partition dataset
│  └── settings.py --- Contains directory info
├── data --- Contains the dataset
│  ├── MNISTnumImages5000_balanced.txt --- Original dataset (images)
│  ├── MNISTnumLabels5000_balanced.txt --- Original dataset (labels)
│  ├── MNIST_test.json --- Partitioned data for testing
│  ├── MNIST_train.json --- Partitioned data for training
│  └── showMNISTnum_balanced100.m --- Example MATLAB script
├── model --- Contains info for the model
│  ├── h3p1.json --- Model info for Problem 1 (classifier)
│  └── h3p2.json --- Model info for Problem 2 (autoencoder)
├── img 
│  ├── first100_balanced.jpg --- Example image file
│  ├── h3p1_cm.png --- Confusion matrix of classifier
│  ├── h3p1_feature.png --- Feature of classifier
│  ├── h3p1_train.png --- Training plot of classifier
│  ├── h3p2_feature.png --- Feature of autoencoder
│  ├── h3p2_outputs.png --- Example output of autoencoder
│  ├── h3p2_test.png --- Testing plot of autoencoder
│  └── h3p2_train.png --- Training plot of autoencoder
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

The exact version of the packages does not need to match but in case of version conflicts, the packages are listed in `code/requirements.txt`. To quickly install the packages, run

```
pip install -r code/requirements.txt
```

or 

```
conda install --file code/requirements.txt
```

in the root directory to resolve the packages. 

## Usage

Running the sources uses standard Python call:

```
python code/preprocess.py
python code/p1_train.py
python code/p1_test
python code/p1_train.py
python code/p2_test.py
```

Running `preprocess.py` as main reads the dataset, partitions it and saves it under `MNIST_train.json` and `MNIST_test.json`. 

Running `p1_train.py` or `p2_train.py `as main trains a classifier or autoencoder network described in the report, then save the network in `model/h3p1.json` or `model/h3p2.json`. 

The json file follows json syntax but uses custom format. The 1st line specifies the meta info of the network (eg. learning rate, momentum, etc), the 2nd to (n+1)-th line details the layers of a n-layer network, where each line stores one layer. **The json files are typically dumped into one line with huge amount of data, so open with caution! Some text editor/IDE may try to buffer the whole line and hang the program**

Running `p1_test.py` or `p2_test.py` as main tests the resulting models and saves the result to plots in `img/*.png`