# EECE-6036 Intelligent Systems HW5

## Directory

```
.
├── HW4_report.pdf
├── code
│  ├── requirements.txt : Package info
│  ├── nn.py : Module to implement nn
│  ├── sofm.py : Module to implement SOFM
│  ├── h3p1_train.py : To train for HW3 Problem 1
│  ├── h3p2_train.py : To train for HW3 Problem 2
│  ├── h4p1_train.py : To train for HW4 Problem 1
│  ├── h4p2_train.py : To train for HW4 Problem 2
│  ├── h5p1_train.py : To train for Problem 1
│  ├── h5p1_test.py : To test for Problem 1
│  ├── h5p2_train.py : To train for Problem 2
│  ├── h5p2_test.py : To test for Problem 2
│  ├── preprocess.py : To partition dataset
│  └── settings.py : Contains directory info
├── data : Contains the dataset
│  ├── MNISTnumImages5000_balanced.txt : Original dataset (images)
│  ├── MNISTnumLabels5000_balanced.txt : Original dataset (labels)
│  ├── MNIST_test.json : Partitioned data for testing
│  ├── MNIST_train.json : Partitioned data for training
│  └── showMNISTnum_balanced100.m : Example MATLAB script
├── model : Contains info for the model
│  ├── h3p1.json : Model info for HW3 Problem 1
│  ├── h3p2.json : Model info for HW3 Problem 2
│  ├── h4p1.json : Model info for HW4 Problem 1
│  ├── h4p2c1.json : Model info for HW4 Problem 2 Case 1
│  ├── h4p2c2.json : Model info for HW4 Problem 2 Case 2
│  ├── h5p1.json : Model info for Problem 1
│  └── h5p2.json : Model info for Problem 2
├── img
│   ├── h5p1_feature.png : Feature map plot
│   ├── h5p1_heat.png : Heat map plot
│   ├── h5p2_cm.png : Confusion matrix plot
│   └── h5p2_train.png : Traing error vs time plot
└── README.md
```

## Dependencies

The source code is written in Python 3.7.7 with extra packages listed below. 

```
numpy==1.19.1
matplotlib==3.2.2
pathlib==1.0.1
tqdm==4.50.2
cupy==8.1.0
```

**To significantly improve the training efficiency, module `cupy` is used that is essentially `numpy` but with support for CUDA. Therefore in order to satisfy the dependencies, an NVIDIA GPU with CUDA core is required. See https://docs.cupy.dev/en/stable/install.html for installation guide. Dr. Minai has agreed for me to use this package, but I apologize for the inconvenience ahead. **

The exact version of the packages does not need to match. However, in case of version conflicts, the packages are listed in `code/requirements.txt`. To quickly install the packages, run

```
pip install -r code/requirements.txt
```

or 

```
conda install --file code/requirements.txt
```

in this directory to resolve the conflicts. 

## Usage

Since the parameters are written within code, simply running the sources renders with standard Python call will execute their respective commands. For example,

```
python code/preprocess.py
python code/h5p1_train.py
python code/h5p1_test
python code/h5p1_train.py
python code/h5p2_test.py
```

Running `preprocess.py` as main reads the dataset, partitions it and saves it under `MNIST_train.json` and `MNIST_test.json`. 

Running `h5p1_train.py` or `h5p2_train.py `as main trains the corresponding models, then saves the network in `model/xxx.json` .

The json file follows json syntax but uses custom format. In the model json files, the 1st line specifies the meta info of the network (eg. learning rate, momentum, etc.), the 2nd to (n+1)-th line details the layers of a n-layer network, where each line stores one layer. 

**Warning: one line in the json file could contain huge amount of data, so open with caution! Some text editor/IDE may try to buffer the whole line and hang your program!!!**

Running `p1_test.py` or `p2_test.py` as main tests the resulting models and saves the result to plots in `img/xxx.png`