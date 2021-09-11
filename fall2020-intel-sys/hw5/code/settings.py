from pathlib import Path

# Folders
ROOT_PATH = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT_PATH / "data"
CODE_PATH = ROOT_PATH / "code"
IMG_PATH = ROOT_PATH / "img"
MODEL_PATH = ROOT_PATH / "model"

# Files
X_FILE = DATA_PATH / "MNISTnumImages5000_balanced.txt"
Y_FILE = DATA_PATH / "MNISTnumLabels5000_balanced.txt"
TRAIN_FILE = DATA_PATH / "MNIST_train.json"
TEST_FILE = DATA_PATH / "MNIST_test.json"

H3P1_GRID = MODEL_PATH / "h3p1_grid.csv"
H3P1_CM_PLOT = IMG_PATH / "h3p1_cm.png"
H3P1_NN = MODEL_PATH / "h3p1.json"
H3P1_TRAIN_PLOT = IMG_PATH / "h3p1_train.png"
H3P1_FEATURE_MAP = IMG_PATH / "h3p1_feature.png"

H3P2_GRID = MODEL_PATH / "h3p2_grid.csv"
H3P2_NN = MODEL_PATH / "h3p2.json"
H3P2_TRAIN_PLOT = IMG_PATH / "h3p2_train.png"
H3P2_TEST_PLOT = IMG_PATH / "h3p2_test.png"
H3P2_FEATURE_MAP = IMG_PATH / "h3p2_feature.png"
H3P2_OUTPUT_MAP = IMG_PATH / "h3p2_outputs.png"

H4P1_GRID = MODEL_PATH / "h4p1_grid.csv"
H4P1_NN = MODEL_PATH / "h4p1.json"
H4P1_NN = MODEL_PATH / "h4p1.json"
H4P1_TRAIN_PLOT = IMG_PATH / "h4p1_train.png"
H4P1_TEST_PLOT = IMG_PATH / "h4p1_test.png"
H4P1_FEATURE_MAP = IMG_PATH / "h4p1_feature.png"
H4P1_OUTPUT_MAP = IMG_PATH / "h4p1_outputs.png"

H4P2_GRID = MODEL_PATH / "h4p2_grid.csv"
H4P2_CM_PLOT = IMG_PATH / "h4p2_cm.png"
H4P2C1_NN = MODEL_PATH / "h4p2c1.json"
H4P2C2_NN = MODEL_PATH / "h4p2c2.json"
H4P2_TRAIN_PLOT = IMG_PATH / "h4p2_train.png"
H4P2_FEATURE_MAP = IMG_PATH / "h4p2_feature.png"

H5P1_GRID = MODEL_PATH / "h5p1_grid.csv"
H5P1_SOFM = MODEL_PATH / "h5p1.json"
H5P1_HEAT_MAP = IMG_PATH / "h5p1_heat.png"
H5P1_FEATURE_MAP = IMG_PATH / "h5p1_feature.png"

H5P2_GRID = MODEL_PATH / "h5p2_grid.csv"
H5P2_NN = MODEL_PATH / "h5p2.json"
H5P2_CM_PLOT = IMG_PATH / "h5p2_cm.png"
H5P2_TRAIN_PLOT = IMG_PATH / "h5p2_train.png"

# Metadata
CLASSES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
SIZES = {
    'classes':10,
    'x':(5000,784,28),
    'y':5000,
    'onehot':(5000,10),
    'pts':5000,
    'pts_per_class':500,
    'train':4000,
    'test':1000,
    'train_per_class':400,
    'test_per_class':100,
    }

# Hyperparameters
HIDDEN_NEURONS = 144
MAX_EPOCHS = 500
VALI_R = 0.25
PATIENCE = 5
NOISE = "s&p"