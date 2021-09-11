from preprocess import get_train
from settings import SIZES, HIDDEN_NEURONS, MAX_EPOCHS, VALI_R, H5P1_SOFM, H5P2_NN, PATIENCE
from nn import NeuralNetwork, DenseLayer
from sofm import SOFM


train_db = get_train()
sofm_layer = SOFM((12,12))
sofm_layer.load(H5P1_SOFM)

network = NeuralNetwork()
network.add_layer(sofm_layer)
network.add_layer(DenseLayer(n_input=HIDDEN_NEURONS, n_neurons=SIZES['classes'], 
                              activation='sigmoid',
                              zero_bias=True,
                              )) 

network.train(train_db['x'], train_db['y'], max_epochs=MAX_EPOCHS, 
                classify=True, 
                threshold=0.25, 
                validation_ratio=VALI_R, 
                earlystop=(0,PATIENCE),
                learning_rate = 0.1, 
                momentum=0.8, 
                weight_decay=1E-5,
                )
network.save(H5P2_NN)