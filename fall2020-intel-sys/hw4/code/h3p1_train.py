from preprocess import get_train, get_test
from settings import SIZES, H3P1_NN, HIDDEN_NEURONS, MAX_EPOCHS, VALI_R, PATIENCE
from nn import NeuralNetwork, DenseLayer

train_db = get_train()
test_db = get_test()
network = NeuralNetwork()
network.add_layer(DenseLayer(n_input=SIZES['x'][1], n_neurons=HIDDEN_NEURONS, 
                             activation='sigmoid'))
network.add_layer(DenseLayer(n_input=HIDDEN_NEURONS, n_neurons=SIZES['classes'], 
                             activation='sigmoid')) 
network.train(train_db['x'],train_db['y'],max_epochs=MAX_EPOCHS, 
              classify=True, threshold=0.25, 
              validation_ratio=VALI_R, earlystop=(0,PATIENCE),
              learning_rate = 0.002, 
              momentum=0.8, 
              weight_decay=1E-4,
              )
network.save(H3P1_NN)
