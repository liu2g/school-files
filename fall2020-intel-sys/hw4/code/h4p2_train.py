from preprocess import get_train
from settings import SIZES, H3P2_NN, HIDDEN_NEURONS, MAX_EPOCHS, VALI_R, H4P1_NN, H4P2C1_NN, H4P2C2_NN, PATIENCE
from nn import NeuralNetwork, DenseLayer


train_db = get_train()

nn_clean = NeuralNetwork()
nn_clean.load(H3P2_NN)
nn_clean.pop_layer()
for layer in nn_clean.layers():
    layer.set_trainable(False)
nn_clean.add_layer(DenseLayer(n_input=HIDDEN_NEURONS, n_neurons=SIZES['classes'], 
                              activation='sigmoid')) 

nn_noise = NeuralNetwork()
nn_noise.load(H4P1_NN)
nn_noise.pop_layer()
for layer in nn_noise.layers():
    layer.set_trainable(False)
nn_noise.add_layer(DenseLayer(n_input=HIDDEN_NEURONS, n_neurons=SIZES['classes'], 
                             activation='sigmoid')) 


nn_clean.train(train_db['x'], train_db['y'], max_epochs=MAX_EPOCHS, 
                classify=True, threshold=0.25, 
                validation_ratio=VALI_R, earlystop=(0,PATIENCE),
                learning_rate = 0.002, 
                momentum=0.8, 
                weight_decay=1E-4,
                )
nn_clean.save(H4P2C1_NN)


nn_noise.train(train_db['x'], train_db['y'], max_epochs=MAX_EPOCHS, 
               classify=True, threshold=0.25, 
               validation_ratio=VALI_R, earlystop=(0,PATIENCE),
               learning_rate = 0.002, 
               momentum=0.8, 
                weight_decay=1E-4,
               )
nn_noise.save(H4P2C2_NN)