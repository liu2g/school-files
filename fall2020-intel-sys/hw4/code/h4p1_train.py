from preprocess import get_train, get_test, add_noise
from settings import SIZES, H4P1_NN, HIDDEN_NEURONS, MAX_EPOCHS, VALI_R, NOISE, PATIENCE
from nn import NeuralNetwork, DenseLayer

train_db = get_train()
test_db = get_test()
autoenc = NeuralNetwork()
autoenc.add_layer(DenseLayer(n_input=SIZES['x'][1], n_neurons=HIDDEN_NEURONS, 
                             activation='sigmoid'))
autoenc.add_layer(DenseLayer(n_input=HIDDEN_NEURONS, n_neurons=SIZES['x'][1], 
                             activation='sigmoid'))
noisy_x = [add_noise(x,NOISE) for x in train_db['x']]
autoenc.train(noisy_x, train_db['x'], max_epochs=MAX_EPOCHS, 
              classify=False, 
              validation_ratio=VALI_R, earlystop=(1E-3,PATIENCE),
              learning_rate = 0.01, 
              momentum=0.8, 
               weight_decay=1E-4,
              )
autoenc.save(H4P1_NN)
