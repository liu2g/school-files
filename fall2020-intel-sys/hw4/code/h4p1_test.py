from preprocess import get_train, get_test, get_rand_list, prepare_img, add_noise, int_to_roman
from settings import CLASSES, SIZES, PATIENCE, NOISE
from settings import H4P1_NN, H4P1_TRAIN_PLOT, H4P1_TEST_PLOT, H4P1_FEATURE_MAP, H4P1_OUTPUT_MAP, H3P2_NN, HIDDEN_NEURONS
from nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# Load the network
train_db = get_train()
test_db = get_test()
autoenc_noise = NeuralNetwork()
autoenc_noise.load(H4P1_NN)
autoenc_clean = NeuralNetwork()
autoenc_clean.load(H3P2_NN)

# Plot training error vs epoch
train_errors = autoenc_clean.train_errors, autoenc_noise.train_errors
epochs = 10*np.arange(len(train_errors[0])), 10*np.arange(len(train_errors[1]))
fig1, ax1 = plt.subplots(1,2,figsize=(12,6))
for i in range(2):
    ax1[i].plot(epochs[i], train_errors[i])
    ax1[i].set_xticks(np.append(np.arange(0,epochs[i][-1],20),epochs[i][-1]))
    ax1[i].set_xlabel("Epoch")
    ax1[i].set_ylabel("Train Error")
ax1[0].set_title("Reconstructing Autoencoder$^*$")
ax1[1].set_title("Denoising Autoencoder$^†$")
fig1.text(0.02, 0.01, '* Training early stopped at epoch {},' 
          'then restored to epoch {}, when error is {:.3f}.\n'
          '† Training early stopped at epoch {},' 
          'then restored to epoch {}, when error is {:.3f}.\n'
          .format(epochs[0][-1], epochs[0][-1-PATIENCE], train_errors[0][-1-PATIENCE],
                  epochs[1][-1], epochs[1][-1-PATIENCE], train_errors[1][-1-PATIENCE],
                  ), ha='left')
fig1.tight_layout(rect=[0, 0.08, 1, 0.95])
fig1.savefig(H4P1_TRAIN_PLOT)

# Plot training errors
fig2, ax2 = plt.subplots(1,2, figsize=(16,6))

for n in range(2):
    if n == 0:
        autoenc = autoenc_clean
        ax2[n].set_title("Reconstructing Autoencoder")
    else:
        autoenc = autoenc_noise
        ax2[n].set_title("Denoising Autoencoder")
    test_errors = [[] for _ in CLASSES]
    train_errors = [[] for _ in CLASSES]
    for i,x in enumerate(train_db['x']):
        c = np.argmax(train_db['y'][i])
        if n == 0:
            train_errors[c].append(autoenc.raw_test([x],[x])) 
        else:
            train_errors[c].append(autoenc.raw_test([add_noise(x,NOISE)],[x])) 
    for i,x in enumerate(test_db['x']):
        c = np.argmax(test_db['y'][i])
        if n==0:
            test_errors[c].append(autoenc.raw_test([x],[x])) 
        else:
            test_errors[c].append(autoenc.raw_test([add_noise(x,NOISE)],[x])) 
    test_errors = np.mean(test_errors,axis=1)
    test_errors = np.insert(test_errors, 0, np.mean(test_errors))
    train_errors = np.mean(train_errors,axis=1)
    train_errors = np.insert(train_errors, 0, np.mean(train_errors))
    width = 0.35
    ticks = [str(c) for c in CLASSES]
    ticks.insert(0,'Overall')
    ax2[n].bar(np.arange(len(ticks)) - width/2, train_errors, width, label='Train Errors')
    ax2[n].bar(np.arange(len(ticks)) + width/2, test_errors, width, label='Test Errors')
    ax2[n].set_xticks(np.arange(len(ticks)))
    ax2[n].set_xticklabels(ticks)
    ax2[n].set_ylabel('Test Error')
    ax2[n].set_xlabel('('+int_to_roman(n+1)+')')
    ax2[n].legend(loc='lower right')
    ax2[n].grid(axis='y')

fig2.tight_layout(rect=[0, 0, 1, 0.95])
fig2.savefig(H4P1_TEST_PLOT)

# Plot feature maps
fig3, ax3 = plt.subplots(5,9, figsize=(18,10))
neuron_i = get_rand_list(HIDDEN_NEURONS)[:20]
features = [0]*2
for i,ni in enumerate(neuron_i):
    features[0] = autoenc_clean.layers(0).weights[:,ni][1:]
    features[1] = autoenc_noise.layers(0).weights[:,ni][1:]
    ax3[i//4][4].axis('off')
    for j in range(2):
        ax3[i//4][i%4+5*j].imshow(prepare_img(features[j]), cmap='binary')
        ax3[i//4][i%4+5*j].set_xticks([])
        ax3[i//4][i%4+5*j].set_yticks([])
        ax3[i//4][i%4+5*j].set_xlabel('('+int_to_roman(i+20*j+1)+')', fontsize=14)

fig3.suptitle('Reconstructing Autoencoder{}Denoising Autoencoder'.format(' '*123), fontsize=14)
fig3.tight_layout(rect=[0, 0, 1, 0.93])
fig3.savefig(H4P1_FEATURE_MAP)

# Plot sample output

img_i = get_rand_list(SIZES['test'])[:8]
fig4, ax4 = plt.subplots(4,8, figsize=(16,8))
for i, ii in enumerate(img_i):
    clean = test_db['x'][ii]
    ax4[0][i].imshow(prepare_img(clean), cmap='binary')
    ax4[0][i].set_xticks([])
    ax4[0][i].set_yticks([])
    ax4[0][i].set_xlabel('('+int_to_roman(i+0*8+1)+')', fontsize=14)
    
    reconstructed = autoenc_clean.predict([clean])
    ax4[1][i].imshow(prepare_img(reconstructed), cmap='binary')
    ax4[1][i].set_xticks([])
    ax4[1][i].set_yticks([])
    ax4[1][i].set_xlabel('('+int_to_roman(i+1*8+1)+')', fontsize=14)
    
    noisy = add_noise(clean, NOISE)
    ax4[2][i].imshow(prepare_img(noisy), cmap='binary')
    ax4[2][i].set_xticks([])
    ax4[2][i].set_yticks([])
    ax4[2][i].set_xlabel('('+int_to_roman(i+2*8+1)+')', fontsize=14)
    
    denoised = autoenc_noise.predict([noisy])
    ax4[3][i].imshow(prepare_img(denoised), cmap='binary')
    ax4[3][i].set_xticks([])
    ax4[3][i].set_yticks([])
    ax4[3][i].set_xlabel('('+int_to_roman(i+3*8+1)+')', fontsize=14)
    
ax4[0][0].set_ylabel("Original", fontsize=14)
ax4[1][0].set_ylabel("Reconstructed", fontsize=14)
ax4[2][0].set_ylabel("Noisy", fontsize=14)
ax4[3][0].set_ylabel("Denoised", fontsize=14)
fig4.tight_layout(rect=[0, 0, 1, 0.95])
fig4.savefig(H4P1_OUTPUT_MAP)

plt.show()
plt.close('all')