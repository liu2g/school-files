from preprocess import get_train, get_test, int_to_roman
from settings import CLASSES, H4P2C1_NN, H4P2C2_NN, H4P2_CM_PLOT, H4P2_TRAIN_PLOT, PATIENCE, H3P1_NN
from nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# Load the autoenc_clean
train_db = get_train()
test_db = get_test()
autoenc_clean = NeuralNetwork()
autoenc_clean.load(H4P2C1_NN)

autoenc_noise = NeuralNetwork()
autoenc_noise.load(H4P2C2_NN)

# Plot training error vs epoch
train_errors = autoenc_clean.train_errors, autoenc_noise.train_errors
epochs = 10*np.arange(len(train_errors[0])), 10*np.arange(len(train_errors[1]))
fig1, ax1 = plt.subplots(1,2,figsize=(12,6))
for i in range(2):
    ax1[i].plot(epochs[i], train_errors[i])
    ax1[i].set_xticks(np.append(np.arange(0,epochs[i][-1],20),epochs[i][-1]))
    ax1[i].set_xlabel("Epoch\n({})".format(int_to_roman(i+1)))
    ax1[i].set_ylabel("Train Error")
ax1[0].set_title("Reconstructing Classifier$^*$")
ax1[1].set_title("Denoising Classifier$^†$")
fig1.text(0.02, 0.01, '* Training early stopped at epoch {},' 
          'then restored to epoch {}, when error is {:.3f}.\n'
          '† Training early stopped at epoch {},' 
          'then restored to epoch {}, when error is {:.3f}.\n'
          .format(epochs[0][-1], epochs[0][-1-PATIENCE], train_errors[0][-1-PATIENCE],
                  epochs[1][-1], epochs[1][-1-PATIENCE], train_errors[1][-1-PATIENCE],
                  ), ha='left')
fig1.tight_layout(rect=[0, 0.08, 1, 0.95])
fig1.savefig(H4P2_TRAIN_PLOT)

# Plot confusion metrix
classifier = NeuralNetwork()
classifier.load(H3P1_NN)

cm = [[0 for _ in range(3)] for _ in range(2)]
cm[0][0] = autoenc_clean.get_cm(train_db['x'], train_db['y'])
cm[1][0] = autoenc_clean.get_cm(test_db['x'], test_db['y'])
cm[0][1] = autoenc_noise.get_cm(train_db['x'], train_db['y'])
cm[1][1] = autoenc_noise.get_cm(test_db['x'], test_db['y'])
cm[0][2] = classifier.get_cm(train_db['x'], train_db['y'])
cm[1][2] = classifier.get_cm(test_db['x'], test_db['y'])

errors = [[0 for _ in range(3)] for _ in range(2)]
errors[0][0] = autoenc_clean.classify_test(train_db['x'], train_db['y'])
errors[1][0] = autoenc_clean.classify_test(test_db['x'], test_db['y'])
errors[0][1] = autoenc_noise.classify_test(train_db['x'], train_db['y'])
errors[1][1] = autoenc_noise.classify_test(test_db['x'], test_db['y'])
errors[0][2] = classifier.classify_test(train_db['x'], train_db['y'])
errors[1][2] = classifier.classify_test(test_db['x'], test_db['y'])

fig2, ax2 = plt.subplots(2,3, figsize=(18,12))
for m in range(2):
    for n in range(3):
        ax2[m,n].imshow(cm[m][n], cmap='Blues')
        ax2[m,n].set_xticks(CLASSES)
        ax2[m,n].set_yticks(CLASSES)
        ax2[m,n].set_xticklabels(CLASSES)
        ax2[m,n].set_yticklabels(CLASSES)
        ax2[m,n].tick_params(axis=u'both', which=u'both',length=0)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                c = 'w' if cm[m][n][i,j]>=50 else 'k'
                text = ax2[m,n].text(j, i, int(cm[m][n][i, j]), ha="center", va="center", color=c, fontsize=12)
        ax2[m,n].set_xlabel("True Class\n({})".format(int_to_roman(n+m*2+1)), fontsize=14)
        ax2[m,n].set_ylabel("Predicted Class", fontsize=14)
        for num in CLASSES:
            ax2[m,n].axvline(num-0.5, c='cornflowerblue', lw=1.5, alpha=0.3)
            ax2[m,n].axhline(num-0.5, c='cornflowerblue', lw=1.5, alpha=0.3)

ax2[0,0].set_title("Reconstructing Classifier on Train Data\n(Overall Accuracy = {:.3f})".format(1-errors[0][0]))
ax2[1,0].set_title("Reconstructing Classifier on Test Data\n(Overall Accuracy = {:.3f})".format(1-errors[1][0]))
ax2[0,1].set_title("Denoising Classifier on Train Data\n(Overall Accuracy = {:.3f})".format(1-errors[0][1]))
ax2[1,1].set_title("Denoising Classifier on Test Data\n(Overall Accuracy = {:.3f})".format(1-errors[1][1]))
ax2[0,2].set_title("BP Classifier on Train Data\n(Overall Accuracy = {:.3f})".format(1-errors[0][2]))
ax2[1,2].set_title("BP Classifier on Test Data\n(Overall Accuracy = {:.3f})".format(1-errors[1][2]))
fig2.tight_layout(rect=[0, 0, 1, 0.96])

fig2.savefig(H4P2_CM_PLOT)

plt.show()
plt.close('all')