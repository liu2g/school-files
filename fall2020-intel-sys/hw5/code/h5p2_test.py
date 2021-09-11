from preprocess import get_train, get_test, int_to_roman, subplt_size
from settings import CLASSES, PATIENCE
from settings import H4P2C1_NN, H4P2C2_NN, H3P1_NN, H5P2_NN
from settings import H5P2_CM_PLOT, H5P2_TRAIN_PLOT

from nn import NeuralNetwork
import numpy as np
import matplotlib.pyplot as plt

# Load models
train_db = get_train()
test_db = get_test()
autoenc_clean = NeuralNetwork()
autoenc_clean.load(H4P2C1_NN)

autoenc_noise = NeuralNetwork()
autoenc_noise.load(H4P2C2_NN)


classifier = NeuralNetwork()
classifier.load(H3P1_NN)

sofm = NeuralNetwork()
sofm.load(H5P2_NN)


# Plot training error vs epoch
train_errors = sofm.train_errors
epochs = 10*np.arange(len(train_errors))
fig1, ax1 = plt.subplots(figsize=(8,6))
ax1.plot(epochs, train_errors)
ax1.set_xticks(np.append(np.arange(0,epochs[-1],20),epochs[-1]))
ax1.set_xlabel("Epoch")
ax1.set_ylabel("Train Error")
ax1.set_title("SOFM-Based Classifier")
fig1.text(0.02, 0.01, '* Training early stopped at epoch {},' 
          'then restored to epoch {}, when error is {:.3f}.\n'
          .format(epochs[-1], epochs[-1-PATIENCE], train_errors[-1-PATIENCE],
                  ), ha='left')
fig1.tight_layout(rect=[0, 0.08, 1, 0.95])
fig1.savefig(H5P2_TRAIN_PLOT)

models = {"SOFM Classifier": sofm, "Reconstructing Classifier": autoenc_clean, 
          "Denoising Classifier": autoenc_noise, "BP Classifier": classifier}
fig2, ax2 = plt.subplots(2,4, figsize=subplt_size((2,4),(5,5)))
# Plot confusion matrix
for m, (name, model) in enumerate(models.items()):
    cm = model.get_cm(train_db['x'], train_db['y']), model.get_cm(test_db['x'], test_db['y'])
    errors = model.classify_test(train_db['x'], train_db['y']), model.classify_test(test_db['x'], test_db['y'])
    for n in range(2):
        ax2[n,m].imshow(cm[n], cmap='Blues')
        ax2[n,m].set_xticks(CLASSES)
        ax2[n,m].set_yticks(CLASSES)
        ax2[n,m].set_xticklabels(CLASSES)
        ax2[n,m].set_yticklabels(CLASSES)
        ax2[n,m].tick_params(axis=u'both', which=u'both',length=0)
        for i in range(len(CLASSES)):
            for j in range(len(CLASSES)):
                c = 'w' if cm[n][i,j]>=np.sum(cm[n][:,0]/2) else 'k'
                text = ax2[n,m].text(j, i, int(cm[n][i, j]), ha="center", va="center", color=c, fontsize=12)
        ax2[n,m].set_xlabel("True Class\n({})".format(int_to_roman(n+m*2+1)), fontsize=12)
        ax2[n,m].set_ylabel("Predicted Class", fontsize=10)
        for num in CLASSES:
            ax2[n,m].axvline(num-0.5, c='cornflowerblue', lw=1.5, alpha=0.3)
            ax2[n,m].axhline(num-0.5, c='cornflowerblue', lw=1.5, alpha=0.3)
    ax2[0,m].set_title("{} on Train Data \n(Overall Accuracy = {:.3f})".format(name, 1-errors[0]))
    ax2[1,m].set_title("{} on Test Data \n(Overall Accuracy = {:.3f})".format(name, 1-errors[1]))
        

fig2.tight_layout(rect=[0, 0, 1, 0.96])
fig2.savefig(H5P2_CM_PLOT)

plt.show()
plt.close('all')