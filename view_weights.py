__author__ = 'islan_000'
import lasagne
import pickle
import numpy as np
from nns import CNN
import matplotlib.pyplot as plt

cnn = CNN()

plt.ion()
for c in range(10, 2660, 10):
    plt.clf()
    print('cnn',c)
    parms = pickle.load((open('libs/cnn{0}.pkl'.format(c), 'rb')))
    lasagne.layers.set_all_param_values(cnn.l_out, parms)

    w = np.asarray(cnn.l_hid1.W.eval())
    for x in range(16):
        plt.subplot(4, 4, x+1)
        plt.imshow(w[x, 0, :, :], interpolation='nearest', cmap=plt.cm.gray)
        plt.colorbar()

    plt.show()
    plt.pause(0.25)