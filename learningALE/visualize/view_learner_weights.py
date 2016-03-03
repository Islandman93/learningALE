import pickle
import matplotlib.pyplot as plt
import lasagne
import numpy as np
from learningALE.learners.nns import As
cnn = CNN((None, 3, 86, 80), 6, .1)

plt.ion()
for c in [1466, 5000, 10000, 250005]:
    plt.clf()
    print('cnn',c)
    parms = pickle.load((open('D:\_code\_reinforcementlearning\experiments\\reproduction\DQN_Async_Target\\async1stepdqn{0}.pkl'.format(c), 'rb')))
    lasagne.layers.set_all_param_values(cnn.l_out, parms)

    w = np.asarray(cnn.l_hid1.W.eval())
    # for x in range(16):
    #     plt.subplot(4, 4, x+1)
    #     plt.imshow(w[x, 0, :, :], interpolation='nearest', cmap=plt.cm.gray)
        # plt.colorbar()
    w = np.asarray(cnn.l_hid2.W.eval())
    for x in range(32):
        plt.subplot(6, 6, x+1)
        plt.imshow(w[x, 0, :, :], interpolation='nearest', cmap=plt.cm.gray)
    #     plt.colorbar()

    plt.show()
    plt.pause(0.05)

plt.ioff()
plt.show()