__author__ = 'ben'
import os
import pickle
import numpy as np
with open(os.getcwd() + '\spc_inv_dataset1d.pkl', 'rb') as inFile:
    states2, actions2, rewards2 = pickle.load(inFile)
rewards2 = np.asarray(rewards2)
rewards2[rewards2>1] = 1
print(sum(sum(rewards2)))
#
# for s in states:
#     states2.append(s)
#
# for s in rewards:
#     rewards2.append(s)
#
# for s in actions:
#     actions2.append(s)
#
# with open(os.getcwd() + '\spc_inv_dataset20.pkl', 'wb') as outFile:
#     pickle.dump((states2, actions2, rewards2), outFile)