import os
import numpy as np

os.chdir('D:\_code\learningALE\experiments\\reproduction\DQN_Original')

epoch_files = list()
for file in os.listdir(os.getcwd()):
    if 'dqn' in file and '.pkl' in file and 'best' not in file:
        epoch_files.append(file)

epoch_dqn = list()
for dqn in epoch_files:
    epoch = dqn.replace('dqn', '')
    epoch = epoch.replace('.pkl', '')
    epoch = round(float(epoch) / 50000, 1)
    epoch_dqn.append([epoch, dqn])

epoch_dqn.sort()

# now test to see best score