__author__ = 'Ben'
import numpy as np

import win32pipe
import win32file
from fifo_old import colorconv

LOAD_STATE = 44
SAVE_STATE = 43
SYSTEM_RESET = 45

# create output pipe
pout = win32pipe.CreateNamedPipe(r'\\.\pipe\alepipein',
    win32pipe.PIPE_ACCESS_OUTBOUND,
    win32pipe.PIPE_TYPE_BYTE | win32pipe.PIPE_WAIT,
    1, 0, 0, 0, None)

# wait for connection
win32pipe.ConnectNamedPipe(pout, None)

# connect to input pipe and read screen data
fileHandle = win32file.CreateFile("\\\\.\\pipe\\alepipeout", win32file.GENERIC_READ, 0, None, win32file.OPEN_EXISTING, 0, None)
data = win32file.ReadFile(fileHandle, 1024)
screenStr = bytes.decode(data[1], encoding='windows-1252')
screenStr = screenStr.split('\n')
screenStr = screenStr[0]
screenStr = screenStr.split('-')
screenWidth = int(screenStr[0])
screenHeight = int(screenStr[1])
print(screenWidth, screenHeight)

# send which parameters I want
readScreen = 1
readRam = 0
readRL = 1
paramStr = '{0},{1},0,{2}\n'.format(readScreen, readRam, readRL)
win32file.WriteFile(pout, str.encode(paramStr))

# while not done
done = False
import time
count = 0
frameCount = 0
st = time.time()
while not done:
    # read ram
    if readRam:
        data = win32file.ReadFile(fileHandle, 204800)

    # read screen
    if readScreen:
        data = win32file.ReadFile(fileHandle, 204800)
        if data[0] == 0:
            hexData = data[1]
            # inspired by https://github.com/kristjankorjus/Replicating-DeepMind/blob/master/src/ale/preprocessor.py
            rgbhexs = [colorconv.color_dic_tuple[hexData[i*2:i*2+2]] for i in range(int(len(hexData)/2))]
            imghex = np.asarray(rgbhexs).reshape((screenHeight, screenWidth, 3))
            # import matplotlib.pyplot as plt
            # plt.subplot(1, 2, 1)
            # plt.imshow(np.mean(imghex,axis=2), cmap=plt.cm.gray)
            # plt.subplot(1, 2, 2)
            # plt.imshow(imghex)
            # plt.show()

    # read state and reward
    if readRL:
        data = win32file.ReadFile(fileHandle, 128)
        if data[0] == 0:
            evnRewardStr = bytes.decode(data[1], encoding='windows-1252')
            evnRewardStr = evnRewardStr.split('\n')
            evnRewardStr = evnRewardStr[0]
            evnRewardStr = evnRewardStr.split(',')
            terminal = evnRewardStr[0]
            terminal = int(terminal)
            reward = int(evnRewardStr[1])

    # if reward > 0:
    #     print('reward', reward)
    #     import matplotlib.pyplot as plt
    #     plt.imshow(d, cmap=plt.cm.gray, interpolation='nearest')
    #     plt.show()

    # send action
    if terminal == 1:
        # done = True
        actionString = '{0},{0}'.format(SYSTEM_RESET)
        win32file.WriteFile(pout, str.encode(actionString))
        print('Game over man game over')
        count += 1
        if count > 9:
            done = True
    else:
        actionString = '{0},{0}'.format(np.random.randint(0, 18))
        win32file.WriteFile(pout, str.encode(actionString))

    frameCount += 1
et = time.time()
print(et-st, count, frameCount, frameCount/(et-st))