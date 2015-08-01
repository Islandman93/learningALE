__author__ = 'ben'
from scipy.misc import imsave
import time


class SaveHandler():
    def __init__(self):
        self.directory = 'logging/'
        self.logFile = 'log.txt'

    def save(self, img, actionPerformed, reward):
        curTime = time.time()*1000

        # write action and reward for previous action
        with open(self.directory + self.logFile, "a") as output:
            output.write(str(curTime) + "," + str(actionPerformed) + "," + str(reward) + "\n")

        # save image
        imsave(self.directory+"{0}.png".format(curTime), img*255)
