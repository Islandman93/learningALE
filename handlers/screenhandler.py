__author__ = 'Ben'
# Parts of this code is lovingly copied from http://pydoc.net/Python/Desktopmagic/14.3.11/desktopmagic.screengrab_win32/
# the author does a great job of handling exceptions
import win32gui
import win32ui
import win32con
import win32api
from PIL import Image
from scipy.misc import imread, imresize, imsave
import os
import numpy as np
from ncorr import ncorr
import copy


class ScreenHandler():
    def __init__(self):
        self.width = 650
        self.height = 480
        self.left = 552
        self.top = 345

        self.oldPScoreImg = None
        self.oldOScoreImg = None
        self.pScore = 0
        self.oScore = 0

        self.resizeVal = 0.25

        self.winImg = np.asarray(imread('images/win.png'), dtype=float)
        self.winImgPlayer = np.asarray(imread('images/winplayer.png'), dtype=float)

    def screenCap(self):
        # Code copied from http://pydoc.net/Python/Desktopmagic/14.3.11/desktopmagic.screengrab_win32/
        hwndDesktop = win32gui.GetDesktopWindow()

        # Retrieve the device context (DC) for the entire virtual screen.
        hwndDevice = win32gui.GetWindowDC(hwndDesktop)
        ##print("device", hwndDevice)
        assert isinstance(hwndDevice, (int)), hwndDevice

        mfcDC = win32ui.CreateDCFromHandle(hwndDevice)
        try:
            saveDC = mfcDC.CreateCompatibleDC()
            saveBitMap = win32ui.CreateBitmap()
            # Above line is assumed to never raise an exception.
            try:
                try:
                    saveBitMap.CreateCompatibleBitmap(mfcDC, self.width, self.height)
                except (win32ui.error, OverflowError) as e:
                    raise GrabFailed("Could not CreateCompatibleBitmap("
                        "mfcDC, %r, %r) - perhaps too big? Error was: %s" % ( self.width, self.height, e))
                saveDC.SelectObject(saveBitMap)
                try:
                    saveDC.BitBlt((0, 0), (self.width, self.height), mfcDC, (self.left, self.top), win32con.SRCCOPY)
                except win32ui.error as e:
                    raise GrabFailed("Error during BitBlt. "
                        "Possible reasons: locked workstation, no display, "
                        "or an active UAC elevation screen. Error was: " + str(e))
                # if saveBmpFilename is not None:
                #     saveBitMap.SaveBitmapFile(saveDC, saveBmpFilename)
            except:
                self.deleteDCAndBitMap(saveDC, saveBitMap)
                # Let's just hope the above line doesn't raise an exception
                # (or it will mask the previous exception)
                raise
        finally:
            mfcDC.DeleteDC()

        bmpinfo = saveBitMap.GetInfo()
        bmpstr = saveBitMap.GetBitmapBits(True)
        barray = np.asarray(Image.frombuffer(
            'RGB',
            (bmpinfo['bmWidth'], bmpinfo['bmHeight']),
            bmpstr, 'raw', 'BGRX', 0, 1))
        barray = barray[:, :, 0:3]
        barray = imresize(barray, 0.25, interp='nearest')
        barray = np.mean(barray, axis=2)

        self.deleteDCAndBitMap(saveDC, saveBitMap)
        win32gui.ReleaseDC(hwndDesktop, hwndDevice)
        return barray

    def isWin(self, img):
        if ncorr(img, self.winImg) > .85 or ncorr(img, self.winImgPlayer) > .85:
            print("looks like end of game")
            return 1
        else:
            return 0

    def isExit(self, img):
        if ncorr(img, self.winImg) < .3 and ncorr(img, self.winImgPlayer) < .3:
            # import matplotlib.pyplot as plt
            # plt.subplot(1,2,1)
            # plt.imshow(img)
            # plt.subplot(1,2,2)
            # plt.imshow(self.winImg)
            # plt.show()
            return 1
        else:
            return 0

    def processScreen(self, inputimg):
        # player score: 910 370 990 425
        newPScoreImg = copy.deepcopy(inputimg[8:18, 89:108])

        # opponent score: 760 370 840 425
        newOScoreImg = copy.deepcopy(inputimg[8:18, 52:71])

        retVal = self.checkScore(newPScoreImg, newOScoreImg)

        # remove score from image
        inputimg[8:18, 89:108] = 0
        inputimg[8:18, 52:71] = 0

        # resize
        inputimg = imresize(inputimg, self.resizeVal, interp='nearest')
        inputimg /= 255
        return inputimg, retVal

    def checkScore(self, newPScoreImg, newOScoreImg):
        retVal = 0

        # check if score increase
        if self.oldPScoreImg is not None:
            compScore = self.compareScore(newPScoreImg, self.oldPScoreImg, 1, self.pScore)
            if compScore < 0.95:
                self.pScore += 1
                retVal = 1
                print("player score increase: {0}".format(self.pScore))
                if not os.path.isfile("images/p{0}.png".format(self.pScore)):
                    imsave("images/p{0}.png".format(self.pScore), newPScoreImg)

        # check if score increase
        if self.oldOScoreImg is not None:
            compScore = self.compareScore(newOScoreImg, self.oldOScoreImg, 0, self.oScore)
            if compScore < 0.95:
                self.oldOScoreImg = newOScoreImg
                self.oScore += 1
                retVal = -1
                print("oppone score increase: {0}".format(self.oScore))
                if not os.path.isfile("images/o{0}.png".format(self.oScore)):
                    imsave("images/o{0}.png".format(self.oScore), newOScoreImg)

        self.oldPScoreImg = newPScoreImg
        self.oldOScoreImg = newOScoreImg

        return retVal

    def compareScore(self, img, oldimg, player, score):
        pstr = ''
        if player:
            pstr = 'p'
        else:
            pstr = 'o'

        filename = "images/"+pstr+str(score+1)+".png"
        if os.path.isfile(filename):
            compimg = np.asarray(imread(filename), dtype=float)
            val = ncorr(compimg, img)
            if val > .95:
                return 0
            else:
                return 1
        else:
            return ncorr(oldimg, img)

    def resetScore(self):
        self.pScore = 0
        self.oldPScoreImg = None
        self.oScore = 0
        self.oldOScoreImg = None

    # Copied from http://pydoc.net/Python/Desktopmagic/14.3.11/desktopmagic.screengrab_win32/
    def deleteDCAndBitMap(self, savedc, bitmap):
        savedc.DeleteDC()
        handle = bitmap.GetHandle()
        # Trying to DeleteObject(0) will throw an exception; it can be 0 in the case
        # of an untouched win32ui.CreateBitmap()
        if handle != 0:
            win32gui.DeleteObject(handle)

    def getAndProcessScreen(self):
        image = self.screenCap()
        processImg, reward = self.processScreen(image)
        isWin = self.isWin(image)
        isExit = self.isExit(image)
        return processImg, reward, isWin, isExit


# Copied from http://pydoc.net/Python/Desktopmagic/14.3.11/desktopmagic.screengrab_win32/
class GrabFailed(Exception):
    """
    Could not take a screenshot.
    """