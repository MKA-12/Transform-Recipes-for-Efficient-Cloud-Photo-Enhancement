import cv2
import numpy as np


def convertRGB_YCbCr(im):
    return cv2.cvtColor(im, cv2.COLOR_RGB2YCR_CB)


def laplacianStack(img):

    stackSize = int(np.log2(img.shape[0]))
    out = np.copy(img)
    pyramid = [out]
    # pyramid creation through downsampling
    for i in range(stackSize):
        out = cv2.pyrDown(out)
        pyramid.append(out)

    # pyramid
    stack = []
    for i in range(1,len(pyramid)):
        downImg = pyramid[i]
        upsampledImg = cv2.pyrUp(downImg)
        laplacianImg = pyramid[i-1] - upsampledImg
        for j in range(i-1):
            laplacianImg = cv2.pyrUp(laplacianImg)
        stack.append(laplacianImg)
    stack = np.array(stack)
    return stack, np.array(pyramid[-1]).reshape(3).astype('int')

