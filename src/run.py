import os
import sys
import cv2
from utils import convertRGB_YCbCr
from utils import laplacianStack
from recipe import recipeMaker
def main(args):
    # image load
    wSize = 64
    stepSize = wSize//2

    inputImage = cv2.imread(args[0])
    outputImage = cv2.imread(args[1])

    # Converting RGB image to YCbCr space
    # inputImage = cv2.resize(inputImage, (512, 512))
    # outputImage = inputImage

    inputImageYCbCr = convertRGB_YCbCr(inputImage)
    outputImageYCbCr = convertRGB_YCbCr(outputImage)

    # Create Recipe

    w, l = inputImageYCbCr.shape[:2]
    i, j = 0, 0
    for i in range(0, w - wSize + 1, stepSize):
        for j in range(0, l - wSize + 1, stepSize):
            recipeMaker(inputImageYCbCr[i:i+wSize,j:j+wSize,:],
                        outputImageYCbCr[i:i+wSize,j:j+wSize,:])

        if j + wSize < l:
            recipeMaker(inputImageYCbCr[i:i+wSize,-wSize:,:],
                        outputImageYCbCr[i:i+wSize,-wSize:,:])

    if i + wSize < w:
        for j in range(0, l - wSize + 1, stepSize):
            recipeMaker(inputImageYCbCr[-wSize:,j:j+wSize,:],
                        outputImageYCbCr[-wSize:,j:j+wSize,:])

        if j + wSize < l:
            recipeMaker(inputImageYCbCr[-wSize:,-wSize:,:],
                        outputImageYCbCr[-wSize:,-wSize:,:])


if __name__ == '__main__':
    inputImageLocation = '../images/inp.jpg'
    outputImageLocation = '../images/out.jpg'
    args = [inputImageLocation, outputImageLocation]
    main(args)




