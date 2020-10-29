import os
import sys
import cv2
from utils import convertRGB_YCbCr
def main(args):
    #image load
    inputImage = cv2.imread(args[0])
    outputImage = cv2.imread(args[1])
    
    
    #image convert to YCbCr
    inputImageYCbCr = convertRGB_YCbCr(inputImage)
    outputImageYCbCr = convertRGB_YCbCr(outputImage)
        
    #Create Recipe
    



if __name__ == '__main__':
    inputImageLocation = input("Enter Input Address :")
    outputImageLocation = input("Enter Output Address :")
    args =[inputImageLocation,outputImageLocation]
    mains(args)