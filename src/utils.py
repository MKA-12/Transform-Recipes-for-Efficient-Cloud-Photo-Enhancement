import cv2

def convertRGB_YCbCr(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2YCR_CB)


