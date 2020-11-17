import cv2
import numpy as np
from utils import convertRGB_YCbCr,convertYCbCr_RGB
from histogram import get_histograms
from PIL import Image
 


def imageCompression(inputImg,compressionQuality=20):
    downsampledShape = (inputImg.shape[1]//4,inputImg.shape[0]//4)
    imageDownsample = cv2.resize(inputImg,downsampledShape)
    imageDownsample = np.array(imageDownsample)

    # JPEG compression of input image
    JPEG_Compression(imageDownsample,compressionQuality)

    #Histograms
    I = inputImg
    I = convertRGB_YCbCr(inputImg)
    I = I.astype(np.float32)
    nlevels = 3
    ref_res = 256*10

    hist_ref,rng_ref = get_histograms(I,nlevels,ref_res)

    return hist_ref,rng_ref
    


def JPEG_Compression(img,jpg_quality = 95):
    path = '../images/compressed.jpg'
    cv2.imwrite(path, img, [int(cv2.IMWRITE_JPEG_QUALITY), jpg_quality])



