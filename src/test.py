import cv2
import numpy as np
from InputImageCompression import imageCompression
from upscalingInputImage import upscaleInputImage
from utils import convertRGB_YCbCr,convertYCbCr_RGB
import shutil 
import gzip
import os


# Downsize input image
# Compression + creating histograms
readImg = cv2.imread('../images/inp.jpg')
readImg = np.vstack([readImg,np.zeros((1,readImg.shape[1],3),dtype = 'uint8')])
readImg = np.hstack([readImg,np.zeros((readImg.shape[0],4,3),dtype = 'uint8')])


hist_ref,rng_ref = imageCompression(readImg,20)

dir = '../CompressedInputs'
if os.path.exists(dir):
    shutil.rmtree(dir)
os.makedirs(dir)

np.savez_compressed('../CompressedInputs/histograms_ranges', hist_ref = hist_ref,rng_ref = rng_ref,input_shape = readImg.shape)
shutil.copyfile('../images/compressed.jpg','../CompressedInputs/compressed.jpg')
shutil.make_archive('../transferFiles','zip','../CompressedInputs')



# # # Server Side 

shutil.unpack_archive('../transferFiles.zip','../transferFiles',"zip")

load_hist_rng = np.load('../transferFiles/histograms_ranges.npz')
jpegCompressed = cv2.imread('../transferFiles/compressed.jpg')

upscaleInputImage(load_hist_rng['input_shape'],load_hist_rng['hist_ref'],load_hist_rng['rng_ref'],jpegCompressed)




