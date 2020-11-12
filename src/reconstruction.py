import numpy as np
from utils import laplacianStack, convertRGB_YCbCr
from params import wSize, k
import cv2

def reconstruct(inputImage, recipes):

    inputImageYCbCr = convertRGB_YCbCr(inputImage)
    stepSize = wSize//2
    w, l = inputImageYCbCr.shape[:2]
    idx=0
    idy=0
    for i in range(0, w - wSize + 1, stepSize):
        for j in range(0, l - wSize + 1, stepSize):    
            reconstruct_block(inputImageYCbCr[i:i+wSize,j:j+wSize,:],recipes,idx,idy)
            idy+=1

        if j + wSize < l:
            reconstruct_block(inputImageYCbCr[i:i+wSize,j:j+wSize,:],recipes,idx,idy)
            
            idy+=1
        idy=0
        idx+=1

    if i + wSize < w:
        for j in range(0, l - wSize + 1, stepSize):
            reconstruct_block(inputImageYCbCr[i:i+wSize,j:j+wSize,:],recipes,idx,idy)
            
            idy+=1

        if j + wSize < l:
            reconstruct_block(inputImageYCbCr[i:i+wSize,j:j+wSize,:],recipes,idx,idy)
            
            idy+=1

def reconstruct_block(inputBlock, recipes, idx, idy):

    
    laplacianInpBlock, residualInpBlock = laplacianStack(inputBlock)
    highFreqData_in = np.sum(laplacianInpBlock,axis=0)
    
    residualOutBlock = recipes[0][:,idx,idy]*(residualInpBlock+1)-1

    laplacianInpBlock_Lum = laplacianInpBlock[:,:,:,0]
    laplacianOutBlock_Lum = np.zeros(laplacianInpBlock_Lum.shape)

    for i in range(len(laplacianInpBlock_Lum)):
        laplacianOutBlock_Lum[i] = recipes[4][:,idx,idy][i] * laplacianInpBlock_Lum[i]
    
    laplacianPyramid_Lum = []
    for i in range(len(laplacianInpBlock_Lum)):
        upsampImg = laplacianInpBlock_Lum[i]
        downSampImg = upsampImg
        for j in range(i):
            downSampImg = cv2.pyrDown(downSampImg)
        laplacianPyramid_Lum.append(downSampImg)

    upSampInpBlock = cv2.pyrUp(residualInpBlock[0].reshape(1,1).astype('uint8'))
    for i in range(len(laplacianOutBlock_Lum)-1,-1,-1):
        laplacianPyramid_Lum[i] += upSampInpBlock
        upSampInpBlock = cv2.pyrUp(np.clip(laplacianPyramid_Lum[i],0,255).astype('uint8'))    
    
    HighFreqCb = np.zeros(highFreqData_in[:,:,0].shape)
    HighFreqY = np.zeros(highFreqData_in[:,:,0].shape)
    HighFreqCr = np.zeros(highFreqData_in[:,:,0].shape)
    in_lum=highFreqData_in[:,:,0]
    maxi=np.max(in_lum)
    mini=np.min(in_lum)
    for i in range(highFreqData_in.shape[0]):
        for j in range(highFreqData_in.shape[1]):

            HighFreqCb[i][j] = recipes[1][:,idx,idy][:4] @ highFreqData_in[i][j] + recipes[1][:,idx,idy][-1]
            HighFreqCr[i][j] = recipes[2][:,idx,idy][:4] @ highFreqData_in[i][j] + recipes[2][:,idx,idy][-1]
            HighFreqY[i][j] = recipes[3][:,idx,idy][:4] @ highFreqData_in[i][j] + recipes[3][:,idx,idy][-1] 

            for p in range(1,k):
                yi = mini + p*(maxi-mini)/k
                si = (in_lum[i][j]>=yi)
                si = si*(in_lum[i][j]-yi)
                HighFreqY[i][j] += si * recipes[5][:idx,idy][p-1]        
    # exit(0)
    return residualOutBlock, laplacianPyramid_Lum[0], [HighFreqY,HighFreqCb,HighFreqCr]