import numpy as np
from utils import laplacianStack, convertRGB_YCbCr, nearest2power, GaussianKernel, convertYCbCr_RGB
from params import wSize, k, sigma
import cv2
from time import time
from tqdm import tqdm
from matplotlib import pyplot as plt
from sklearn.linear_model import Lasso
from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning
simplefilter("ignore", category=ConvergenceWarning)
def reconstruct(inputImage, recipes):

    inputImageYCbCr = convertRGB_YCbCr(inputImage)
    stepSize = wSize//2
    w, l = inputImageYCbCr.shape[:2]
    idx=0
    idy=0
    # kernel = GaussianKernel()
    kernel = np.ones((wSize,wSize))
    kernel2Layer = np.append(kernel.reshape(wSize,wSize,1),kernel.reshape(wSize,wSize,1),axis=2)
    kernel3Layer = np.append(kernel2Layer,kernel.reshape(wSize,wSize,1),axis=2)
    
    residual_out = np.zeros((inputImage.shape[0],inputImage.shape[1],2))
    OYTemp = np.zeros(inputImage.shape[:2])
    HY = np.zeros(inputImage.shape[:2])
    HCb = np.zeros(inputImage.shape[:2])
    HCr = np.zeros(inputImage.shape[:2])

    kernelSummation = np.zeros(inputImage.shape)
    for i in tqdm(range(0, w - wSize + 1, stepSize)):
        for j in range(0, l - wSize + 1, stepSize):    
            residual_block, OYTemp_block, HY_block, HCb_block, HCr_block=reconstruct_block(inputImageYCbCr[i:i+wSize,j:j+wSize,:],recipes,idx,idy)
            residual_out[i:i+wSize,j:j+wSize,:] += kernel2Layer * residual_block
            OYTemp[i:i+wSize,j:j+wSize] += kernel * OYTemp_block
            HY[i:i+wSize,j:j+wSize] += kernel * HY_block
            HCb[i:i+wSize,j:j+wSize] += kernel * HCb_block
            HCr[i:i+wSize,j:j+wSize] += kernel * HCr_block
            kernelSummation[i:i+wSize,j:j+wSize,:] += kernel3Layer
            idy += 1
            # print(idx,idy)
        if j + wSize < l:
            residual_block, OYTemp_block, HY_block, HCb_block, HCr_block=reconstruct_block(inputImageYCbCr[i:i+wSize,-wSize:,:],recipes,idx,idy)
            residual_out[i:i+wSize,-wSize:,:] += kernel2Layer * residual_block
            OYTemp[i:i+wSize,-wSize:] += kernel * OYTemp_block
            HY[i:i+wSize,-wSize:] += kernel * HY_block
            HCb[i:i+wSize,-wSize:] += kernel * HCb_block
            HCr[i:i+wSize,-wSize:] += kernel * HCr_block 
            kernelSummation[i:i+wSize,-wSize:,:] += kernel3Layer
            
            idy+=1
            # print(idx,idy)
        
        idy=0
        idx+=1

    if i + wSize < w:
        for j in range(0, l - wSize + 1, stepSize):
            residual_block, OYTemp_block, HY_block, HCb_block, HCr_block=reconstruct_block(inputImageYCbCr[-wSize:,j:j+wSize,:],recipes,idx,idy)
            residual_out[-wSize:,j:j+wSize,:] = kernel2Layer * residual_block
            OYTemp[-wSize:,j:j+wSize] = kernel * OYTemp_block
            HY[-wSize:,j:j+wSize] = kernel * HY_block
            HCb[-wSize:,j:j+wSize] = kernel * HCb_block
            HCr[-wSize:,j:j+wSize] = kernel * HCr_block      
            kernelSummation[-wSize:,j:j+wSize,:] += kernel3Layer

            idy+=1
            # print(idx,idy)

        if j + wSize < l:
            residual_block, OYTemp_block, HY_block, HCb_block, HCr_block=reconstruct_block(inputImageYCbCr[-wSize:,-wSize:,:],recipes,idx,idy)
            residual_out[-wSize:,-wSize:,:] = kernel2Layer * residual_block
            OYTemp[-wSize:,-wSize:] = kernel * OYTemp_block
            HY[-wSize:,-wSize:] = kernel * HY_block
            HCb[-wSize:,-wSize:] = kernel * HCb_block
            HCr[-wSize:,-wSize:] = kernel * HCr_block        
            kernelSummation[-wSize:,-wSize:,:] += kernel3Layer

            idy+=1
            # print(idx,idy)

    output = np.zeros(inputImage.shape)
    output[:,:,1:] += residual_out
    output[:,:,0] += OYTemp + HY
    output[:,:,1] += HCb
    output[:,:,2] += HCr

    output /= kernelSummation
    output = np.clip(output,0,255).astype('uint8')
    return convertYCbCr_RGB(output)

def reconstruct_block(inputBlock, recipes, idx, idy):
    
    # --------1
    
    laplacianInpBlock, residualInpBlock = laplacianStack(inputBlock)
    highFreqData_in = np.sum(laplacianInpBlock,axis=0)
    
    residualOutBlock = recipes[0][:,idx,idy]*(residualInpBlock+1)-1

    laplacianInpBlock_Lum = laplacianInpBlock[:,:,:,0]
    laplacianOutBlock_Lum = np.zeros(laplacianInpBlock_Lum.shape)
    
    # ---------2

    for i in range(len(laplacianInpBlock_Lum)):
        laplacianOutBlock_Lum[i] = recipes[4][:,idx,idy][i] * laplacianInpBlock_Lum[i]
    
    # Collapsing laplacian stack to get an intermediate output luminance channel Ô Y
    #----------3
    upSampInpBlock = residualInpBlock[:,:,0].reshape(1,1).astype('uint8')
    for i in range(int(np.log2(wSize))):
        upSampInpBlock = cv2.pyrUp(upSampInpBlock)
    upSampInpBlock = upSampInpBlock.astype('float32')
    
    upSampInpBlock += np.sum(laplacianInpBlock_Lum,axis=0)
    upSampInpBlock = upSampInpBlock.astype('uint8')
    
    HighFreqCb = np.zeros(highFreqData_in[:,:,0].shape)
    HighFreqY = np.zeros(highFreqData_in[:,:,0].shape)
    HighFreqCr = np.zeros(highFreqData_in[:,:,0].shape)
    in_lum=highFreqData_in[:,:,0]
    maxi=np.max(in_lum)
    mini=np.min(in_lum)
    recipe_b = recipes[1][:,idx,idy][:3].reshape(-1,1)
    recipe_c = recipes[2][:,idx,idy][:3].reshape(-1,1)
    recipe_d = recipes[3][:,idx,idy][:3].reshape(-1,1)
    recipe_b_offset = (recipes[1][:,idx,idy][-1])
    recipe_c_offset = (recipes[2][:,idx,idy][-1])
    recipe_d_offset = (recipes[3][:,idx,idy][-1])
    recipe_f = recipes[5][:,idx,idy]
    
    yi = mini + np.arange(1,k)*(maxi-mini)/k
    
    si = np.zeros((in_lum.shape[0],in_lum.shape[1],k-1))
    for i in range(0,k-1):
        si[:,:,i] = (in_lum>=yi[i])*(in_lum-yi[i])*recipe_f[i]
    sum_si = np.sum(si,axis=2)

    # X = highFreqData_in.reshape(wSize*wSize,3)
    # a = time()
    # modelCb=Lasso().fit(np.zeros((wSize*wSize,3)),np.zeros((wSize*wSize,1)))
    # b = time()
    # print(b-a,"Cb fit")
    # modelCb.coef_ = recipe_b.reshape(3,)
    # modelCb.intercept_ = recipe_b_offset.reshape(1,)
    # a = time()
    # HighFreqCb = modelCb.predict(X).reshape(wSize,wSize)
    # b = time()
    # print(b-a,"Cb pred")


    # modelCr=Lasso().fit(np.zeros((wSize*wSize,3)),np.zeros((wSize*wSize,1)))
    # modelCr.coef_ = recipe_c.reshape(3,)
    # modelCr.intercept_ = recipe_c_offset.reshape(1,)
    # HighFreqCr = modelCr.predict(X).reshape(wSize,wSize)

    # in_lum=highFreqData_in[:,:,0]
    # maxi=np.max(in_lum)
    # mini=np.min(in_lum)
    # for i in range(1,k):
    #     yi = mini + i*(maxi-mini)/k
    #     si = (in_lum>=yi)
    #     si = si*(in_lum-yi)
    #     si.shape +=(1,)
    #     X = np.concatenate((X,si.reshape(wSize*wSize,1)),axis=1)
    
    # modelY=Lasso()
    # a = time()
    # modelY = modelY.fit(np.zeros(X.shape),np.zeros((wSize*wSize,1)))
    # b = time()
    # print(b-a,"Y fit")
    # modelY.coef_[0:3] = recipe_d.reshape(-1,)
    # modelY.coef_[3:] = recipe_f.reshape(-1,)
    # modelY.intercept_ = recipe_d_offset.reshape(1,)
    
    # a = time()
    # HighFreqY = modelY.predict(X).reshape(wSize,wSize)
    # b = time()
    # print(b-a,"Y pred")
    
    HighFreqCb = recipe_b[0] * highFreqData_in[:,:,0] + recipe_b[1] * highFreqData_in[:,:,1] + recipe_b[2] * highFreqData_in[:,:,2] + recipe_b_offset
    HighFreqCr = recipe_c[0] * highFreqData_in[:,:,0] + recipe_c[1] * highFreqData_in[:,:,1] + recipe_c[2] * highFreqData_in[:,:,2] + recipe_c_offset
    HighFreqY  = recipe_d[0] * highFreqData_in[:,:,0] + recipe_d[1] * highFreqData_in[:,:,1] + recipe_d[2] * highFreqData_in[:,:,2] + recipe_d_offset + sum_si
    
    # print(HighFreqY[HighFreqY < 0])
    
    # ---------6
    upSampResidual = residualOutBlock.reshape(1,1,3)
    for i in range(int(np.log2(wSize))):
        upSampResidual = cv2.pyrUp(upSampResidual)
    return upSampResidual[:,:,1:3], upSampInpBlock, HighFreqY,HighFreqCb,HighFreqCr
    # return np.zeros((inputBlock.shape[0],inputBlock.shape[1],2)), np.zeros(inputBlock.shape[0]), np.zeros(inputBlock.shape[0]), np.zeros(inputBlock.shape[0]), np.zeros(inputBlock.shape[0])