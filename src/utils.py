import cv2
import numpy as np
from params import wSize, sigma

def float2uint8(R):
    return np.uint8(np.clip(np.round(R),0,255))

def convertRGB_YCbCr(im):
    im = im.astype(np.float32)
    YCbCr = np.zeros(im.shape)
    YCbCr[:,:,0] = 0.299*im[:,:,0] + 0.587*im[:,:,1] + 0.114*im[:,:,2]
    YCbCr[:,:,1] = 128 - 0.168736*im[:,:,0] - 0.331264*im[:,:,1] + 0.5*im[:,:,2]
    YCbCr[:,:,2] = 128 + 0.5*im[:,:,0] - 0.418688*im[:,:,1] - 0.081312*im[:,:,2]

    YCbCr = float2uint8(YCbCr)
    return YCbCr

def convertYCbCr_RGB(im):
    im = im.astype(np.float32)
    RGB = np.zeros(im.shape)
    RGB[:,:,0] = im[:,:,0] + 1.402*(im[:,:,2]-128)
    RGB[:,:,1] = im[:,:,0] - 0.34414*(im[:,:,1]-128) - 0.71414*(im[:,:,2]-128)
    RGB[:,:,2] = im[:,:,0] + 1.772*(im[:,:,1]-128)

    RGB = float2uint8(RGB)

    return RGB

# def laplacianStack(img):

#     stackSize = int(np.log2(img.shape[0]))
#     out = np.copy(img)
#     pyramid = [out]
#     # pyramid creation through downsampling
#     for i in range(stackSize):
#         out = cv2.pyrDown(out)
#         pyramid.append(out)

#     # pyramid
#     stack = []
#     for i in range(1,len(pyramid)):
#         downImg = pyramid[i]
#         upsampledImg = cv2.pyrUp(downImg)
#         laplacianImg = pyramid[i-1] - upsampledImg
#         for j in range(i-1):
#             laplacianImg = cv2.pyrUp(laplacianImg)
#         stack.append(laplacianImg)
#     stack = np.array(stack)
#     return stack, np.array(pyramid[-1]).reshape(3).astype('int')

def laplacianStack(I, nLevels= -1, minSize = 1, useStack = True):
    if nLevels == -1:
        nLevels = int(np.log2(I.shape[0]))+1

    pyramid = nLevels*[None]
    pyramid[0] = I
    if len(pyramid[0].shape) < 3:
        pyramid[0].shape += (1,)
    # All levels have the same resolution
    if useStack:
        # Gaussian pyramid
        for i in range(nLevels-1):
            srcSz = pyramid[i].shape[0:2]
            newSz = tuple([a/2 for a in pyramid[i].shape[0:2]])
            newSz = (newSz[1],newSz[0])
            pyramid[i+1] = cv2.pyrDown(pyramid[i])
            if len(pyramid[i+1].shape) < 3:
                pyramid[i+1].shape += (1,)

        # Make a stack
        for lvl in range(0,nLevels-1):
            for i in range(nLevels-1,lvl,-1):
                newSz = pyramid[i-1].shape[0:2]
                up = cv2.pyrUp(pyramid[i],dstsize=(newSz[1],newSz[0]))
                if len(up.shape) < 3:
                    up.shape += (1,)
                pyramid[i] = np.array(up)

        lapl = nLevels*[None]
        lapl[nLevels-1] = np.copy(pyramid[nLevels-1])
        for i in range(0,nLevels-1):
            lapl[i] = pyramid[i].astype(np.float32) - pyramid[i+1].astype(np.float32)
        pyramid = lapl
        pyramid = np.array(pyramid)
        residual = pyramid[-1]
        for i in range(int(np.log2(wSize))):
            residual = cv2.pyrDown(residual)
        
        # print(pyramid[:-1].shape,residual.shape)
        
        return pyramid[:-1], residual

    else:
        for i in range(nLevels-1):
            srcSz = pyramid[i].shape[0:2]
            newSz = tuple([a/2 for a in pyramid[i].shape[0:2]])
            newSz = (newSz[1],newSz[0])
            pyramid[i+1] = cv2.pyrDown(pyramid[i])
            if len(pyramid[i+1].shape) < 3:
                pyramid[i+1].shape += (1,)

        for i in range(nLevels-1):
            newSz = pyramid[i].shape[0:2]
            up = cv2.pyrUp(pyramid[i+1],dstsize=(newSz[1],newSz[0])).astype(np.float32)
            if len(up.shape) < 3:
                up.shape += (1,)
            pyramid[i] = pyramid[i].astype(np.float32) - up
        
        return pyramid

def GaussianKernel(size=wSize, sigma=sigma):
    """\
    creates gaussian kernel with side length l and a sigma of sig
    """
    x = np.linspace(- (size // 2), size // 2,size)
    x /= np.sqrt(2)*sigma
    x2 = x**2
    kernel = np.exp(- x2[:, None] - x2[None, :])
    return kernel / kernel.sum()

def nearest2power(num):
    return 2**int(np.log2(num)+1)