import os
import sys
import cv2
import numpy as np
from utils import convertRGB_YCbCr
from utils import laplacianStack
from recipe import recipeMaker
from reconstruction import reconstruct
import matplotlib.pyplot as plt
from params import wSize, k

from tqdm import tqdm
import signal 
import sys
def signal_handler(signal, frame):
    sys.exit(0)
signal.signal(signal.SIGINT, signal_handler)

def main(args):
    # image load
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
    recipe_wDim= (w-wSize)//stepSize + (1 if w%stepSize==0 else 2)
    recipe_lDim= (l-wSize)//stepSize + (1 if w%stepSize==0 else 2)

    # recipe intialization
    recipe_a = np.zeros((recipe_wDim,recipe_lDim,3))
    recipe_b = np.zeros((recipe_wDim,recipe_lDim,4))
    recipe_c = np.zeros((recipe_wDim,recipe_lDim,4))
    recipe_d = np.zeros((recipe_wDim,recipe_lDim,4))
    recipe_e = np.zeros((recipe_wDim,recipe_lDim,int(np.log2(wSize))))
    recipe_f = np.zeros((recipe_wDim,recipe_lDim,k-1))

    idx=0
    idy=0
    print("Recipe construction start")
    for i in tqdm(range(0, w - wSize + 1, stepSize)):
        for j in range(0, l - wSize + 1, stepSize):
            recipe_a[idx][idy], recipe_b[idx][idy], recipe_c[idx][idy], recipe_d[idx][idy], recipe_e[idx][idy], recipe_f[idx][idy] =recipeMaker(inputImageYCbCr[i:i+wSize,j:j+wSize,:],
                                                                                                                                    outputImageYCbCr[i:i+wSize,j:j+wSize,:])
            idy+=1

        if j + wSize < l:
            recipe_a[idx][idy], recipe_b[idx][idy], recipe_c[idx][idy], recipe_d[idx][idy], recipe_e[idx][idy], recipe_f[idx][idy] =recipeMaker(inputImageYCbCr[i:i+wSize,-wSize:,:],
                                                                                                                                    outputImageYCbCr[i:i+wSize,-wSize:,:])
            idy+=1
        idy=0
        idx+=1

    if i + wSize < w:
        for j in range(0, l - wSize + 1, stepSize):
            recipe_a[idx][idy], recipe_b[idx][idy], recipe_c[idx][idy], recipe_d[idx][idy], recipe_e[idx][idy], recipe_f[idx][idy] =recipeMaker(inputImageYCbCr[-wSize:,j:j+wSize,:],
                                                                                                                                    outputImageYCbCr[-wSize:,j:j+wSize,:])
            idy+=1

        if j + wSize < l:
            recipe_a[idx][idy], recipe_b[idx][idy], recipe_c[idx][idy], recipe_d[idx][idy], recipe_e[idx][idy], recipe_f[idx][idy] =recipeMaker(inputImageYCbCr[-wSize:,-wSize:,:],
                                                                                                                                    outputImageYCbCr[-wSize:,-wSize:,:])
            idy+=1
    
    print("Recipe Done")
    #recipe normaliztion
    recipes=[recipe_a,recipe_b,recipe_c,recipe_d,recipe_e,recipe_f]
    recipes_norm=[]
    for recipe in recipes:
        recipe_norm=[]
        for dim in range(recipe.shape[2]):
            layer = recipe[:,:,dim]
            norm = layer/((np.max(layer)-np.min(layer)) if (np.max(layer)-np.min(layer))!=0 else 1)
            recipe_norm.append(norm)
        recipes_norm.append(recipe_norm)
    recipes_norm=np.array(recipes_norm)
    recipes_norm[0] = np.array(recipes_norm[0])
    for i in range(1,len(recipes_norm)):
        recipes_norm[i] = np.round(np.array(recipes_norm[i]) * 255)

    print("Recipe Normalized")    
    np.savez_compressed('recipe',recipe=recipes_norm)
    #display recipe
    # recipes_norm = np.load('recipe.npz',allow_pickle=True)['recipe']
    fig, axes = plt.subplots(6)
    for i in range(recipes_norm.shape[0]):
        img=np.array(recipes_norm[i][0])
        for j in range(1,len(recipes_norm[i])):
            img=np.hstack([img,recipes_norm[i][j]])
        axes[i].imshow(np.array(img),cmap='gray')
    fig.savefig('hello.jpg')
    fig.show()
    plt.close(fig)

    output_image = reconstruct(inputImage,recipes_norm)
    print("Reconstruction Done")
    cv2.imwrite('out.jpg',cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR))

if __name__ == '__main__':
    inputImageLocation = '../images/inp.jpg'
    outputImageLocation = '../images/out.jpg'
    args = [inputImageLocation, outputImageLocation]
    main(args)