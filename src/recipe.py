from utils import laplacianStack
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso

def recipeMaker(imgBlock_in, imgBlock_out):
    lstack_in, residual_in = laplacianStack(imgBlock_in)
    lstack_out, residual_out = laplacianStack(imgBlock_out)

    # recipe for the residual layer
    residual_recipe = (residual_out+1)/(residual_in+1)

    #calculate high frequency data

    highFreqData_in = np.sum(lstack_in,axis=0)
    highFreqData_out = np.sum(lstack_out,axis=0)

    # recipe for chrominance layers
    highFreqData_out_chrom1=highFreqData_out[:,:,1]
    highFreqData_out_chrom2=highFreqData_out[:,:,2]
    
    Y1 = highFreqData_out_chrom1.reshape(64*64)
    X = highFreqData_in.reshape(64*64,3)
    reg_chrom1 = LinearRegression().fit(X, Y1)

    Y2 = highFreqData_out_chrom2.reshape(64*64)
    reg_chrom2 = LinearRegression().fit(X, Y2)
    

    # recipe for luminance layer
    highFreqData_out_lum = highFreqData_out[:,:,0]
    
    X=highFreqData_in.reshape(64*64,3)
    for layer in lstack_in:
        X = np.concatenate((X,layer[:,:,0].reshape(64*64,1)),axis=1)
    
    #this is k for the piecewise function
    k=6
    in_lum=highFreqData_in[:,:,0]
    maxi=np.max(in_lum)
    mini=np.min(in_lum)
    for i in range(1,k):
        yi = mini + i*(maxi-mini)/k
        si = (in_lum>=yi)
        si = si*(in_lum-yi)
        si.shape +=(1,)
        X = np.concatenate((X,si.reshape(64*64,1)),axis=1)

    model=Lasso( fit_intercept = True, precompute = True,  max_iter = 1e4)
    reg_lum = model.fit(X,highFreqData_out_lum.reshape(64*64,1))
    
    # return 

