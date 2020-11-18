from utils import laplacianStack
import numpy as np
from sklearn.linear_model import LinearRegression, Lasso
from params import wSize, k

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
    
    Y1 = highFreqData_out_chrom1.reshape(64*64,1)
    X = highFreqData_in.reshape(64*64,3)
    Y2 = highFreqData_out_chrom2.reshape(64*64,1)
    model1=Lasso()
    reg_chrom1 = model1.fit(-X, Y1)
    model2=Lasso()
    reg_chrom2 = model2.fit(-X, Y2)
    

    # recipe for luminance layer
    highFreqData_out_lum = highFreqData_out[:,:,0]
    
    X=highFreqData_in.reshape(64*64,3)
    for layer in lstack_in:
        X = np.concatenate((X,layer[:,:,0].reshape(64*64,1)),axis=1)
    
    in_lum=highFreqData_in[:,:,0]
    maxi=np.max(in_lum)
    mini=np.min(in_lum)
    for i in range(1,k):
        yi = mini + i*(maxi-mini)/k
        si = (in_lum>=yi)
        si = si*(in_lum-yi)
        si.shape +=(1,)
        X = np.concatenate((X,si.reshape(64*64,1)),axis=1)

    model=Lasso( fit_intercept = True, precompute = True,  max_iter = 1e9)
    reg_lum = model.fit(-X,highFreqData_out_lum.reshape(64*64,1))
    
    recipe_a = residual_recipe
    recipe_b = np.append(reg_chrom1.coef_,-reg_chrom1.intercept_)
    recipe_c = np.append(reg_chrom2.coef_,-reg_chrom2.intercept_)
    recipe_d = np.array([reg_lum.coef_[0],reg_lum.coef_[1],reg_lum.coef_[2],-reg_lum.intercept_[0]])
    recipe_e = np.array([reg_lum.coef_[i] for i in range(3,3+len(lstack_in))])
    recipe_f = np.array([reg_lum.coef_[i] for i in range(3+len(lstack_in),len(reg_lum.coef_))])

    return recipe_a, recipe_b, recipe_c, recipe_d, recipe_e, recipe_f