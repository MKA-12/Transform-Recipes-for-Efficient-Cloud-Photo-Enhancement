from utils import laplacianStack
import numpy as np
from sklearn.linear_model import LinearRegression

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
    reg1 = LinearRegression().fit(X, Y1)

    Y2 = highFreqData_out_chrom2.reshape(64*64)
    reg2 = LinearRegression().fit(X, Y2)
    
    print(reg1.coef_,reg1.intercept_,reg1.score(X,Y1))
    print(reg2.coef_,reg2.intercept_,reg2.score(X,Y2))

    # print(highFreqData_in_chrom.shape)