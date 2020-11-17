import cv2 
import numpy as np
from utils import convertRGB_YCbCr,convertYCbCr_RGB
from pyramid import laplacianPyramid,reconstructFromLaplacianPyramid,buildLaplacianPyramid
from histogram import get_histograms,get_histogram,extract_histogram
from scipy.interpolate import interp1d
from PIL import Image
def upscaleInputImage(shapeInputImage,hist_ref,rng_ref,jpegDownsampledImage):        
    hist_resample = 1
    ref_res       = 256*10
    deg_res       = 256*10

    jpegCompUpsampled = cv2.resize(jpegDownsampledImage,(shapeInputImage[1],shapeInputImage[0]))
    cv2.imwrite('../images/upsized.jpg',jpegCompUpsampled)
    # O = process(hist_ref,rng_ref,jpegCompUpsampled, transfer_color = True, nlevels = nlevels, deg_res = deg_res, ref_res = ref_res, hist_resample = hist_resample)

    #Processing the image and upsampling
    Id = convertRGB_YCbCr(jpegCompUpsampled)
    Id = Id.astype(np.float32)

    OutputImage = transfer(Id,hist_ref,rng_ref, deg_res, ref_res, hist_resample = hist_resample, transfer_color = True)

    OutputImage[OutputImage<0]   = 0
    OutputImage[OutputImage>255] = 255
    OutputImage = convertYCbCr_RGB(OutputImage)
    OutputImage = OutputImage.astype(np.uint8)

    return OutputImage


    cv2.imwrite('../images/upscaledInput.jpg',OutputImage)



def add_noise(I, sigma = 1.0):
    """ Add Gaussian random noise to input I """
    noisemap = np.random.randn(6000,6000,3)
    sz = I.shape
    I = I.astype(np.float32)
    if len(sz) > 2:
        I = I+sigma*noisemap[:sz[0], :sz[1], :sz[2]]
    else:
        I = I+sigma*noisemap[:sz[0], :sz[1],0]
    return I



def get_transfer_function(source, h2,bins2, res1 = 256, res2 = 256*10):
    """ Computes a 1D histogram remapping function """

    h1,bins1 = get_histogram(source, res1)
    # m1 = bins1[0]
    # M1 = bins1[-1]
    m2 = bins2[0]
    M2 = bins2[-1]

    h1 = h1.astype(np.float32)
    h2 = h2.astype(np.float32)

    cdf1 = np.cumsum(h1)*1.0/sum(h1)
    cdf2 = np.cumsum(h2)*1.0/sum(h2)


    # Resample in preparation of the inversion
    if res2-1 != len(cdf2):
        old_ticks = np.linspace(0,1,endpoint = True, num = len(cdf2))
        new_ticks = np.linspace(0,1,endpoint = True, num = res2)
        cdf2_i = interp1d(old_ticks, cdf2, kind="linear")
        cdf2_i = cdf2_i(new_ticks)
    else:
        cdf2_i = cdf2

    cdf2_i[-1] = 1
    cdf1[-1] = 1

    f = np.zeros((len(cdf1)-1,))
    for i in range(len(f)):
        found = np.where(cdf2_i >= cdf1[i])
        idx = found[0][0]
        f[i] = idx *1.0/ res2
    return f,m2,M2


def apply_transfer_function(source, f, m2, M2):
    """ Remap values """

    m = np.amin(source)
    M = np.amax(source)
    src = (source-m).astype(np.float64)
    if M != m:
        src /= (M-m)

    n = len(f)-1
    f = np.append(f, f[-1])
    idx  = (np.floor(n*src)).astype(int)
    x      = (n*src-idx)
    O      = ((1-x)*f[idx]+x*f[idx+1])
    O *= (M2-m2)
    O += m2

    return O

    
def transfer(Id,hist_ref,rng_ref, deg_res = 256, ref_res = 256*2, hist_resample = 10, transfer_color = True, output_dir = None):
    """ Transfer histogram hist_ref whose range of values is rng_ref 
        to image Id. The resolution of the reference histogram is
        ref_res and the resolution of the target's histogram is def_res.
        Specifiy a resampling factor before inverting the histogram.
    """

    nlevels = hist_ref[1]

    LId = buildLaplacianPyramid(Id, nLevels = nlevels + 1 )
    
    LO = []
    # print(len(LId))


    # Remap Laplacian
    for il in range(nlevels):
        ld   = LId[il]
        # print("LId = ",LId[il].shape)
        ld = add_noise(ld,sigma = 3.0)

        lo = np.zeros(ld.shape)
        for c in range(ld.shape[2]):
            ldc = ld[:,:,c]
            h,bins    = extract_histogram(hist_ref,rng_ref,il,c)

            f,m,M     = get_transfer_function(ldc,h,bins,deg_res,hist_resample*ref_res)
            lo[:,:,c] = apply_transfer_function(ldc, f,m, M)


        if len(lo.shape) == 2:
            lo.shape += (1,)
        LO.append(lo)
    LO.append(LId[-1])
    Id = reconstructFromLaplacianPyramid(LO)
    Id = np.squeeze(Id)

    # Remap colors
    if transfer_color:
        il = nlevels
        for c in range(Id.shape[2]):
            h,bins    = extract_histogram(hist_ref,rng_ref,il,c)
            f,m,M     = get_transfer_function(Id[:,:,c],h,bins,deg_res,hist_resample*ref_res)
            Id[:,:,c] = apply_transfer_function(Id[:,:,c], f,m, M)


    # Clamp values
    Id[Id<0]   = 0
    Id[Id>255] = 255
    return Id

# def process(hist_ref,rng_ref, Id, transfer_color = True, nlevels = 3, deg_res = 256, ref_res = 256*2, hist_resample = 10, output_dir = None):

#     Id = RGB_to_YCbCr(Id)
#     Id = Id.astype(np.float32)

#     O = transfer(Id,hist_ref,rng_ref, deg_res, ref_res, hist_resample, transfer_color, output_dir = output_dir)

#     O[O<0]   = 0
#     O[O>255] = 255
#     O = YCbCr_to_RGB(O)
#     O = O.astype(np.uint8)

#     return O