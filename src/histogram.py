
import cv2 
import numpy as np
from pyramid import laplacianPyramid,buildLaplacianPyramid

def get_histogram(array, resolution = 256):
    """ Computes a 1D histogram of the given array """

    array = np.ravel(array)
    # if mini is None:
    mini = np.amin(array)
    # if maxi is None:
    maxi = np.amax(array)
    bins = np.linspace(mini,maxi,endpoint = True, num = resolution)
    h = np.histogram(array, bins = bins)[0]
    return (h,bins)



def get_histograms(I, nlevels = 3, resolution = 256):
    """ Compute rescaling histograms """

    if len(I.shape) > 2:
        nchan = I.shape[2]
    else:
        nchan = 1

    nHistPerChannel = nlevels + 1
    LI            = buildLaplacianPyramid(I,nLevels = nlevels+1)
    histograms    = np.zeros((nchan*nHistPerChannel*((resolution-1)) + 3,), dtype=np.int32)

    histograms[0] = resolution
    histograms[1] = nlevels
    histograms[2] = nchan
    idx           = 3

    rng = np.zeros(nchan*nHistPerChannel*2, dtype=np.float32)

    for il in range(nlevels):
        l    = LI[il]
        for c in range(nchan):
            h,bins = get_histogram(l[:,:,c], resolution)
            n = len(h)
            rng[2*c+2*nchan*il]   = bins[0]
            rng[2*c+2*nchan*il+1] = bins[-1]
            histograms[idx:idx+n] = h
            idx += n

    # histogram of YCbCr values
    il = nlevels
    for c in range(nchan):
        h,bins = get_histogram(I[:,:,c], resolution)
        n = len(h)
        rng[2*c+2*nchan*il]   = bins[0]
        rng[2*c+2*nchan*il+1] = bins[-1]
        histograms[idx:idx+n] = h
        idx += n

    return histograms,rng


def extract_histogram(histograms, rng, lvl, chan):
    """ Extract an histogram from the packed data """

    resolution = histograms[0]
    # nlevels    = histograms[2]
    nchan      = histograms[2]
    n          = resolution -1

    # nHistPerChannel = nlevels + 1
    idx = 3 + lvl*(nchan*((resolution-1))) + chan*((resolution-1))

    mini = rng[2*chan+2*lvl*nchan]
    maxi = rng[2*chan+2*lvl*nchan+1]
    h    = histograms[idx:idx+n]
    bins = np.linspace(mini,maxi,endpoint = True, num = resolution)

    return (h,bins)