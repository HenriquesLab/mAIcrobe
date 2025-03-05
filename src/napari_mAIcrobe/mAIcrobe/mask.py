"""
Module that contains the logic for mask computation
"""

import numpy as np
from scipy import ndimage, signal
from skimage.filters import threshold_isodata, threshold_local
from skimage.morphology import binary_closing, binary_dilation
from skimage.transform import EuclideanTransform, warp

def mask_computation(base_image:np.ndarray, algorithm:str="Isodata",
                     blocksize:int=151,offset:float=0.02,closing:int=1,
                     dilation:int=0,fillholes:bool=False):
    
    # Binarization
    if algorithm == "Isodata":
        mask = base_image > threshold_isodata(base_image)
        mask = mask.astype(int)
        mask = 1 - mask
    elif algorithm == "Local Average":
        if blocksize%2==0:
            blocksize += 1
        mask = base_image > threshold_local(base_image, block_size=blocksize, method="gaussian", offset=offset)
        mask = mask.astype(int)
        mask = 1 - mask
    
    # remove spots (both white and dark)
    if closing > 0:
        # removes small white spots and then small dark spots
        closing_matrix = np.ones((int(closing), int(closing)))
        mask = binary_closing(mask, closing_matrix)
        mask = 1 - binary_closing(1 - mask, closing_matrix)

    # dilation
    for f in range(dilation):
        mask = binary_dilation(mask, np.ones((3, 3)))

    # binary fill holes
    if fillholes:
        mask = ndimage.binary_fill_holes(mask)

    return mask 

def mask_alignment(mask:np.ndarray, fluor_image:np.ndarray):
        
    corr = signal.fftconvolve(mask,fluor_image[::-1,::-1])
    deviation = np.unravel_index(np.argmax(corr),corr.shape)
    cm = ndimage.center_of_mass(np.ones(corr.shape))
        
    dy,dx = np.subtract(deviation,cm)
    matrix = EuclideanTransform(rotation=0, translation=(dx,dy))

    aligned_fluor = warp(fluor_image, matrix.inverse, preserve_range=True) # TODO check if fluor intensity values stay the same

    return aligned_fluor