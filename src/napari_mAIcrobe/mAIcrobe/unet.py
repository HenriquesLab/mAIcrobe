from tensorflow.keras.models import load_model
import numpy as np

from math import ceil

from skimage.morphology import binary_erosion
from skimage.morphology import binary_closing, binary_dilation, binary_opening
from skimage.segmentation import watershed
from scipy.ndimage import label as lbl
from scipy import ndimage


# TODO move this 3 function to the maicrobe folder
############################################################################
## THIS FUNCTION ARE COPIED FROM THE ZEROCOSTDL4MIC UNET JUPYTER NOTEBOOK ##
############################################################################
def normalizePercentile(x, pmin=1, pmax=99.8, axis=None, clip=False, eps=1e-20, dtype=np.float32):
    """This function is adapted from Martin Weigert"""
    """Percentile-based image normalization."""

    mi = np.percentile(x,pmin,axis=axis,keepdims=True)
    ma = np.percentile(x,pmax,axis=axis,keepdims=True)
    return normalize_mi_ma(x, mi, ma, clip=clip, eps=eps, dtype=dtype)

############################################################################
## THIS FUNCTION ARE COPIED FROM THE ZEROCOSTDL4MIC UNET JUPYTER NOTEBOOK ##
############################################################################
def normalize_mi_ma(x, mi, ma, clip=False, eps=1e-20, dtype=np.float32):#dtype=np.float32
    """This function is adapted from Martin Weigert"""
    if dtype is not None:
        x   = x.astype(dtype,copy=False)
        mi  = dtype(mi) if np.isscalar(mi) else mi.astype(dtype,copy=False)
        ma  = dtype(ma) if np.isscalar(ma) else ma.astype(dtype,copy=False)
        eps = dtype(eps)

    try:
        import numexpr
        x = numexpr.evaluate("(x - mi) / ( ma - mi + eps )")
    except ImportError:
        x =                   (x - mi) / ( ma - mi + eps )

    if clip:
        x = np.clip(x,0,1)

    return x

############################################################################
## THIS FUNCTION ARE COPIED FROM THE ZEROCOSTDL4MIC UNET JUPYTER NOTEBOOK ##
############################################################################
def predict_as_tiles(img, model):

  # Read the data in and normalize
  Image_raw = normalizePercentile(img)

  # Get the patch size from the input layer of the model
  patch_size = model.layers[0].output_shape[0][1:3]

  # Pad the image with zeros if any of its dimensions is smaller than the patch size
  if Image_raw.shape[0] < patch_size[0] or Image_raw.shape[1] < patch_size[1]:
    Image = np.zeros((max(Image_raw.shape[0], patch_size[0]), max(Image_raw.shape[1], patch_size[1])))
    Image[0:Image_raw.shape[0], 0: Image_raw.shape[1]] = Image_raw
  else:
    Image = Image_raw

  # Calculate the number of patches in each dimension
  n_patch_in_width = ceil(Image.shape[0]/patch_size[0])
  n_patch_in_height = ceil(Image.shape[1]/patch_size[1])

  prediction = np.zeros(Image.shape, dtype = 'uint8')

  for x in range(n_patch_in_width):
    for y in range(n_patch_in_height):
      xi = patch_size[0]*x
      yi = patch_size[1]*y

      # If the patch exceeds the edge of the image shift it back
      if xi+patch_size[0] >= Image.shape[0]:
        xi = Image.shape[0]-patch_size[0]

      if yi+patch_size[1] >= Image.shape[1]:
        yi = Image.shape[1]-patch_size[1]

      # Extract and reshape the patch
      patch = Image[xi:xi+patch_size[0], yi:yi+patch_size[1]]
      patch = np.reshape(patch,patch.shape+(1,))
      patch = np.reshape(patch,(1,)+patch.shape)

      # Get the prediction from the patch and paste it in the prediction in the right place
      predicted_patch = model.predict(patch, batch_size = 1)

      prediction[xi:xi+patch_size[0], yi:yi+patch_size[1]] = (np.argmax(np.squeeze(predicted_patch), axis = -1)).astype(np.uint8)


  return prediction[0:Image_raw.shape[0], 0: Image_raw.shape[1]]


def computelabel_unet(path2model, base_image, closing, dilation, fillholes):
   
    model = load_model(path2model)
    prediction = predict_as_tiles(base_image, model) 

    mask = prediction>0

    mask = binary_opening(mask, np.ones((3, 3)))

    #edges = prediction==1
    insides = prediction==2
    for _ in range(0): # TODO 
        insides = binary_erosion(insides)
    insides = insides.astype(np.uint16)
    insides, _ = lbl(insides)

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

    labels = watershed(~mask,markers=insides,mask=mask)

    return mask, labels