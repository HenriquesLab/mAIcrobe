"""
Module responsible for GUI to do label computation and channel alignment. 
"""

from typing import TYPE_CHECKING
from typing import cast

if TYPE_CHECKING:
    import napari

from .mAIcrobe.mask import mask_computation, mask_alignment
from .mAIcrobe.segments import SegmentsManager

from magicgui.widgets import Container, create_widget, SpinBox, ComboBox, FileEdit, Label, PushButton, CheckBox

from qtpy.QtCore import Qt
from qtpy import QtWidgets

from tensorflow.keras.models import load_model
import numpy as np

from math import ceil

from skimage.morphology import binary_erosion
from skimage.morphology import binary_closing, binary_dilation
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



class compute_label(Container):

    def __init__(self, viewer:"napari.viewer.Viewer"):

        self._viewer = viewer

        # IMAGE INPUTS
        self._baseimg_combo = cast(ComboBox, create_widget(annotation="napari.layers.Image",label='Base Image'))
        self._fluor1_combo = cast(ComboBox, create_widget(annotation="napari.layers.Image",label='Fluor 1'))
        self._fluor2_combo = cast(ComboBox, create_widget(annotation="napari.layers.Image",label='Fluor 2'))

        self._closinginput = SpinBox(min=0, max=5, step=1, value=0, label='Binary Closing')
        self._dilationinput = SpinBox(min=0, max=5, step=1, value=0, label='Binary Dilation')
        self._fillholesinput = CheckBox(label='Fill Holes')
        self._autoaligninput = CheckBox(label='Auto Align')

        # MASK ALGORITHM
        self._algorithm_combo = cast(ComboBox, create_widget(options={"choices":["Isodata","Local Average","Unet"]},label='Mask algorithm'))
        self._algorithm_combo.changed.connect(self._on_algorithm_changed)

        self._titlemasklabel = Label(value='Parameters for Mask computation')
        self._titlemasklabel.native.setAlignment(Qt.AlignCenter)
        self._titlemasklabel.native.setStyleSheet("background-color: rgb(037, 041, 049); border: 1px solid rgb(059, 068, 077);")
        self._titlemasklabel.native.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)        
        
        self._placeholder = Label(value='...')
        self._placeholder.native.setAlignment(Qt.AlignCenter)
        self._placeholder.native.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)        

        self._blocksizeinput = SpinBox(min=0, max=1000, step=1, value=151, label='Blocksize', visible=False)
        self._offsetinput =  SpinBox(min=0, max=1, step=0.001, value=0.02, label='Offset',visible=False)
        self._path2unet = FileEdit(mode='r',label='Path to UnetModel',visible=False)

        # WATERSHED ALGORITHM
        self._titlewatershedlabel = Label(value='Parameters for Watershed Algorithm')
        self._titlewatershedlabel.native.setStyleSheet("background-color: rgb(037, 041, 049); border: 1px solid rgb(059, 068, 077);")
        self._titlewatershedlabel.native.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)

        self._titlewatershedlabel.native.setAlignment(Qt.AlignCenter)
        self._peak_min_distance_from_edge = SpinBox(min=0, max=50, step=1, value=10, label='Peak Min Distance From Edge')
        self._peak_min_distance = SpinBox(min=0, max=50, step=1, value=5, label='Peak Min Distance')
        self._peak_min_height = SpinBox(min=0, max=50, step=1, value=5, label='Peak Min Height')
        self._max_peaks = SpinBox(min=0, max=100000, step=100, value=100000, label='Max Peaks')

        # RUN
        self._run_button = PushButton(label='Run')
        self._run_button.clicked.connect(self.compute)


        super().__init__(widgets=[self._baseimg_combo, self._fluor1_combo, self._fluor2_combo, self._closinginput, self._dilationinput, self._fillholesinput, self._autoaligninput, self._algorithm_combo, self._titlemasklabel, self._placeholder, self._blocksizeinput, self._offsetinput, self._path2unet, self._titlewatershedlabel, self._peak_min_distance_from_edge,self._peak_min_distance, self._peak_min_height, self._max_peaks, self._run_button], labels=True)

    def _on_algorithm_changed(self, new_algorithm: str):

        if new_algorithm=='Isodata':
            self[9].visible = True
            self[10].visible = False
            self[11].visible = False
            self[12].visible = False
        elif new_algorithm=='Local Average':
            self[9].visible = False
            self[10].visible = True
            self[11].visible = True
            self[12].visible = False
        elif new_algorithm=='Unet':
            self[9].visible = False
            self[10].visible = False
            self[11].visible = False
            self[12].visible = True

        return
    

    def compute(self):
        _algorithm = self._algorithm_combo.value

        _baseimg = self._baseimg_combo.value
        _fluor1 = self._fluor1_combo.value
        _fluor2 = self._fluor2_combo.value
        
        _binary_closing = self._closinginput.value
        _binary_dilation = self._dilationinput.value
        _binary_fillholes = self._fillholesinput.value
        _autoalign = self._autoaligninput.value
        
        _LAblocksize = self._blocksizeinput.value
        _LAoffset = self._offsetinput.value
        
        _pars = {'peak_min_distance_from_edge':self._peak_min_distance_from_edge.value,'peak_min_distance':self._peak_min_distance.value,'peak_min_height':self._peak_min_height.value,'max_peaks':self._max_peaks.value}

        if _algorithm == "Unet":
            model = load_model(self._path2unet.value)
            prediction = predict_as_tiles(_baseimg.data, model) 

            mask = prediction>0
            #edges = prediction==1
            insides = prediction==2
            for _ in range(0): # TODO 
                insides = binary_erosion(insides)
            insides = insides.astype(np.uint16)
            insides, _ = lbl(insides)
 
            if _binary_closing > 0:
                # removes small white spots and then small dark spots
                closing_matrix = np.ones((int(_binary_closing), int(_binary_closing)))
                mask = binary_closing(mask, closing_matrix)
                mask = 1 - binary_closing(1 - mask, closing_matrix)

            # dilation
            for f in range(_binary_dilation):
                mask = binary_dilation(mask, np.ones((3, 3)))

            # binary fill holes
            if _binary_fillholes:
                mask = ndimage.binary_fill_holes(mask)

            labels = watershed(~mask,markers=insides,mask=mask)

        else:
            mask = mask_computation(base_image=_baseimg.data,algorithm=_algorithm,blocksize=_LAblocksize,
                            offset=_LAoffset,closing=_binary_closing,dilation=_binary_dilation,fillholes=_binary_fillholes)

            seg_man = SegmentsManager()
            seg_man.compute_segments(_pars, mask)

            labels = seg_man.labels

        # add mask to viewer
        self._viewer.add_labels(mask,name='Mask')
        # add labelimg to viewer
        self._viewer.add_labels(labels,name='Labels')

        if _autoalign:
            aligned_fluor_1 = mask_alignment(mask, _fluor1.data)
            aligned_fluor_2 = mask_alignment(mask, _fluor2.data)

            self._viewer.layers[_fluor1.name].data = aligned_fluor_1
            self._viewer.layers[_fluor2.name].data = aligned_fluor_2

        