"""
Module responsible for GUI to do mask computation and channel alignment. 
"""

import typing
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import napari

from magicgui import magic_factory

from .mAIcrobe.mask import mask_computation, mask_alignment

@magic_factory(algorithm={"choices":["Isodata","Local Average"]})
def compute_mask(Viewer:"napari.Viewer",Base:"napari.layers.Image",Fluor_1:"napari.layers.Image",
                 Fluor_2:"napari.layers.Image",algorithm="Isodata",blocksize:int=151,offset:float=0.02,closing:int=1,
                 dilation:int=0,fillholes:bool=False,autoalign:bool=False)->typing.List["napari.types.LayerDataTuple"]:
    
    mask = mask_computation(base_image=Base.data,algorithm=algorithm,blocksize=blocksize,
                            offset=offset,closing=closing,dilation=dilation,fillholes=fillholes)

    if autoalign:
        aligned_fluor_1 = mask_alignment(mask, Fluor_1.data)
        aligned_fluor_2 = mask_alignment(mask, Fluor_2.data)

        if (aligned_fluor_2 == aligned_fluor_1).all():
            return [(mask, {'name': 'Mask'}, 'Labels'),
                    (aligned_fluor_1,{'name':'aligned_'+Fluor_1.name},'Image'),]
        else:
            return [(mask, {'name': 'Mask'}, 'Labels'),
                    (aligned_fluor_1,{'name':'aligned_'+Fluor_1.name},'Image'),
                    (aligned_fluor_2,{'name':'aligned_'+Fluor_2.name},'Image')]
    else:
        return [(mask, {'name': 'Mask'}, 'Labels'),]