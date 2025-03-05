"""Module responsible for the identification of single regions inside the
mask, which should correspond to the cell regions.
The regions are then labeled using the watershed algorithm. Requires an
ImageManager object containg the loaded images (base + fluor) and the mask.
Contains a single class, Segments, which stores the data from the processing
of the mask: features and labels, which will later be used to define the different cell regions
"""

import numpy as np
from skimage.feature import peak_local_max
from skimage import segmentation
from scipy import ndimage

class SegmentsManager(object):
    """Main class of the module.
    The class is responsible for the computation of the features of the image.
    Requires an instance of the ImageManager class after loading both the
    base and fluor and image aswell as computing the mask
    The first step of the analysis consists on the usage of the distance
    peaks algorithm (based on the Euclidean Distances) and the second on the
    usage of the watershed algorithm to define each individual region of the
    mask"""

    def __init__(self):
        self.features = None
        self.labels = None
        self.base_w_features = None
        self.fluor_w_features = None

    def clear_all(self):
        """Resets the class instance to the initial state"""
        self.features = None
        self.labels = None
        self.base_w_features = None
        self.fluor_w_features = None

    @staticmethod
    def compute_distance_peaks(mask, params):
        """Method used when the selected algorithm for the feature computation
        is the Distance Peaks.
        Returns a list of the centers of the different identified regions,
        which should be used in the compute_features method"""

        distance = ndimage.morphology.distance_transform_edt(mask)

        mindist = params['peak_min_distance']
        minmargin = params['peak_min_distance_from_edge']

        centers = peak_local_max(distance, min_distance=mindist,
                                 threshold_abs=params['peak_min_height'],
                                 exclude_border=True,
                                 num_peaks=params['max_peaks'])

        placedmask = np.ones(distance.shape)
        lx, ly = distance.shape
        result = []
        heights = []
        circles = []

        for c in centers:
            x, y = c

            if x >= minmargin and y >= minmargin and x <= lx - minmargin \
                    and y <= ly - minmargin and placedmask[x, y]:
                placedmask[x - mindist:x + mindist +
                                       1, y - mindist:y + mindist + 1] = 0
                s = distance[x, y]
                circles.append((x, y))
                heights.append(s)

        ixs = np.argsort(heights)
        for ix in ixs:
            result.append(circles[ix])

        return result

    def compute_features(self, params, mask):
        """Method used to compute the features of an image using the mask.
        requires a mask and an instance of the imageprocessingparams
        if the selected algorithm used is Distance Peak, used the method
        compute_distance_peaks to compute the features"""
        
        features = np.zeros(mask.shape)

        if params['peak_min_distance_from_edge'] < 1:
            params['peak_min_distance_from_edge'] = 1

        circles = self.compute_distance_peaks(mask, params)

        for ix, c in enumerate(circles):
            x, y = c
            for f in range(3):
                features[x - 1 + f, y] = ix + 1
                features[x, y - 1 + f] = ix + 1

        self.features = features

    def overlay_features(self, mask):
        """Method used to produce an image with an overlay of the features on
        the base image requires a base image, the features and the clip
        values to overlay the images returns a image matrix which can be saved
        using the save_image method or directly using the imsave
        from skimage.io"""

        clipped_base = np.zeros(mask.shape)

        places = self.features > 0.5
        clipped_base[places] = 1
        self.base_w_features = clipped_base.astype(int)

    def compute_labels(self, mask):
        """Computes the labels for each region based on the previous computed
        features. Requires the mask, the features and an
        instance of the imageprocessingparams"""

        markers = self.features

        distance = - ndimage.morphology.distance_transform_edt(mask)
        mindist = np.min(distance)
        markpoints = markers > 0
        distance[markpoints] = mindist
        labels = segmentation.watershed(distance, markers, mask=mask).astype(int)

        self.labels = labels

    def compute_segments(self, params, mask):
        """Calls the different methods of the module in the right order.
        Can be used as the interface of this module in the main module of the
        software"""

        self.compute_features(params, mask)
        self.overlay_features(mask)
        self.compute_labels(mask)