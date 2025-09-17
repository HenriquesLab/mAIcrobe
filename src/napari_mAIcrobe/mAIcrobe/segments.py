"""Module responsible for the identification of single regions inside the
mask, which should correspond to the cell regions.
The regions are then labeled using the watershed algorithm. Requires an
ImageManager object containg the loaded images (base + fluor) and the mask.
Contains a single class, Segments, which stores the data from the processing
of the mask: features and labels, which will later be used to define the
different cell regions
"""

import numpy as np
from scipy import ndimage
from skimage import segmentation
from skimage.feature import peak_local_max


class SegmentsManager:
    """Segmentation seed detection and labeling via watershed.

    Computes distance transform peaks as seeds, overlays features, and
    runs watershed to generate integer labels.

    This class is used to generate a label image after a binary mask has
    been computed, normally through thresholding and morphological
    operations.

    #TODO remove deprecated attributes and methods.

    Attributes
    ----------
    features : numpy.ndarray or None
        Seed markers image.
    labels : numpy.ndarray or None
        Integer label image.
    base_w_features : numpy.ndarray or None
        Binary overlay of features. Deprecated.
    fluor_w_features : numpy.ndarray or None
        Reserved for fluorescence overlay. Deprecated.
    """

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
        """Compute peaks (maximum values) from the euclidean distance
        transform.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask (non-zero inside cell regions).
        params : dict
            Dictionary with keys:
            - "peak_min_distance_from_edge" : int
            - "peak_min_distance" : int
            - "peak_min_height" : float
            - "max_peaks" : int

        Returns
        -------
        list[tuple[int, int]]
            List of (x, y) marker coordinates to be used by the
            compute_features method.
        """

        distance = ndimage.distance_transform_edt(mask)

        mindist = params["peak_min_distance"]
        minmargin = params["peak_min_distance_from_edge"]

        centers = peak_local_max(
            distance,
            min_distance=mindist,
            threshold_abs=params["peak_min_height"],
            exclude_border=True,
            num_peaks=params["max_peaks"],
        )

        placedmask = np.ones(distance.shape)
        lx, ly = distance.shape
        result = []
        heights = []
        circles = []

        for c in centers:
            x, y = c

            if (
                x >= minmargin
                and y >= minmargin
                and x <= lx - minmargin
                and y <= ly - minmargin
                and placedmask[x, y]
            ):
                placedmask[
                    x - mindist : x + mindist + 1,
                    y - mindist : y + mindist + 1,
                ] = 0
                s = distance[x, y]
                circles.append((x, y))
                heights.append(s)

        ixs = np.argsort(heights)
        for ix in ixs:
            result.append(circles[ix])

        return result

    def compute_features(self, params, mask):
        """Generate marker features image from peak coordinates.

        Parameters
        ----------
        params : dict
            Parameters for distance peaks (see `compute_distance_peaks`).
        mask : numpy.ndarray
            Binary mask image.
        """

        features = np.zeros(mask.shape)

        if params["peak_min_distance_from_edge"] < 1:
            params["peak_min_distance_from_edge"] = 1

        circles = self.compute_distance_peaks(mask, params)

        for ix, c in enumerate(circles):
            x, y = c
            for f in range(3):
                features[x - 1 + f, y] = ix + 1
                features[x, y - 1 + f] = ix + 1

        self.features = features

    def overlay_features(self, mask):
        """Create a binary overlay image of the features.
         DEPRECATED.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask used only for shape reference.
        """

        clipped_base = np.zeros(mask.shape)

        places = self.features > 0.5
        clipped_base[places] = 1
        self.base_w_features = clipped_base.astype(int)

    def compute_labels(self, mask):
        """Run watershed to obtain integer labels. Uses the features
        attribute computed in compute_features as markers for the
        watershed.

        Parameters
        ----------
        mask : numpy.ndarray
            Binary mask used to constrain watershed.

        Returns
        -------
        None
            Results stored in `self.labels`.
        """

        markers = self.features

        distance = -ndimage.distance_transform_edt(mask)
        mindist = np.min(distance)
        markpoints = markers > 0
        distance[markpoints] = mindist
        labels = segmentation.watershed(distance, markers, mask=mask).astype(
            int
        )

        self.labels = labels

    def compute_segments(self, params, mask):
        """Full pipeline: features, overlay, and labels.

        Parameters
        ----------
        params : dict
            Parameters for peak detection.
        mask : numpy.ndarray
            Binary mask image.
        """

        self.compute_features(params, mask)
        self.overlay_features(mask)
        self.compute_labels(mask)
