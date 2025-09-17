import math

import numpy as np
from skimage.morphology import binary_erosion
from skimage.transform import resize, rotate
from sklearn.decomposition import PCA


class CellAverager:
    """
    Class in charge of building an average heatmap.

    Builds an average fluorescence heatmap by aligning per-cell cropped
    images to a common orientation and computing their mean.

    Parameters
    ----------
    fluor : numpy.ndarray
        Fluorescence field-of-view image used for per-cell crops.
    """

    def __init__(self, fluor):
        """Initialize the averager.

        Parameters
        ----------
        fluor : numpy.ndarray
            Full fluorescence field-of-view image. It is the channel that
            will be used to build the average heatmap.
        """

        self.fluor = fluor
        self.model = None
        self.aligned_fluor_masks = []

    def align(self, cell):
        """Align a cell crop to the common reference.

        The method computes the rotation angle from the cell's major axis and
        appends the rotated crop (fluorescence within the cell mask) to
        `aligned_fluor_masks`.

        Parameters
        ----------
        cell : napari_mAIcrobe.mAIcrobe.cells.Cell
            Cell object providing `image_box(fluor)` and `cell_mask`.
        """

        angle = self.calculate_rotation_angle(cell)
        self.aligned_fluor_masks.append(
            rotate(cell.image_box(self.fluor) * cell.cell_mask, angle)
        )

    def average(self):
        """Compute the average heatmap from aligned masks.

        Resizes aligned masks to a common median shape and computes the mean
        image, storing the result in `self.model`.
        """

        mean_x = int(np.median([s.shape[0] for s in self.aligned_fluor_masks]))
        mean_y = int(np.median([s.shape[1] for s in self.aligned_fluor_masks]))

        fluor_crops_array = [
            resize(s, (mean_x, mean_y)) for s in self.aligned_fluor_masks
        ]

        model_cell = np.zeros((mean_x, mean_y))
        for cell in fluor_crops_array:
            model_cell += cell
        model_cell /= float(len(fluor_crops_array))

        self.model = model_cell

    def calculate_rotation_angle(self, cell):
        """Estimate the rotation angle that aligns the cell's major axis
        vertically.

        Parameters
        ----------
        cell : napari_mAIcrobe.mAIcrobe.cells.Cell
            Cell instance used to extract the per-cell fluorescence and
            mask.

        Returns
        -------
        float
            Rotation angle in degrees to align the major axis.
        """

        binary = cell.image_box(self.fluor) * cell.cell_mask
        outline = self.calculate_cell_outline(binary)
        major_axis = self.calculate_major_axis(outline)
        return self.calculate_axis_angle(major_axis)

    @staticmethod
    def calculate_cell_outline(binary):
        """Compute the outline of a binary object.

        Parameters
        ----------
        binary : numpy.ndarray
            Binary image (non-zero values indicate object).

        Returns
        -------
        numpy.ndarray
            Binary image of the outline pixels.
        """

        outline = binary * (1 - binary_erosion(binary))

        return outline

    @staticmethod
    def calculate_major_axis(outline):
        """Compute major axis endpoints using PCA on outline coordinates.

        Parameters
        ----------
        outline : numpy.ndarray
            Binary image of the cell outline.

        Returns
        -------
        list[list[float]]
            Two endpoints [[x0, y0], [x1, y1]] of the major axis in image
            coordinates.
        """

        x, y = np.nonzero(outline)
        x = [[val] for val in x]
        y = [[val] for val in y]
        coords = np.concatenate((x, y), axis=1)

        pca = PCA(n_components=1)
        pca.fit(coords)

        pos_x, pos_y = pca.mean_
        eigenvector_x, eigenvector_y = pca.components_[0]
        eigenval = pca.explained_variance_[0]

        return [
            [
                pos_x - eigenvector_x * eigenval,
                pos_y - eigenvector_y * eigenval,
            ],
            [
                pos_x + eigenvector_x * eigenval,
                pos_y + eigenvector_y * eigenval,
            ],
        ]

    @staticmethod
    def calculate_axis_angle(major_axis):
        """Compute rotation angle from major axis endpoints.

        Notes
        -----
        TODO: refactor, atan2 should pick correct quadrant.

        Parameters
        ----------
        major_axis : list[list[float]]
            Two endpoints [[x0, y0], [x1, y1]] of the major axis.

        Returns
        -------
        float
            Rotation angle in degrees.
        """

        x0, y0 = major_axis[0]
        x1, y1 = major_axis[1]

        if x0 - x1 == 0:
            angle = 0.0

        elif y0 - y1 == 0:
            angle = 90.0

        else:
            if y1 > y0:
                if x1 > x0:
                    direction = -1
                    opposite = x1 - x0
                    adjacent = y1 - y0
                else:
                    direction = 1
                    opposite = x0 - x1
                    adjacent = y1 - y0

            elif y0 > y1:
                if x1 > x0:
                    direction = 1
                    opposite = x1 - x0
                    adjacent = y0 - y1
                else:
                    direction = -1
                    opposite = x0 - x1
                    adjacent = y0 - y1

            angle = math.degrees(math.atan(opposite / adjacent)) * direction

        if angle != 0:
            angle = 90.0 - angle
        else:
            angle = 90

        return angle
