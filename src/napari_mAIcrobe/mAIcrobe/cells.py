import math

import numpy as np
import pandas as pd
from skimage import exposure, morphology
from skimage.draw import line
from skimage.filters import threshold_isodata
from skimage.measure import label, regionprops_table
from skimage.util import img_as_float

from .cellaverager import CellAverager
from .cellcycleclassifier import CellCycleClassifier
from .cellprocessing import bound_rectangle, bounded_point, rotation_matrices
from .colocmanager import ColocManager
from .reports import ReportManager


class Cell:
    """Template for each cell object.

    Represents a single labeled cell with derived region masks (cell,
    membrane, cytoplasm, and optionally septum) alongside morphologic
    and fluorescence statistics.

    Parameters
    ----------
    label : int
        Integer label for this cell.
    regionmask : numpy.ndarray
        Binary mask for the cell region within the full image.
    properties : pandas.DataFrame
        Row of region properties (bbox, centroid, orientation, axis
        lengths, etc.), calculated from skimage's
        `regionprops_table`.
    intensity : numpy.ndarray
        Fluorescence image (primary channel).
    params : dict
        Analysis parameters dict controlling region computation and
        other params.
    optional : numpy.ndarray, optional
        Optional fluorescence image (e.g., DNA), by default None. If
        None, DNA visualization panels are rendered as zeros.

    Attributes
    ----------
    box : tuple[int, int, int, int]
        Bounding box (min_row, min_col, max_row, max_col) with padding.
    long_axis : numpy.ndarray
        Two endpoints defining the long axis, integer indices.
    short_axis : numpy.ndarray
        Two endpoints defining the short axis, integer indices.
    cell_mask : numpy.ndarray
        Cell region mask (cropped to bounding box).
    perim_mask : numpy.ndarray or None
        Membrane/perimeter mask (cropped).
    sept_mask : numpy.ndarray or None
        Septum mask (cropped), if computed.
    cyto_mask : numpy.ndarray or None
        Cytoplasm mask (cropped)
    membsept_mask : numpy.ndarray or None
        Union mask of membrane and septum (cropped), if computed.
    stats : dict
        Per-cell fluorescence and morphology statistics.
    image : numpy.ndarray or None
        Image mosaic of fluorescence and masks for visualization. Used
        for reports.
    """

    def __init__(
        self, label, regionmask, properties, intensity, params, optional=None
    ):
        """Construct a Cell object from the label, respective masks,
        parameters, and images.

        Parameters
        ----------
        label : int
            Cell label identifier.
        regionmask : numpy.ndarray
            Binary mask for the cell (same size as full image).
        properties : pandas.DataFrame
            Properties row for this cell (from `regionprops_table`).
        intensity : numpy.ndarray
            Fluorescence channel image.
        params : dict
            Analysis parameters controlling region and stats computation.
        optional : numpy.ndarray, optional
            Optional fluorescence channel image, by default None.
        """

        self.label = label

        # THESE 3 PARAMETERS HAVE TO GO
        # self.mask = regionmask.astype(int)
        # self.fluor = intensity
        # self.optional = optional

        self.params = params

        self.box_margin = 5

        # properties = regionprops(regionmask, intensity)[0]

        self.box = (
            properties["bbox-0"].item(),
            properties["bbox-1"].item(),
            properties["bbox-2"].item(),
            properties["bbox-3"].item(),
        )  # (min_row, min_col, max_row, max_col)

        w, h = intensity.shape
        self.img_shape = intensity.shape
        self.box = (
            max(self.box[0] - self.box_margin, 0),
            max(self.box[1] - self.box_margin, 0),
            min(self.box[2] + self.box_margin, w - 1),
            min(self.box[3] + self.box_margin, h - 1),
        )

        y0, x0 = (
            properties["centroid-0"].item(),
            properties["centroid-1"].item(),
        )
        x1 = (
            x0
            + math.cos(properties["orientation"].item())
            * 0.5
            * properties["axis_minor_length"].item()
        )
        y1 = (
            y0
            - math.sin(properties["orientation"].item())
            * 0.5
            * properties["axis_minor_length"].item()
        )
        x2 = (
            x0
            - math.cos(properties["orientation"].item())
            * 0.5
            * properties["axis_minor_length"].item()
        )
        y2 = (
            y0
            + math.sin(properties["orientation"].item())
            * 0.5
            * properties["axis_minor_length"].item()
        )

        # NOTE THE SWAP ON X AND Y
        self.long_axis = np.rint(np.array([[y1, x1], [y2, x2]])).astype(int)

        x1 = (
            x0
            - math.sin(properties["orientation"].item())
            * 0.5
            * properties["axis_major_length"].item()
        )
        y1 = (
            y0
            - math.cos(properties["orientation"].item())
            * 0.5
            * properties["axis_major_length"].item()
        )
        x2 = (
            x0
            + math.sin(properties["orientation"].item())
            * 0.5
            * properties["axis_major_length"].item()
        )
        y2 = (
            y0
            + math.cos(properties["orientation"].item())
            * 0.5
            * properties["axis_major_length"].item()
        )

        # NOTE THE SWAP ON X AND Y
        self.short_axis = np.rint(np.array([[y1, x1], [y2, x2]])).astype(int)

        # TODO: check if short/long axis can fall outside box.

        self.cell_mask = self.image_box(regionmask)
        self.fluor_mask = self.image_box(intensity)
        self.optional_mask = self.image_box(optional)

        self.perim_mask = None
        self.sept_mask = None
        self.cyto_mask = None
        self.membsept_mask = None

        self.stats = dict(
            [
                ("Baseline", 0),
                ("Cell Median", 0),
                ("Membrane Median", 0),
                ("Septum Median", 0),
                ("Cytoplasm Median", 0),
                ("Fluor Ratio", 0),
                ("Fluor Ratio 75%", 0),
                ("Fluor Ratio 25%", 0),
                ("Fluor Ratio 10%", 0),
                ("Cell Cycle Phase", 0),
                ("Area", properties["area"].item()),
                ("Perimeter", properties["perimeter"].item()),
                ("Eccentricity", properties["eccentricity"].item()),
            ]
        )

        self.selection_state = 1
        self.compute_regions(self.params)
        self.compute_fluor_stats(self.params, regionmask, intensity)

        self.image = None
        self.set_image(intensity, optional)

    def image_box(self, image):
        """Return an image crop corresponding to the cell bounding box.

        Parameters
        ----------
        image : numpy.ndarray or None
            Full image to crop; if None, returns None.

        Returns
        -------
        numpy.ndarray or None
            Cropped image of shape (x1-x0+1, y1-y0+1) or None.
        """
        x0, y0, x1, y1 = self.box
        try:
            return image[x0 : x1 + 1, y0 : y1 + 1]
        except TypeError:
            return None

    def compute_perim_mask(self, thick):
        """Compute membrane/perimeter mask by eroding the cell mask.

        Parameters
        ----------
        thick : int
            Thickness parameter controlling erosion.

        Returns
        -------
        numpy.ndarray
            Binary perimeter mask (float array with 0 and 1).
        """
        mask = self.cell_mask

        eroded = morphology.binary_erosion(
            mask, np.ones((thick * 2 - 1, thick - 1))
        ).astype(float)
        perim = mask - eroded

        return perim

    def compute_sept_mask(self, thick, algorithm):
        """Compute septum mask using a specified algorithm.

        Parameters
        ----------
        thick : int
            Thickness parameter for morphology.
        algorithm : {"Isodata", "Box"}
            Septum detection algorithm.

        Returns
        -------
        numpy.ndarray
            Binary septum mask.

        Notes
        -----
        Prints a warning if the algorithm name is invalid.
        """

        mask = self.cell_mask

        if algorithm == "Isodata":
            return self.compute_sept_isodata(thick)

        elif algorithm == "Box":
            return self.compute_sept_box(mask, thick)

        else:
            print("Not a a valid algorithm")

    def compute_opensept_mask(self, thick, algorithm):
        """Compute open-septum mask using a specified algorithm.

        Parameters
        ----------
        thick : int
            Thickness parameter for morphology.
        algorithm : {"Isodata", "Box"}
            Open-septum detection algorithm.

        Returns
        -------
        numpy.ndarray
            Binary open-septum mask.
        """

        mask = self.cell_mask

        if algorithm == "Isodata":
            return self.compute_opensept_isodata(mask, thick)
        elif algorithm == "Box":
            return self.compute_sept_box(thick)

        else:
            print("Not a a valid algorithm")

    def compute_sept_isodata(self, thick):
        """Create septum mask using isodata thresholding on inner region
        and separate the cytoplam from the septum.

        Parameters
        ----------
        thick : int
            Thickness parameter for the inner mask.

        Returns
        -------
        numpy.ndarray
            Binary septum mask.
        """
        cell_mask = self.cell_mask
        fluor_box = self.fluor_mask
        perim_mask = self.compute_perim_mask(thick)
        inner_mask = cell_mask - perim_mask
        inner_fluor = (inner_mask > 0) * fluor_box

        threshold = threshold_isodata(inner_fluor[inner_fluor > 0])
        interest_matrix = inner_mask * (inner_fluor > threshold)

        label_matrix = label(interest_matrix, connectivity=2)
        interest_label = 0
        interest_label_sum = 0

        for l in range(np.max(label_matrix)):
            if (
                np.sum(img_as_float(label_matrix == l + 1))
                > interest_label_sum
            ):
                interest_label = l + 1
                interest_label_sum = np.sum(
                    img_as_float(label_matrix == l + 1)
                )

        return img_as_float(label_matrix == interest_label)

    def compute_opensept_isodata(self, thick):
        """Create open-septum mask via isodata.

        Parameters
        ----------
        thick : int
            Thickness parameter for the inner mask.

        Returns
        -------
        numpy.ndarray
            Binary mask for one or two largest septal components.
        """
        cell_mask = self.cell_mask
        fluor_box = self.fluor_mask
        perim_mask = self.compute_perim_mask(thick)
        inner_mask = cell_mask - perim_mask
        inner_fluor = (inner_mask > 0) * fluor_box

        threshold = threshold_isodata(inner_fluor[inner_fluor > 0])
        interest_matrix = inner_mask * (inner_fluor > threshold)

        label_matrix = label(interest_matrix, connectivity=2)
        label_sums = []

        for l in range(np.max(label_matrix)):
            label_sums.append(np.sum(img_as_float(label_matrix == l + 1)))

        # print(label_sums)

        sorted_label_sums = sorted(label_sums)

        first_label = 0
        second_label = 0

        for i in range(len(label_sums)):
            if label_sums[i] == sorted_label_sums[-1]:
                first_label = i + 1
                label_sums.pop(i)
                break

        for i in range(len(label_sums)):
            if label_sums[i] == sorted_label_sums[-2]:
                second_label = i + 2
                label_sums.pop(i)
                break

        if second_label != 0:
            return img_as_float(
                (label_matrix == first_label) + (label_matrix == second_label)
            )
        else:
            return img_as_float(label_matrix == first_label)

    def compute_sept_box(self, thick):
        """Create a septum mask by creating a box around the cell and
        then defining the septum as the dilated short axis within the
        cell box.

        Parameters
        ----------
        thick : int
            Dilation kernel size.

        Returns
        -------
        numpy.ndarray
            Binary mask for the septum estimate.
        """

        mask = self.cell_mask

        x0, y0, x1, y1 = self.box
        lx0, ly0 = self.short_axis[0]
        lx1, ly1 = self.short_axis[1]
        x, y = line(lx0 - x0, ly0 - y0, lx1 - x0, ly1 - y0)

        linmask = np.zeros((x1 - x0 + 1, y1 - y0 + 1))
        linmask[x, y] = 1
        linmask = morphology.binary_dilation(
            linmask, np.ones((thick, thick))
        ).astype(float)

        if mask is not None:
            linmask = mask * linmask

        return linmask

    def get_outline_points(self, data):
        """Extract outline pixel coordinates from a binary mask. Used to
        get the outline pixels of the septum

        Parameters
        ----------
        data : numpy.ndarray
            Binary mask.

        Returns
        -------
        list[tuple[int, int]]
            List of (x, y) outline points.
        """
        outline = []
        for x in range(0, len(data)):
            for y in range(0, len(data[x])):
                if data[x, y] == 1:
                    if x == 0 and y == 0:
                        neighs_sum = (
                            data[x, y]
                            + data[x + 1, y]
                            + data[x + 1, y + 1]
                            + data[x, y + 1]
                        )
                    elif x == len(data) - 1 and y == len(data[x]) - 1:
                        neighs_sum = (
                            data[x, y]
                            + data[x, y - 1]
                            + data[x - 1, y - 1]
                            + data[x - 1, y]
                        )
                    elif x == 0 and y == len(data[x]) - 1:
                        neighs_sum = (
                            data[x, y]
                            + data[x, y - 1]
                            + data[x + 1, y - 1]
                            + data[x + 1, y]
                        )
                    elif x == len(data) - 1 and y == 0:
                        neighs_sum = (
                            data[x, y]
                            + data[x - 1, y]
                            + data[x - 1, y + 1]
                            + data[x, y + 1]
                        )
                    elif x == 0:
                        neighs_sum = (
                            data[x, y]
                            + data[x, y - 1]
                            + data[x, y + 1]
                            + data[x + 1, y - 1]
                            + data[x + 1, y]
                            + data[x + 1, y + 1]
                        )
                    elif x == len(data) - 1:
                        neighs_sum = (
                            data[x, y]
                            + data[x, y - 1]
                            + data[x, y + 1]
                            + data[x - 1, y - 1]
                            + data[x - 1, y]
                            + data[x - 1, y + 1]
                        )
                    elif y == 0:
                        neighs_sum = (
                            data[x, y]
                            + data[x - 1, y]
                            + data[x + 1, y]
                            + data[x - 1, y + 1]
                            + data[x, y + 1]
                            + data[x + 1, y + 1]
                        )
                    elif y == len(data[x]) - 1:
                        neighs_sum = (
                            data[x, y]
                            + data[x - 1, y]
                            + data[x + 1, y]
                            + data[x - 1, y - 1]
                            + data[x, y - 1]
                            + data[x + 1, y - 1]
                        )
                    else:
                        neighs_sum = (
                            data[x, y]
                            + data[x - 1, y]
                            + data[x + 1, y]
                            + data[x - 1, y - 1]
                            + data[x, y - 1]
                            + data[x + 1, y - 1]
                            + data[x - 1, y + 1]
                            + data[x, y + 1]
                            + data[x + 1, y + 1]
                        )
                    if neighs_sum != 9:
                        outline.append((x, y))
        return outline

    def compute_sept_box_fix(self, outline, maskshape):
        """Method used to create a box around the septum, so that the
        short axis of this box can be used to choose the pixels of the
        membrane mask that need to be removed.

        Parameters
        ----------
        outline : list[tuple[int, int]]
            Outline points of the septum.
        maskshape : tuple[int, int]
            Shape of the mask to clamp coordinates.

        Returns
        -------
        tuple[int, int, int, int]
            Bounding box (x0, y0, x1, y1).
        """
        points = np.asarray(outline)  # in two columns, x, y
        bm = self.box_margin
        w, h = maskshape
        box = (
            max(min(points[:, 0]) - bm, 0),
            max(min(points[:, 1]) - bm, 0),
            min(max(points[:, 0]) + bm, w - 1),
            min(max(points[:, 1]) + bm, h - 1),
        )

        return box

    def remove_sept_from_membrane(self, maskshape):
        """Remove septum pixels from the membrane mask.

        Parameters
        ----------
        maskshape : tuple[int, int]
            Shape of the septum/membrane masks.

        Returns
        -------
        numpy.ndarray
            Binary line mask used to subtract from membrane.
        """

        # get outline points of septum mask
        septum_outline = []
        septum_mask = self.sept_mask
        septum_outline = self.get_outline_points(septum_mask)

        # compute box of the septum
        septum_box = self.compute_sept_box_fix(septum_outline, maskshape)

        # compute axis of the septum
        rotations = rotation_matrices(5)
        points = np.asarray(septum_outline)  # in two columns, x, y
        width = len(points) + 1

        # no need to do more rotations, due to symmetry
        for rix in range(int(len(rotations) / 2) + 1):
            r = rotations[rix]
            nx0, ny0, nx1, ny1, nwidth = bound_rectangle(
                np.asarray(np.dot(points, r))
            )

            if nwidth < width:
                width = nwidth
                x0 = nx0
                x1 = nx1
                y0 = ny0
                y1 = ny1
                angle = rix

        rotation = rotations[angle]

        # midpoints
        mx = (x1 + x0) / 2
        my = (y1 + y0) / 2

        # assumes long is X. This duplicates rotations but simplifies
        # using different algorithms such as brightness
        long = [[x0, my], [x1, my]]
        short = [[mx, y0], [mx, y1]]
        short = np.asarray(np.dot(short, rotation.T), dtype=np.int32)
        long = np.asarray(np.dot(long, rotation.T), dtype=np.int32)

        # check if axis fall outside area due to rounding errors
        bx0, by0, bx1, by1 = septum_box
        short[0] = bounded_point(bx0, bx1, by0, by1, short[0])
        short[1] = bounded_point(bx0, bx1, by0, by1, short[1])
        long[0] = bounded_point(bx0, bx1, by0, by1, long[0])
        long[1] = bounded_point(bx0, bx1, by0, by1, long[1])

        length = np.linalg.norm(long[1] - long[0])
        width = np.linalg.norm(short[1] - short[0])

        if length < width:
            dum = length
            length = width
            width = dum
            dum = short
            short = long
            long = dum

        # expand long axis to create a linmask
        bx0, by0 = long[0]
        bx1, by1 = long[1]

        h, w = self.sept_mask.shape
        linmask = np.zeros((h, w))

        h, w = self.sept_mask.shape[0] - 2, self.sept_mask.shape[1] - 2
        bin_factor = int(width)

        if bx1 - bx0 == 0:
            x, y = line(bx0, 0, bx0, w)
            linmask[x, y] = 1
            try:
                linmask = morphology.binary_dilation(
                    linmask, np.ones((bin_factor, bin_factor))
                ).astype(float)
            except RuntimeError:
                bin_factor = 4
                linmask = morphology.binary_dilation(
                    linmask, np.ones((bin_factor, bin_factor))
                ).astype(float)

        else:
            m = (by1 - by0) / (bx1 - bx0)
            b = by0 - m * bx0

            if b < 0:
                l_y0 = 0
                l_x0 = int(-b / m)

                if h * m + b > w:
                    l_y1 = w
                    l_x1 = int((w - b) / m)

                else:
                    l_x1 = h
                    l_y1 = int(h * m + b)

            elif b > w:
                l_y0 = w
                l_x0 = int((w - b) / m)

                if h * m + b < 0:
                    l_y1 = 0
                    l_x1 = int(-b / m)

                else:
                    l_x1 = h
                    l_y1 = int((h - b) / m)

            else:
                l_x0 = 0
                l_y0 = int(b)

                if m > 0:
                    if h * m + b > w:
                        l_y1 = w
                        l_x1 = int((w - b) / m)
                    else:
                        l_x1 = h
                        l_y1 = int(h * m + b)

                elif m < 0:
                    if h * m + b < 0:
                        l_y1 = 0
                        l_x1 = int(-b / m)
                    else:
                        l_x1 = h
                        l_y1 = int(h * m + b)

                else:
                    l_x1 = h
                    l_y1 = int(b)

            x, y = line(l_x0, l_y0, l_x1, l_y1)
            linmask[x, y] = 1
            try:
                linmask = morphology.binary_dilation(
                    linmask, np.ones((bin_factor, bin_factor))
                ).astype(float)
            except RuntimeError:
                bin_factor = 4
                linmask = morphology.binary_dilation(
                    linmask, np.ones((bin_factor, bin_factor))
                ).astype(float)
        return img_as_float(linmask)

    def recursive_compute_sept(self, inner_mask_thickness, algorithm):
        """Compute septum mask, reducing thickness on failure.

        Parameters
        ----------
        inner_mask_thickness : int
            Initial thickness to try.
        algorithm : {"Isodata", "Box"}
            Septum detection algorithm.
        """
        try:
            self.sept_mask = self.compute_sept_mask(
                inner_mask_thickness, algorithm
            )
        except IndexError:
            try:
                self.recursive_compute_sept(
                    inner_mask_thickness - 1, algorithm
                )
            except RuntimeError:
                self.recursive_compute_sept(inner_mask_thickness - 1, "Box")

    def recursive_compute_opensept(self, inner_mask_thickness, algorithm):
        """Compute open-septum mask, reducing thickness on failure.

        Parameters
        ----------
        inner_mask_thickness : int
            Initial thickness to try.
        algorithm : {"Isodata", "Box"}
            Open-septum detection algorithm.
        """
        try:
            self.sept_mask = self.compute_opensept_mask(
                inner_mask_thickness, algorithm
            )
        except IndexError:
            try:
                self.recursive_compute_opensept(
                    inner_mask_thickness - 1, algorithm
                )
            except RuntimeError:
                self.recursive_compute_opensept(
                    inner_mask_thickness - 1, "Box"
                )

    def compute_regions(self, params):
        """Compute masks for whole cell, membrane, septum (optional),
        and cytoplasm.

        Parameters
        ----------
        params : dict
            Analysis parameters controlling septum detection and
            thickness.
        """

        if params["find_septum"]:
            self.recursive_compute_sept(
                params["inner_mask_thickness"], params["septum_algorithm"]
            )

            if params["septum_algorithm"] == "Isodata":
                self.perim_mask = self.compute_perim_mask(
                    params["inner_mask_thickness"]
                )
                self.membsept_mask = (self.perim_mask + self.sept_mask) > 0
                linmask = self.remove_sept_from_membrane(self.img_shape)
                self.cyto_mask = (
                    self.cell_mask - self.perim_mask - self.sept_mask
                ) > 0
                if linmask is not None:
                    old_membrane = self.perim_mask
                    self.perim_mask = (old_membrane - linmask) > 0
            else:
                self.perim_mask = (
                    self.compute_perim_mask(params["inner_mask_thickness"])
                    - self.sept_mask
                ) > 0
                self.membsept_mask = (self.perim_mask + self.sept_mask) > 0
                self.cyto_mask = (
                    self.cell_mask - self.perim_mask - self.sept_mask
                ) > 0
        elif params["find_openseptum"]:
            self.recursive_compute_opensept(
                params["inner_mask_thickness"], params["septum_algorithm"]
            )

            if params["septum_algorithm"] == "Isodata":
                self.perim_mask = self.compute_perim_mask(
                    params["inner_mask_thickness"]
                )

                self.membsept_mask = (self.perim_mask + self.sept_mask) > 0
                linmask = self.remove_sept_from_membrane(self.img_shape)
                self.cyto_mask = (
                    self.cell_mask - self.perim_mask - self.sept_mask
                ) > 0
                if linmask is not None:
                    old_membrane = self.perim_mask
                    self.perim_mask = (old_membrane - linmask) > 0
            else:
                self.perim_mask = (
                    self.compute_perim_mask(params["inner_mask_thickness"])
                    - self.sept_mask
                ) > 0
                self.membsept_mask = (self.perim_mask + self.sept_mask) > 0
                self.cyto_mask = (
                    self.cell_mask - self.perim_mask - self.sept_mask
                ) > 0
        else:
            self.sept_mask = None
            self.perim_mask = self.compute_perim_mask(
                params["inner_mask_thickness"]
            )
            self.cyto_mask = (self.cell_mask - self.perim_mask) > 0

    def compute_fluor_baseline(self, mask, fluor, margin):
        """Compute baseline fluorescence around the cell. Mask and fluor
        are the global images.

        Parameters
        ----------
        mask : numpy.ndarray
            Global mask image where 0 indicates cell regions and 1
            indicates background.
        fluor : numpy.ndarray
            Full-field fluorescence image.
        margin : int
            Margin to expand the bounding box for baseline calculation.

        Notes
        -----
        Mask is 0 (black) at cells and 1 (white) outside
        Updates self.stats["Baseline"] with the computed median baseline
        fluorescence.
        """
        # compatibility
        mask = 1 - mask

        # here zero is cell
        x0, y0, x1, y1 = self.box
        wid, hei = mask.shape
        x0 = max(x0 - margin, 0)
        y0 = max(y0 - margin, 0)
        x1 = min(x1 + margin, wid - 1)
        y1 = min(y1 + margin, hei - 1)
        mask_box = mask[x0:x1, y0:y1]

        count = 0
        # here zero is background
        inverted_mask_box = 1 - mask_box

        while count < 5:
            inverted_mask_box = morphology.binary_dilation(inverted_mask_box)
            count += 1

        # here zero is cell
        mask_box = 1 - inverted_mask_box

        fluor_box = fluor[x0:x1, y0:y1]
        self.stats["Baseline"] = np.median(
            mask_box[mask_box > 0] * fluor_box[mask_box > 0]
        )

    def measure_fluor(self, fluorbox, roi, fraction=1.0):
        """Computes median fluorescence in a region of interest (ROI).

        Parameters
        ----------
        fluorbox : numpy.ndarray
            Cropped fluorescence image corresponding to the cell
            bounding box.
        roi : numpy.ndarray
            Binary mask for the region of interest (same shape as
            fluorbox).
        fraction : float, optional
            Fraction of brightest pixels to consider (0 < fraction <= 1),
            by default 1.0

        Returns
        -------
        float
            Median fluorescence in the ROI, considering only the
            specified fraction of brightest pixels.

        Notes
        -----
        fraction=0.1 means median of the top 10% brightest pixels in the
        ROI
        fluorbox and roi must be the same shape
        """
        if roi is not None:
            bright = fluorbox * roi
            bright = bright[roi > 0.5]
            # check if not enough points

            if (len(bright) * fraction) < 1.0:
                return 0.0

            if fraction < 1:
                sortvals = np.sort(bright, axis=None)[::-1]
                sortvals = sortvals[np.nonzero(sortvals)]
                sortvals = sortvals[: int(len(sortvals) * fraction)]
                return np.median(sortvals)

            else:
                return np.median(bright)
        else:
            return 0

    def compute_fluor_stats(self, params, mask, fluor):
        """Compute per-region fluorescence statistics and ratios.

        Parameters
        ----------
        params : dict
            Analysis parameters including `find_septum` and
            `baseline_margin`.
        mask : numpy.ndarray
            Global mask image used for baseline.
        fluor : numpy.ndarray
            Full-field fluorescence image.
        """
        self.compute_fluor_baseline(mask, fluor, params["baseline_margin"])

        fluorbox = self.fluor_mask

        self.stats["Cell Median"] = (
            self.measure_fluor(fluorbox, self.cell_mask)
            - self.stats["Baseline"]
        )

        self.stats["Membrane Median"] = (
            self.measure_fluor(fluorbox, self.perim_mask)
            - self.stats["Baseline"]
        )

        self.stats["Cytoplasm Median"] = (
            self.measure_fluor(fluorbox, self.cyto_mask)
            - self.stats["Baseline"]
        )

        if params["find_septum"] or params["find_openseptum"]:
            self.stats["Septum Median"] = (
                self.measure_fluor(fluorbox, self.sept_mask)
                - self.stats["Baseline"]
            )

            self.stats["Fluor Ratio"] = (
                self.measure_fluor(fluorbox, self.sept_mask)
                - self.stats["Baseline"]
            ) / (
                self.measure_fluor(fluorbox, self.perim_mask)
                - self.stats["Baseline"]
            )

            self.stats["Fluor Ratio 75%"] = (
                self.measure_fluor(fluorbox, self.sept_mask, 0.75)
                - self.stats["Baseline"]
            ) / (
                self.measure_fluor(fluorbox, self.perim_mask)
                - self.stats["Baseline"]
            )

            self.stats["Fluor Ratio 25%"] = (
                self.measure_fluor(fluorbox, self.sept_mask, 0.25)
                - self.stats["Baseline"]
            ) / (
                self.measure_fluor(fluorbox, self.perim_mask)
                - self.stats["Baseline"]
            )

            self.stats["Fluor Ratio 10%"] = (
                self.measure_fluor(fluorbox, self.sept_mask, 0.10)
                - self.stats["Baseline"]
            ) / (
                self.measure_fluor(fluorbox, self.perim_mask)
                - self.stats["Baseline"]
            )
            self.stats["Memb+Sept Median"] = (
                self.measure_fluor(fluorbox, self.membsept_mask)
                - self.stats["Baseline"]
            )

        else:
            self.stats["Septum Median"] = 0

            self.stats["Fluor Ratio"] = 0

            self.stats["Fluor Ratio 75%"] = 0

            self.stats["Fluor Ratio 25%"] = 0

            self.stats["Fluor Ratio 10%"] = 0

            self.stats["Memb+Sept Median"] = 0

    def set_image(self, fluor, optional):
        """Compose a 7-panel per-cell visualization image.

        Panels include raw and masked channels and region-specific
        overlays.

        Parameters
        ----------
        fluor : numpy.ndarray
            Fluorescence image.
        optional : numpy.ndarray or None
            Optional fluorescence image. If None, a zero-valued image
            is used for DNA panels.
        """

        fluor = img_as_float(fluor)
        fluor = exposure.rescale_intensity(fluor)

        if optional is None:
            optional = np.zeros_like(fluor)

        optional = img_as_float(optional)
        optional = exposure.rescale_intensity(optional)

        perim = self.perim_mask
        axial = self.sept_mask
        cyto = self.cyto_mask

        x0, y0, x1, y1 = self.box
        img = np.zeros((x1 - x0 + 1, 7 * (y1 - y0 + 1)))
        bx0 = 0
        bx1 = x1 - x0 + 1
        by0 = 0
        by1 = y1 - y0 + 1

        # 7 images

        # #1 is the fluorescence
        img[bx0:bx1, by0:by1] = fluor[x0 : x1 + 1, y0 : y1 + 1]
        by0 = by0 + y1 - y0 + 1
        by1 = by1 + y1 - y0 + 1

        # #2 is the fluorescence segmented
        img[bx0:bx1, by0:by1] = (
            fluor[x0 : x1 + 1, y0 : y1 + 1] * self.cell_mask
        )
        by0 = by0 + y1 - y0 + 1
        by1 = by1 + y1 - y0 + 1

        # #3 is the dna
        img[bx0:bx1, by0:by1] = optional[x0 : x1 + 1, y0 : y1 + 1]
        by0 = by0 + y1 - y0 + 1
        by1 = by1 + y1 - y0 + 1

        # #4 is the dna segmented
        img[bx0:bx1, by0:by1] = (
            optional[x0 : x1 + 1, y0 : y1 + 1] * self.cell_mask
        )
        by0 = by0 + y1 - y0 + 1
        by1 = by1 + y1 - y0 + 1

        # 5,6,7 is perimeter, cytoplasm and septa
        img[bx0:bx1, by0:by1] = fluor[x0 : x1 + 1, y0 : y1 + 1] * perim
        by0 = by0 + y1 - y0 + 1
        by1 = by1 + y1 - y0 + 1

        img[bx0:bx1, by0:by1] = fluor[x0 : x1 + 1, y0 : y1 + 1] * cyto

        if self.params["find_septum"] or self.params["find_openseptum"]:
            by0 = by0 + y1 - y0 + 1
            by1 = by1 + y1 - y0 + 1
            img[bx0:bx1, by0:by1] = fluor[x0 : x1 + 1, y0 : y1 + 1] * axial

        self.image = img


class CellManager:
    """
    Manages cell property computations, classifications, colocalization,
    and reporting.

    Parameters
    ----------
    label_img : ndarray
        Labeled image. Supported shapes are 2D `(Y, X)` and timelapse
        2D+t `(T, Y, X)`.
    fluor : ndarray
        Primary fluorescence image matching `label_img` shape.
    optional : ndarray or None
        Optional secondary image (e.g., DNA) matching `label_img`
        shape. Can be None.
    params : dict
        Dictionary of parameters controlling the behavior of the class.
        Keys include:
        - "classify_cell_cycle" : bool
            Whether to classify the cell cycle phase.
        - "model" : str
            Model type for cell cycle classification.
        - "custom_model_path" : str
            Path to a custom model for classification.
        - "custom_model_input" : int
            Input size for the custom model.
        - "custom_model_maxsize" : int
            Maximum size for the custom model.
        - "cell_averager" : bool
            Whether to perform cell averaging.
        - "coloc" : bool
            Whether to compute colocalization metrics.
        - "generate_report" : bool
            Whether to generate a report.
        - "report_path" : str
            Path to save the generated report.
        - "report_id" : str, optional
            Identifier for the report.
        - "find_septum" : bool
            Whether to find the septum in cells.
        - "find_openseptum" : bool
            Whether to find open septa in cells.
        - "inner_mask_thickness" : int
            Thickness for inner mask computation.
        - "septum_algorithm" : str
            Algorithm for septum detection ("Isodata" or "Box").
        - "baseline_margin" : int
            Margin for baseline fluorescence calculation.

    Attributes
    ----------
    label_img : ndarray
        Labeled image where each cell is represented by a unique
        integer.
    fluor_img : ndarray
        Fluorescence image corresponding to the labeled image.
    optional_img : ndarray
        Optional image used for additional calculations.
    params : dict
        Dictionary of parameters controlling the behavior of the class.
    properties : dict or None
        Dictionary containing per-cell properties. In
        timelapse mode, cells from all frames are combined and include a `frame`
        key. Keys include:
        - "frame"
        - "label"
        - "Area"
        - "Perimeter"
        - "Eccentricity"
        - "Baseline"
        - "Cell Median"
        - "Membrane Median"
        - "Septum Median"
        - "Cytoplasm Median"
        - "Fluor Ratio"
        - "Fluor Ratio 75%"
        - "Fluor Ratio 25%"
        - "Fluor Ratio 10%"
        - "Cell Cycle Phase"
        - "DNA Ratio"
    heatmap_model : ndarray or None
        Heatmap model generated by the cell averager, if applicable.
    all_cells : list or None
        List of all cell images, used for generating reports.

    Methods
    -------
    compute_cell_properties()
        Computes various properties for each cell in the labeled image(s).
    calculate_DNARatio(cell_object, dna_fov, thresh)
        Static method to calculate the ratio of area that has
        discernable DNA signal for a given cell.

    Notes
    -----
    This class integrates multiple functionalities such as cell
    property computation, cell cycle classification, colocalization
    analysis, and report generation.
    """

    def __init__(self, label_img, fluor, optional, params):
        """
        Initialize the class with the provided images and parameters.

        Parameters
        ----------
        label_img : ndarray
            Label image `(Y, X)` or timelapse label stack `(T, Y, X)`.
        fluor : ndarray
            Primary fluorescence image/stack with shape matching
            `label_img`.
        optional : ndarray or None
            Optional secondary fluorescence image/stack with shape
            matching `label_img`.
        params : dict
            A dictionary of parameters used for processing or analysis.

        Attributes:
        -----------
        label_img : ndarray
            Stores the labeled image.
        fluor_img : ndarray
            Stores the fluorescence image.
        optional_img : ndarray
            Stores the optional image.
        params : dict
            Stores the parameters dictionary.
        properties : None or dict
            Placeholder for storing computed properties of the cells.
        heatmap_model : None or object
            Placeholder for storing a heatmap model.
        all_cells : None or object
            Placeholder for storing all cell-related data.
        """

        self.label_img = label_img
        self.fluor_img = fluor
        self.optional_img = optional

        self.params = params

        self.properties = None
        self.heatmap_model = None

        self.all_cells = None

    def _model_requires_dna(self):
        """Return True if the configured classifier input needs DNA."""

        if self.params["model"] == "custom":
            return "DNA" in self.params["custom_model_input"]

        return "DNA" in self.params["model"]

    @staticmethod
    def _compute_dna_threshold(label_img, optional_img):
        """Compute DNA threshold for one frame.

        Returns NaN when no optional image is available or when no
        positive optional signal is present.
        """

        if optional_img is None:
            return np.nan

        optional_img_cells = optional_img * (label_img > 0).astype(int)
        nonzero = optional_img_cells[np.nonzero(optional_img_cells)]
        if nonzero.size == 0:
            return np.nan

        histcounts, binedges = np.histogram(nonzero, bins="auto")
        maxintensity = binedges[np.argmax(histcounts) + 1]

        optimg = optional_img.copy()
        optimg[optimg >= maxintensity] = maxintensity
        opt_nonzero = optimg[np.nonzero(optimg)]
        if opt_nonzero.size == 0:
            return np.nan

        return threshold_isodata(opt_nonzero)

    def _append_cell_row(self, rows, c, frame_index, dna_img, dnathresh):
        """Append one cell row to accumulator lists.

        DNA ratio is stored as NaN when DNA data is unavailable for the
        frame.
        """

        rows["frame"].append(frame_index)
        rows["label"].append(c.label)
        rows["Area"].append(c.stats["Area"])
        rows["Perimeter"].append(c.stats["Perimeter"])
        rows["Eccentricity"].append(c.stats["Eccentricity"])
        rows["Baseline"].append(c.stats["Baseline"])
        rows["Cell Median"].append(c.stats["Cell Median"])
        rows["Membrane Median"].append(c.stats["Membrane Median"])
        rows["Septum Median"].append(c.stats["Septum Median"])
        rows["Cytoplasm Median"].append(c.stats["Cytoplasm Median"])
        rows["Fluor Ratio"].append(c.stats["Fluor Ratio"])
        rows["Fluor Ratio 75%"].append(c.stats["Fluor Ratio 75%"])
        rows["Fluor Ratio 25%"].append(c.stats["Fluor Ratio 25%"])
        rows["Fluor Ratio 10%"].append(c.stats["Fluor Ratio 10%"])
        rows["Cell Cycle Phase"].append(c.stats["Cell Cycle Phase"])

        if dna_img is None or np.isnan(dnathresh):
            rows["DNA Ratio"].append(np.nan)
        else:
            rows["DNA Ratio"].append(
                self.calculate_DNARatio(c, dna_img, dnathresh)
            )

    @staticmethod
    def _init_rows_dict():
        """Create property accumulators for output rows."""
        return {
            "frame": [],
            "label": [],
            "Area": [],
            "Perimeter": [],
            "Eccentricity": [],
            "Baseline": [],
            "Cell Median": [],
            "Membrane Median": [],
            "Septum Median": [],
            "Cytoplasm Median": [],
            "Fluor Ratio": [],
            "Fluor Ratio 75%": [],
            "Fluor Ratio 25%": [],
            "Fluor Ratio 10%": [],
            "Cell Cycle Phase": [],
            "DNA Ratio": [],
        }

    @staticmethod
    def _rows_to_properties(rows):
        """Convert list in dicts to np arrays."""
        return {key: np.array(values) for key, values in rows.items()}

    def _frame_data(self, frame_index):
        """Return one frame as 2D arrays.

        For 2D inputs, returns the original arrays regardless of
        `frame_index`.
        """

        if self.label_img.ndim == 2:
            return self.label_img, self.fluor_img, self.optional_img

        optional = None
        if self.optional_img is not None:
            optional = self.optional_img[frame_index]

        return (
            self.label_img[frame_index],
            self.fluor_img[frame_index],
            optional,
        )

        def compute_cell_properties(self):
            """Compute per-cell properties from 2D or 2D+t timelapse data.

            The method validates input shapes, processes each image or each frame
            independently for 2D+t inputs, and stores property arrays in
            `self.properties`.

            Notes
            -----
            - Timelapse mode is enabled for `(T, Y, X)` arrays and adds a
                `frame` property column.
            - No tracking is performed; labels are treated independently per
                frame.
            - DNA-dependent metrics (`DNA Ratio`, colocalization) are
                skipped or set to NaN when optional input is unavailable.
            - Classification raises a ValueError when the selected model
                requires DNA but no optional input is provided.
            """

        if self.label_img.ndim not in (2, 3):
            raise ValueError("label_img must be 2D or 3D (T, Y, X)")

        if self.fluor_img.ndim != self.label_img.ndim:
            raise ValueError("label_img and fluor_img must have same dims")

        if self.fluor_img.shape != self.label_img.shape:
            raise ValueError("label_img and fluor_img must have same shape")

        if self.optional_img is not None:
            if self.optional_img.ndim != self.label_img.ndim:
                raise ValueError(
                    "optional_img and label_img must have same dims"
                )
            if self.optional_img.shape != self.label_img.shape:
                raise ValueError(
                    "optional_img and label_img must have same shape"
                )

        if self.params["classify_cell_cycle"] and self.optional_img is None:
            if self._model_requires_dna():
                raise ValueError(
                    "Selected cell cycle model requires DNA image, "
                    "but DNA image is missing."
                )

        timelapse = self.label_img.ndim == 3
        self.params["include_frame"] = timelapse

        rows = self._init_rows_dict()
        all_cells = []

        ccc = None
        if self.params["classify_cell_cycle"] and not timelapse:
            ccc = CellCycleClassifier(
                self.fluor_img,
                self.optional_img,
                self.params["model"],
                self.params["custom_model_path"],
                self.params["custom_model_input"],
                self.params["custom_model_maxsize"],
            )

        ca = None
        if self.params["cell_averager"] and not timelapse:
            print("Cell averager...")
            ca = CellAverager(self.fluor_img)

        coloc = None
        coloc_enabled = self.params["coloc"] and self.optional_img is not None
        if self.params["coloc"] and self.optional_img is None:
            print("Colocalization skipped: Optional image not provided.")
        if coloc_enabled:
            coloc = ColocManager()

        n_frames = self.label_img.shape[0] if timelapse else 1
        print("Per cell stats...")

        for frame_index in range(n_frames):
            label_img, fluor_img, optional_img = self._frame_data(frame_index)

            if self.params["classify_cell_cycle"] and timelapse:
                ccc = CellCycleClassifier(
                    fluor_img,
                    optional_img,
                    self.params["model"],
                    self.params["custom_model_path"],
                    self.params["custom_model_input"],
                    self.params["custom_model_maxsize"],
                )

            dnathresh = self._compute_dna_threshold(label_img, optional_img)

            proptable = pd.DataFrame(
                regionprops_table(
                    label_img,
                    properties=[
                        "label",
                        "bbox",
                        "centroid",
                        "orientation",
                        "axis_minor_length",
                        "axis_major_length",
                        "area",
                        "perimeter",
                        "eccentricity",
                    ],
                )
            )

            label_list = np.unique(label_img)
            for l in label_list:
                if l == 0:
                    continue

                mask = (label_img == l).astype(int)
                c = Cell(
                    label=l,
                    regionmask=mask,
                    intensity=fluor_img,
                    properties=proptable[proptable["label"] == l],
                    params=self.params,
                    optional=optional_img,
                )

                if self.params["generate_report"]:
                    all_cells.append(c.image)

                if self.params["cell_averager"]:
                    ca.fluor = fluor_img
                    ca.align(c)

                if self.params["classify_cell_cycle"]:
                    c.stats["Cell Cycle Phase"] = ccc.classify_cell(c)
                else:
                    c.stats["Cell Cycle Phase"] = 0

                self._append_cell_row(
                    rows,
                    c,
                    frame_index,
                    optional_img,
                    dnathresh,
                )

                if coloc_enabled:
                    report_key = str(c.label)
                    if timelapse:
                        report_key = f"{frame_index}:{c.label}"
                    coloc.computes_cell_pcc(
                        fluor_img,
                        optional_img,
                        c,
                        self.params,
                        cell_label=report_key,
                    )

        self.properties = self._rows_to_properties(rows)

        if self.params["cell_averager"] and len(ca.aligned_fluor_masks) > 0:
            ca.average()
            self.heatmap_model = ca.model

        if self.params["generate_report"]:
            self.all_cells = all_cells
            rm = ReportManager(
                parameters=self.params,
                properties=self.properties,
                allcells=all_cells,
            )
            rm.generate_report(
                self.params["report_path"],
                report_id=self.params.get("report_id", None),
            )
            if coloc_enabled:
                coloc.save_report(
                    rm.cell_data_filename, self.params["find_septum"]
                )

    @staticmethod
    def calculate_DNARatio(cell_object, dna_fov, thresh):
        """Calculate the ratio of area that has discernable DNA signal
        for a given cell.

        Parameters
        ----------
        cell_object : Cell
            The cell object for which to calculate the DNA ratio.
        dna_fov : np.ndarray or None
            The field-of-view image containing the DNA signal.
        thresh : float
            The threshold value for determining discernable DNA signal.

        Returns
        -------
        float
            The ratio of discernable DNA signal area to total cell area,
            or NaN when DNA data/threshold is unavailable.
        """

        if dna_fov is None or np.isnan(thresh):
            return np.nan

        x0, y0, x1, y1 = cell_object.box
        cell_mask = cell_object.cell_mask
        optional_cell = dna_fov[x0 : x1 + 1, y0 : y1 + 1]
        optional_signal = (optional_cell * cell_mask) > thresh

        return np.sum(optional_signal) / np.sum(cell_mask)
