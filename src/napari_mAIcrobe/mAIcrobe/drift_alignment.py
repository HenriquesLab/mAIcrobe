import os
from datetime import datetime
from importlib import metadata
from math import fabs, sqrt

import cv2
import numpy as np
import scipy as sp
from scipy.interpolate import RegularGridInterpolator as _interpolate
from scipy.interpolate import interp1d
from scipy.optimize import minimize

# FUNCTIONS HEAVILY INSPIRED ON NANOPYX'S IMPLEMENTATION
# https://github.com/HenriquesLab/NanoPyx


def estimate_drift_alignment(
    image_array,
    save_as_npy=True,
    save_drift_table_path=None,
    roi=None,
    return_drift_table=False,
    **kwargs
):
    """
    Function use to estimate the drift in a microscopy image.
    :param image_array: numpy array  with shape (z, y, x)
    :param save_as_npy (optional): bool, whether to save as npy (if true) or csv (if false)
    :param save_drift_table_path (optional): str, path to save drift table
    :param roi (optional): in case of use should have shape (x0, y0, x1, y1)
    :param kwargs: additional keyword arguments
    :param return_drift_table (optional): bool, if True returns
        `(aligned_image, estimator_table)`.
    :return: aligned image as numpy array, or tuple with estimator table
        when `return_drift_table=True`
    """
    estimator = DriftEstimator()
    corrected_img = estimator.estimate(image_array, roi=roi, **kwargs)
    print(save_drift_table_path)
    estimator.save_drift_table(
        save_as_npy=save_as_npy, path=save_drift_table_path
    )
    if corrected_img is not None and return_drift_table:
        return corrected_img, estimator.estimator_table
    if corrected_img is not None:
        return corrected_img
    else:
        pass


def apply_drift_alignment(image_array, path=None, drift_table=None):
    """
    Function used to correct the drift in a microscopy image given a previously calculated drift table.
    :param image_array: numpy array  with shape (z, y, x); image to be corrected
    :param path (optional): str; path to previously saved
    :param drift_table (optional): estimator table object; object containing previously calculated drift table
    :return: aligned image as numpy array
    """
    corrector = DriftCorrector()
    if drift_table is None:
        corrector.load_estimator_table(path=path)
    else:
        corrector.estimator_table = drift_table
    corrected_img = corrector.apply_correction(image_array)
    return corrected_img


class DriftCorrector:
    """
    Main class for aligning timelapse images with drift.

    This class is used for aligning timelapse images with drift correction. It requires a previously calculated drift table.
    The class implements the following methods:
    - apply_correction
    - load_estimator_table
    - _translate_slice

    Args:
        None

    Attributes:
        estimator_table (DriftEstimatorTable): An instance of the DriftEstimatorTable containing the drift table data.
        image_arr (numpy.ndarray): The timelapse image array with shape (n_slices, rows, columns).

    Methods:
        __init__(): Initialize the `DriftCorrector` object.

        _translate_slice(slice_idx): Translate an individual image slice based on the drift table.

        apply_correction(image_array): Apply drift correction to the entire image array.

        load_estimator_table(path=None): Load the drift table from a file.

    Example:
        corrector = DriftCorrector()
        corrected_image = corrector.apply_correction(image_array)
        corrector.load_estimator_table("drift_table.csv")

    Note:
        The `DriftCorrector` class is used for correcting drift in timelapse images using a precomputed drift table.
    """

    def __init__(self):
        """
        Initialize the `DriftCorrector` object.

        Args:
            None

        Returns:
            None

        Example:
            corrector = DriftCorrector()
        """
        self.estimator_table = DriftEstimatorTable()
        self.image_arr = None

    def _translate_slice(self, slice_idx):
        """
        Translate an individual image slice based on the drift table.

        Args:
            slice_idx (int): The index of the slice to be translated.

        Returns:
            numpy.ndarray: The translated image slice.

        Example:
            translated_slice = self._translate_slice(0)

        Note:
            This method is used to translate individual image slices based on the drift information in the drift table.
        """
        drift_x = self.estimator_table.drift_table[slice_idx][1]
        drift_y = self.estimator_table.drift_table[slice_idx][2]

        if drift_x == 0 and drift_y == 0:
            return self.image_arr[slice_idx]
        else:
            return cv2.warpAffine(
                self.image_arr[slice_idx].astype(np.float32),
                np.float32([[1, 0, drift_x], [0, 1, drift_y]]),
                self.image_arr[slice_idx].shape[:2][::-1],
            ).astype(self.image_arr.dtype)

    # @timeit
    def apply_correction(self, image_array):
        """
        Apply drift correction to the entire image array.

        Args:
            image_array (numpy.ndarray): The input image array with shape (n_slices, rows, columns).

        Returns:
            numpy.ndarray: The aligned image array with shape (n_slices, rows, columns).

        Example:
            corrected_image = self.apply_correction(image_array)

        Note:
            This is the main method of the `DriftCorrector` class, which applies drift correction to the entire image array.
        """
        if self.estimator_table.drift_table is not None:
            self.image_arr = image_array
            corrected_image = [
                self._translate_slice(i).astype(np.float32)
                for i in range(0, image_array.shape[0])
            ]
            return np.array(corrected_image)
            # return np.array(translation.translate_array(image_array.astype(np.float32),
            #                                             np.array(self.estimator_table.drift_table).astype(np.float32)))

        else:
            print("Missing drift calculation")
            return None

    def load_estimator_table(self, path=None):
        """
        Load the drift table from a file.

        Args:
            path (str, optional): The path to the drift table file (CSV or NPY format). Default is None.

        Returns:
            None

        Example:
            self.load_estimator_table("drift_table.csv")

        Note:
            This method is used to load the drift table data from a file into the `estimator_table` attribute.
        """
        if path is None:
            path = input("Please provide a filepath to the drift table")

        if path.split(".")[-1] == "npy":
            self.estimator_table.import_npy(path)
        elif path.split(".")[-1] == "csv":
            self.estimator_table.import_csv(path)


class DriftEstimator:
    """
    Drift estimator class for estimating and correcting drift in image stacks.

    This class provides methods for estimating and correcting drift in image stacks using cross-correlation.

    Args:
        None

    Attributes:
        estimator_table (DriftEstimatorTable): A table of parameters for drift estimation and correction.
        cross_correlation_map (numpy.ndarray): The cross-correlation map calculated during drift estimation.
        drift_xy (numpy.ndarray): The drift magnitude at each time point.
        drift_x (numpy.ndarray): The drift in the X direction at each time point.
        drift_y (numpy.ndarray): The drift in the Y direction at each time point.

    Methods:
        __init__(): Initialize the `DriftEstimator` object.

        estimate(image_array, **kwargs): Estimate and correct drift in an image stack.

        compute_temporal_averaging(image_arr): Compute temporal averaging of image frames.

        get_shift_from_ccm_slice(slice_index): Get the drift shift from a slice of the cross-correlation map.

        get_shifts_from_ccm(): Get the drift shifts from the entire cross-correlation map.

        create_drift_table(): Create a table of drift values.

        save_drift_table(save_as_npy=True, path=None): Save the drift table to a file.

        set_estimator_params(**kwargs): Set parameters for drift estimation and correction.

    Example:
        estimator = DriftEstimator()
        drift_params = {
            "time_averaging": 2,
            "max_expected_drift": 5,
            "shift_calc_method": "rcc",
            "ref_option": 0,
            "apply": True,
        }
        drift_corrected_image = estimator.estimate(image_stack, **drift_params)

    Note:
        The `DriftEstimator` class is used for estimating and correcting drift in image stacks.
        It provides methods for estimating drift using cross-correlation and applying drift correction to an image stack.
    """

    def __init__(self, verbose=True):
        """
        Initialize the `DriftEstimator` object.

        Args:
            None

        Returns:
            None

        Example:
            estimator = DriftEstimator()
        """
        self.verbose = verbose
        self.estimator_table = DriftEstimatorTable()
        self.cross_correlation_map = None
        self.drift_xy = None
        self.drift_x = None
        self.drift_y = None

    def estimate(self, image_array, **kwargs):
        """
        Estimate and correct drift in an image stack.

        Args:
            image_array (numpy.ndarray): The input image stack with shape [n_slices, height, width].
            **kwargs: Keyword arguments for setting drift estimation parameters.

        Returns:
            numpy.ndarray or None: The drift-corrected image stack if `apply` is True, else None.

        Example:
            drift_params = {
                "time_averaging": 2,
                "max_expected_drift": 5,
                "ref_option": 0,
                "apply": True,
            }
            drift_corrected_image = estimator.estimate(image_stack, **drift_params)

        Note:
            This method estimates and corrects drift in an image stack using specified parameters.
        """
        self.set_estimator_params(**kwargs)

        n_slices = image_array.shape[0]

        # x0, y0, x1, y1 correspond to the exact coordinates of the roi to be used or full image dims and should be a tuple
        if (
            self.estimator_table.params["use_roi"]
            and self.estimator_table.params["roi"] is not None
        ):  # crops image to roi
            print(
                self.estimator_table.params["use_roi"],
                self.estimator_table.params["roi"],
            )
            x0, y0, x1, y1 = tuple(self.estimator_table.params["roi"])
            image_arr = image_array[:, y0 : y1 + 1, x0 : x1 + 1]
        else:
            image_arr = image_array

        self.estimator_table.drift_table = estimate_drift(
            np.asarray(image_arr, dtype=np.float32),
            time_averaging=self.estimator_table.params["time_averaging"],
            max_drift=self.estimator_table.params["max_expected_drift"],
            ref_option=self.estimator_table.params["ref_option"],
        )

        if self.estimator_table.params["apply"]:
            drift_corrector = DriftCorrector()
            drift_corrector.estimator_table = self.estimator_table
            tmp = drift_corrector.apply_correction(image_array)
            return tmp
        else:
            return None

    def save_drift_table(self, save_as_npy=True, path=None):
        """
        Save the drift table to a file.

        Args:
            save_as_npy (bool, optional): Whether to save the table as a NumPy binary file. Default is True.
            path (str, optional): The file path to save the table. If not provided, a user input prompt will be used.

        Returns:
            None

        Example:
            self.save_drift_table(save_as_npy=True, path="drift_table.npy")

        Note:
            This method allows saving the drift table to a file in either NumPy binary or CSV format.
        """
        if save_as_npy:
            self.estimator_table.export_npy(path=path)
        else:
            self.estimator_table.export_csv(path=path)

    def set_estimator_params(self, **kwargs):
        """
        Set parameters for drift estimation and correction.

        Args:
            **kwargs: Keyword arguments for setting drift estimation parameters.

        Returns:
            None

        Example:
            params = {
                "time_averaging": 2,
                "max_expected_drift": 5,
                "shift_calc_method": "rcc",
                "ref_option": 0,
                "apply": True,
            }
            self.set_estimator_params(**params)

        Note:
            This method allows setting parameters for drift estimation and correction.
        """
        self.estimator_table.set_params(**kwargs)


class DriftEstimatorTable:
    """
    Class used to store DriftAlignment parameters as a dictionary.
    Parameters can be changes individually by setting the corresponding params key value to desired parameter
    """

    def __init__(self):
        self.params = {}
        self.params["lib_version"] = metadata.version("nanopyx")
        self.params["date"] = datetime.today()
        self.params["apply"] = False
        self.params["do_batch"] = False
        self.params[
            "ref_option"
        ] = 1  # 0 if it is to use first frame, 1 if uses the previous frame
        self.params["time_averaging"] = 1
        self.params["max_expected_drift"] = 0
        self.params["normalize"] = True
        self.params["shift_calc_method"] = "Max Fitting"
        self.params["use_roi"] = False
        self.params["roi"] = None
        self.params["show_ccm"] = True  # used for napari
        self.params["show_drift_plot"] = True  # used for napari
        self.params["show_drift_table"] = True  # used for napari
        self.params["comments"] = None

        self.drift_table = None

    def set_params(self, **kwargs):
        """
        Method used to set the parameters of drift alignment using keyword arguments.
        :param kwargs: same as self.params.keys()
        """
        for key, value in kwargs.items():
            self.params[key] = value

    def set_comments(self, comment_string: str):
        """
        Method used to set comments for drift alignment operation
        :param comment_string: str, comment text to be added
        """
        self.params["comments"] = comment_string

    def export_npy(self, path: str = None):
        """
        Method used to export drift table as a npy file.
        :param path: Path to export drift table as npy
        """
        tmp = []
        for key in self.params.keys():
            tmp.append((key, self.params[key]))
        tmp.append(self.drift_table)
        if path is None:
            path = (
                input("Please provide a filepath to export drift table as npy")
                + "_drift_table.npy"
            )
        else:
            path = os.path.join(path, "_drift_table.npy")

        np.save(path, np.array(tmp, dtype=object))

    def import_npy(self, path: str = None):
        """
        Method used to import drift table as a npy file.
        :param path: str, Path to drift table saved as a npy file
        """
        if path is None:
            path = input("Please provide a filepath to import drift table")

        tmp = np.load(path, allow_pickle=True)

        for i in range(tmp.shape[0] - 1):
            key, value = tmp[i]
            self.params[key] = value
        self.drift_table = tmp[tmp.shape[0] - 1]

    def export_csv(self, path: str = None):
        """
        Method used to export drift table as a csv file.
        :param path: str, Path to export drift table as csv
        """
        if path is None:
            path = (
                input("Please provide a filepath to export drift table as csv")
                + "_drift_table.csv"
            )
        else:
            path = os.path.join(path, "_drift_table.csv")

        txt = ""
        for key in self.params.keys():
            txt += key + ";" + str(self.params[key]) + "\n"
        txt += "Drift Table\n"
        txt += "XY;X;Y\n"
        for i in range(self.drift_table.shape[0]):
            txt += (
                str(self.drift_table[i][0])
                + ";"
                + str(self.drift_table[i][1])
                + ";"
                + str(self.drift_table[i][2])
                + "\n"
            )

        open(path, "w").writelines(txt)

    def import_csv(self, path: str = None):
        """
        Method used to import drift table from a csv file
        :param path: str, path to import drift table as csv
        """
        if path is None:
            path = input("Please provide a filepath to import drift table")

        tmp = open(path).readlines()

        count = 0
        for line in tmp:
            if line == "Drift Table\n":
                break
            else:
                count += 1
            param_split = line.split(";")
            key = param_split[0]
            value = param_split[1].split("\n")[0]
            if value == "True":
                value = True
            elif value == "False":
                value = False
            elif value == "None":
                value = None
            self.params[key] = value

        if self.params["roi"] is not None:
            roi_str_list = self.params["roi"][1:-1].split(", ")
            self.params["roi"] = tuple(int(coord) for coord in roi_str_list)

        drift_table = []

        for row in tmp[count + 2 :]:
            row_split = row.split(";")
            drift_xy = float(row_split[0])
            drift_x = float(row_split[1])
            drift_y = float(row_split[2])
            drift_table.append([drift_xy, drift_x, drift_y])

        self.drift_table = np.array(drift_table)


def estimate_drift(image, time_averaging=2, max_drift=5, ref_option=0):

    # Ensure image is an even square
    n_slices = image.shape[0]
    n_rows = image.shape[1]
    n_cols = image.shape[2]

    # Find the minimum dimension
    min_dim = min(n_rows, n_cols)
    # Make it even
    if min_dim % 2 != 0:
        min_dim = min_dim - 1

    # Crop to square
    row_start = (
        n_rows - min_dim
    ) // 2  # how much extra rows and cols we have, divided by 2 to crop equally on both sides
    col_start = (n_cols - min_dim) // 2
    image = image[
        :, row_start : row_start + min_dim, col_start : col_start + min_dim
    ]

    # get image dimensions again, should already be an even square
    n_slices = image.shape[0]
    n_rows = image.shape[1]
    n_cols = image.shape[2]

    # ensures time averaging has an acceptable value
    if time_averaging < 1:
        time_averaging = 1
    elif time_averaging > (n_slices // 2):
        time_averaging = n_slices // 2

    n_blocks = n_slices // time_averaging

    averaged = np.empty((n_blocks, n_rows, n_cols), dtype=np.float32)

    if time_averaging == 1:
        averaged = image
    else:
        for idx in range(n_blocks):
            averaged[idx, :, :] = np.mean(
                image[idx * time_averaging : (idx + 1) * time_averaging, :, :],
                axis=0,
            )

    if (
        max_drift > 0
        and max_drift * 2 + 1 < n_rows
        and max_drift * 2 + 1 < n_cols
    ):
        row_start = int(n_rows / 2 - max_drift)
        col_start = int(n_cols / 2 - max_drift)
        ccm = _calculate_ccm(averaged, ref_option)[
            :,
            row_start : row_start + (max_drift * 2),
            col_start : col_start + (max_drift * 2),
        ]
    else:
        ccm = _calculate_ccm(averaged, ref_option)

    drift_table = np.zeros((n_blocks, 2), dtype=np.float32)

    output = np.zeros((image.shape[0], 3), dtype=np.float32)

    bias_row = 0.0
    bias_col = 0.0

    for i in range(n_blocks):

        optimizer = GetMaxOptimizer(
            np.ascontiguousarray(ccm[i], dtype=np.float32)
        )
        shift_y, shift_x = optimizer.get_max()

        drift_table[i, 0] = round((ccm.shape[1] / 2) - shift_y - 0.5, 3)
        drift_table[i, 1] = round((ccm.shape[2] / 2) - shift_x - 0.5, 3)

        if i == 0:
            bias_row = drift_table[i, 0]
            bias_col = drift_table[i, 1]
        drift_table[i, 0] = drift_table[i, 0] - bias_row
        drift_table[i, 1] = drift_table[i, 1] - bias_col

        if ref_option == 1 and i > 0:
            drift_table[i, 0] = drift_table[i, 0] + drift_table[i - 1, 0]
            drift_table[i, 1] = drift_table[i, 1] + drift_table[i - 1, 1]

    if time_averaging > 1:
        lin = np.linspace(
            1,
            image.shape[0],
            num=drift_table.shape[0],
            endpoint=True,
            dtype=int,
        )
        x_interpolator = interp1d(
            lin, np.array(drift_table[:, 1]), kind="cubic"
        )
        y_interpolator = interp1d(
            lin, np.array(drift_table[:, 0]), kind="cubic"
        )

        drift_x = np.asarray(
            x_interpolator(range(1, image.shape[0] + 1)), dtype=np.float32
        ).reshape(n_slices)
        output[:, 1] = drift_x
        drift_y = np.asarray(
            y_interpolator(range(1, image.shape[0] + 1)), dtype=np.float32
        ).reshape(n_slices)
        output[:, 2] = drift_y

    else:
        output[:, 1] = drift_table[:, 1]  # switch order of rows and cols
        output[:, 2] = drift_table[:, 0]  # switch order of rows and cols

    for s in range(n_slices):
        output[s, 0] = sqrt(
            (output[s, 1] * output[s, 1]) + (output[s, 2] * output[s, 2])
        )

    return np.asarray(output).astype(np.float32)


def _calculate_ccm(img_stack, ref):

    stack_w = img_stack.shape[2]
    stack_h = img_stack.shape[1]
    stack_n = img_stack.shape[0]
    ccm = np.empty((stack_n, stack_h, stack_w), dtype=np.float32)

    for i in range(stack_n):
        if ref == 0:
            img_ref = img_stack[0]
        else:
            _n = max(0, i - 1)
            img_ref = img_stack[_n]
        ccm[i] = _calculate_slice_ccm(img_ref, img_stack[i])

    return ccm


def _calculate_slice_ccm(img_ref, img_slice):

    ccm_slice = sp.fft.fftshift(
        sp.fft.ifft2(sp.fft.fft2(img_ref) * sp.fft.fft2(img_slice).conj())
    ).real.astype(np.float32)
    ccm_slice = ccm_slice[::-1, ::-1]

    _normalize_ccm(img_ref, img_slice, ccm_slice)

    return ccm_slice[0 : ccm_slice.shape[0], 0 : ccm_slice.shape[1]]


def _normalize_ccm(img_ref, img_slice, ccm_slice):
    """
    Function used to normalize the cross correlation matrix.

    The code above does the following:
    1. Find the maximum and minimum values of the cross-correlation matrix
    2. Calculate the maximum and minimum PPMCC
    3. Normalize the matrix values to the PPMCC values
    """

    w = ccm_slice.shape[1]
    h = ccm_slice.shape[0]

    # print(np.min(ccm_slice), np.max(ccm_slice))

    min_value = np.min(ccm_slice)
    max_value = np.max(ccm_slice)
    x_max = 0
    y_max = 0
    x_min = 0
    y_min = 0

    coords = np.unravel_index(np.argmax(ccm_slice), (w, h))
    y_max = coords[0]
    x_max = coords[1]

    coords = np.unravel_index(np.argmin(ccm_slice), (w, h))
    y_min = coords[0]
    x_min = coords[1]

    shift_x_max = x_max - w // 2
    shift_y_max = y_max - h // 2
    shift_x_min = x_min - w // 2
    shift_y_min = y_min - h // 2

    max_ppmcc = _calculate_ppmcc(img_ref, img_slice, shift_x_max, shift_y_max)
    min_ppmcc = _calculate_ppmcc(img_ref, img_slice, shift_x_min, shift_y_min)

    delta_v = max_value - min_value
    value = 0.0
    delta_ppmcc = max_ppmcc - min_ppmcc

    for j in range(h):
        for i in range(w):
            value = (ccm_slice[j, i] - min_value) / delta_v
            value = value * delta_ppmcc + min_ppmcc
            ccm_slice[j, i] = value


def _calculate_ppmcc(im1, im2, shift_x, shift_y):
    w = im1.shape[1]
    h = im1.shape[0]
    new_w = int(w - fabs(shift_x))
    new_h = int(h - fabs(shift_y))

    x0 = max(0, -shift_x)
    y0 = max(0, -shift_y)
    x1 = x0 + shift_x
    y1 = y0 + shift_y

    return _pearson_correlation(
        im1[y0 : y0 + new_h, x0 : x0 + new_w],
        im2[y1 : y1 + new_h, x1 : x1 + new_w],
    )


def _pearson_correlation(im1, im2):

    w = im1.shape[1]
    h = im1.shape[0]
    wh = w * h

    mean_im1 = 0.0
    mean_im2 = 0.0
    sum_im12 = 0.0
    sum_im11 = 0.0
    sum_im22 = 0.0

    for j in range(h):
        for i in range(w):
            mean_im1 += im1[j, i]
            mean_im2 += im2[j, i]

    mean_im1 /= wh
    mean_im2 /= wh

    for j in range(h):
        for i in range(w):
            d_im1 = im1[j, i] - mean_im1
            d_im2 = im2[j, i] - mean_im2
            sum_im12 += d_im1 * d_im2
            sum_im11 += d_im1 * d_im1
            sum_im22 += d_im2 * d_im2

    if sum_im11 == 0 or sum_im22 == 0:
        return 0
    else:
        return sum_im12 / (sum_im11 * sum_im22) ** 0.5


class GetMaxOptimizer:
    """
    Class GetMaxOptimizer, used to extract the maximum value from a cross correlation matrix with subpixel precision.
    """

    def __init__(self, slice_ccm) -> None:
        """
        Creates an instance of GetMaxOptimizer.
        :param slice_ccm: numpy array with shape (y, x); ccm from which to extract the maximum value with subpixel
        precision.
        """
        self.slice_ccm = slice_ccm
        self.interpolator = _interpolate(
            (np.arange(slice_ccm.shape[0]), np.arange(slice_ccm.shape[1])),
            slice_ccm,
            bounds_error=True,
        )

    def get_interpolated_px_value(self, coords):
        """
        Method to be used for calculating the interpolated values of cross correlation matrices.
        :param coords: tuple of coordinates.
        :return: float; value of cross correlation matrix at given coordinates.
        For minimizer reasons -> negatives values become positive and positive become negative.
        """
        return -self.interpolator([coords[0], coords[1]])[0]

    def get_max(self):
        """
        Method used to calculate the maximum value and corresponding coordinates of a ccm. Uses a minimizer approach.
        :return: tuple; coordinates of maximum value of ccm with subpixel precision
        """
        y_max, x_max = np.unravel_index(
            self.slice_ccm.argmax(), self.slice_ccm.shape
        )
        minimizer = minimize(
            self.get_interpolated_px_value,
            (y_max, x_max),
            method="Nelder-Mead",
            options={"maxiter": 1000},
        )
        return minimizer.x
