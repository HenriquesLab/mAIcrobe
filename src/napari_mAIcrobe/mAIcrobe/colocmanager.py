import os

from scipy.stats import pearsonr


class ColocManager:
    """Calculate and export per-cell Pearson correlation coefficients
    between two fluorescence metrics.

    Attributes
    ----------
    report : dict
        Mapping of cell label (str) to computed metrics
    """

    def __init__(self):
        """Initialize an empty report dict."""
        self.report = {}

    def save_report(self, reportID, sept=False):
        """Write a CSV report with the computed Pearson metrics that are
        stored in the report attribute.

        Parameters
        ----------
        reportID : str
            Base output directory for the report CSV.
        sept : bool, optional
            Include septum-related metrics if available, by default
            False.
        """

        sorted_keys = sorted(self.report.keys())

        header = ["Whole Cell", "Membrane", "Cytoplasm"]
        if sept:
            header.extend(["Septum", "MembSept"])

        results = "Cell ID;"
        results += ";".join(header)
        results += ";\n"

        for key in sorted_keys:
            results += key + ";"
            for measurement in header:
                results += str(self.report[key][measurement]) + ";"

            results += "\n"

        open(reportID + os.sep + "_pcc_report.csv", "w").writelines(results)

    def pearsons_score(self, channel_1, channel_2, mask):
        """Compute Pearson correlation within a masked region.

        Zeros are removed from both channels before computation.

        Parameters
        ----------
        channel_1 : numpy.ndarray
            First channel crop.
        channel_2 : numpy.ndarray
            Second channel crop.
        mask : numpy.ndarray
            Binary mask selecting pixels of interest.

        Returns
        -------
        tuple[float, float]
            (r, pvalue) from `scipy.stats.pearsonr`.
        """

        filtered_1 = (channel_1 * mask).flatten()
        filtered_1 = filtered_1[
            filtered_1 > 0.0
        ]  # removes 0s from entering pcc calculation
        filtered_2 = (channel_2 * mask).flatten()
        filtered_2 = filtered_2[
            filtered_2 > 0.0
        ]  # removes 0s from entering pcc calculation

        return pearsonr(filtered_1, filtered_2)

    def computes_cell_pcc(self, fluor_image, optional_image, cell, parameters):
        """Compute and store Pearson metrics for a single cell.

        Parameters
        ----------
        fluor_image : numpy.ndarray
            Full-field fluorescence image (channel 1).
        optional_image : numpy.ndarray
            Full-field optional image (channel 2).
        cell : napari_mAIcrobe.mAIcrobe.cells.Cell
            Cell object with region masks and bounding box.
        parameters : dict
            Analysis parameters including `find_septum`.
        """

        key = str(cell.label)
        self.report[key] = {}
        x0, y0, x1, y1 = cell.box

        fluor_box = fluor_image[x0 : x1 + 1, y0 : y1 + 1]
        optional_box = optional_image[x0 : x1 + 1, y0 : y1 + 1]

        try:
            self.report[key]["Channel 1"] = fluor_box
            self.report[key]["Channel 2"] = optional_box

            self.report[key]["Whole Cell"] = self.pearsons_score(
                fluor_box, optional_box, cell.cell_mask
            )[0]
            self.report[key]["Membrane"] = self.pearsons_score(
                fluor_box, optional_box, cell.perim_mask
            )[0]
            self.report[key]["Cytoplasm"] = self.pearsons_score(
                fluor_box, optional_box, cell.cyto_mask
            )[0]

            if parameters["find_septum"]:
                self.report[key]["Septum"] = self.pearsons_score(
                    fluor_box, optional_box, cell.sept_mask
                )[0]
                self.report[key]["MembSept"] = self.pearsons_score(
                    fluor_box, optional_box, cell.membsept_mask
                )[0]
        except ValueError:
            del self.report[key]

    def compute_pcc(
        self, fluor_image, optional_image, cells, parameters, reportID
    ):
        """Compute Pearson metrics for all cells and save a report.

        DEPRECATED: use `computes_cell_pcc` for single cells instead.

        Parameters
        ----------
        fluor_image : numpy.ndarray
            Full-field fluorescence image (channel 1).
        optional_image : numpy.ndarray
            Full-field optional image (channel 2).
        cells : list[napari_mAIcrobe.mAIcrobe.cells.Cell]
            Iterable of Cell objects.
        parameters : dict
            Analysis parameters including `find_septum`.
        reportID : str
            Base output directory for the report CSV.
        """
        self.report = {}

        for cell in cells:
            key = str(cell.label)
            self.report[key] = {}

            x0, y0, x1, y1 = cell.box

            fluor_box = fluor_image[x0 : x1 + 1, y0 : y1 + 1]
            optional_box = optional_image[x0 : x1 + 1, y0 : y1 + 1]

            try:
                self.report[key]["Channel 1"] = fluor_box
                self.report[key]["Channel 2"] = optional_box

                self.report[key]["Whole Cell"] = self.pearsons_score(
                    fluor_box, optional_box, cell.cell_mask
                )[0]
                self.report[key]["Membrane"] = self.pearsons_score(
                    fluor_box, optional_box, cell.perim_mask
                )[0]
                self.report[key]["Cytoplasm"] = self.pearsons_score(
                    fluor_box, optional_box, cell.cyto_mask
                )[0]

                if parameters["find_septum"]:
                    self.report[key]["Septum"] = self.pearsons_score(
                        fluor_box, optional_box, cell.sept_mask
                    )[0]
                    self.report[key]["MembSept"] = self.pearsons_score(
                        fluor_box, optional_box, cell.membsept_mask
                    )[0]
            except ValueError:
                del self.report[key]

        self.save_report(reportID, sept=parameters["find_septum"])
