"""Module used to create the report of the cell identification"""
import os
from decimal import Decimal

import numpy as np
import pandas as pd
from skimage.io import imsave
from skimage.util import img_as_ubyte

from .cellprocessing import stats_format


class ReportManager:
    """
    Generate HTML and CSV reports for analyzed cells.

    Parameters
    ----------
    parameters : dict
        Analysis parameters dictionary.
    properties : dict
        Per-cell properties dictionary (e.g., label, frame, Area,
        etc.).
    allcells : list[numpy.ndarray]
        List of per-cell montage images for visualization.

    Attributes
    ----------
    cells : list[numpy.ndarray]
        Padded per-cell images.
    properties : dict
        Properties passed at initialization.
    params : dict
        Parameters passed at initialization.
    keys : list[tuple[str, int]]
        Property labels and display precision from `stats_format`.
    cell_data_filename : str or None
        Base path of the generated report.
    """

    def __init__(self, parameters, properties, allcells):
        """Initialize report content and pad cell images.

        Parameters
        ----------
        parameters : dict
            Analysis parameters.
        properties : dict
            Per-cell properties.
        allcells : list[numpy.ndarray]
            List of per-cell montage images.

        Notes
        -----
        If `allcells` is empty, report metadata is still initialized and
        CSV export remains available.
        """

        self.cells = allcells

        if len(self.cells) > 0:
            self.max_shape = np.max(
                [cell.shape for cell in self.cells], axis=0
            )
        else:
            self.max_shape = (1, 1)

        self.properties = properties
        self.params = parameters
        self.keys = stats_format(parameters)

        self.cell_data_filename = None

    def html_report(self, filename):
        """Write an HTML report with per-cell details or timelapse summary.

        Parameters
        ----------
        filename : str
            Output directory path for the HTML report and images.

        Notes
        -----
        For timelapse runs with many cells, generates a lightweight summary.
        For 2D runs, generates detailed per-cell table.
        """
        cells = self.cells
        is_timelapse = self.params.get("include_frame", False)
        num_cells = len(self.properties["label"])

        # For timelapse, cell images were already written to disk per-frame.
        # Discover them so the HTML table can reference them without re-loading.
        images_dir = filename + "/_images"
        if is_timelapse and len(cells) == 0 and os.path.isdir(images_dir):
            pre_saved_count = len(
                [
                    f
                    for f in os.listdir(images_dir)
                    if f.startswith("cell_") and f.endswith(".png")
                ]
            )
        else:
            pre_saved_count = 0

        HTML_HEADER = """<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN"
                        "http://www.w3.org/TR/html4/strict.dtd">
                    <html lang="en">
                      <head>
                        <meta http-equiv="content-type" content="text/html; charset=utf-8">
                        <title>mAIcrobe Report</title>
                        <link rel="stylesheet" type="text/css" href="style.css">
                        <script type="text/javascript" src="script.js"></script>
                      </head>
                      <body>\n"""

        report = [HTML_HEADER]

        report.append(
            "\n<h1>mAIcrobe Report - <a href='https://github.com/HenriquesLab/mAIcrobe/blob/main/docs/user-guide/getting-started.md' target='_blank'>mAIcrobe</a></h1>"
        )
        report.append("\n<h3>Total cells: " + str(num_cells) + "</h3>")

        if is_timelapse:
            report.append(
                "\n<p>Full cell data available in <strong>Analysis.csv</strong></p>"
            )

        if self.params["classify_cell_cycle"]:
            _, pcounts = np.unique(
                list(self.properties["Cell Cycle Phase"]) + [1, 2, 3],
                return_counts=True,
            )
            report.append(
                "\n<h3>Phase 1 cells: " + str(pcounts[0] - 1) + "</h3>"
            )
            report.append(
                "\n<h3>Phase 2 cells: " + str(pcounts[1] - 1) + "</h3>"
            )
            report.append(
                "\n<h3>Phase 3 cells: " + str(pcounts[2] - 1) + "</h3>"
            )

        n_display = pre_saved_count if pre_saved_count > 0 else len(cells)
        if n_display > 0:
            # Force a dedicated timelapse frame column in HTML output.
            display_keys = [k for k in self.keys if k[0] != "frame"]
            header = "<table>\n"
            if is_timelapse and "frame" in self.properties:
                header += "<th>Frame</th>"
            header += "<th>Cell ID</th><th>Images"
            for k in display_keys:
                label, digits = k
                header = header + "</th><th>" + label
            header += "</th>\n"
            selects = ["\n<h1>Selected cells:</h1>\n" + header + "\n"]

            print("Total Cells: " + str(n_display))

            for idx in range(n_display):
                cell_filename = f"cell_{idx}.png"
                cell_path = filename + "/_images" + os.sep + cell_filename

                if pre_saved_count == 0:
                    # In-memory (2D run): write image now.
                    cell = cells[idx]
                    if not os.path.exists(cell_path):
                        imsave(
                            cell_path,
                            img_as_ubyte(cell),
                            check_contrast=False,
                        )

                lin = "<tr>"
                if is_timelapse and "frame" in self.properties:
                    # Convert to 1-based indexing for easier interpretation.
                    lin += "<td>" + str(int(self.properties["frame"][idx]) + 1)
                lin += (
                    "</td><td>"
                    + str(self.properties["label"][idx])
                    + '</td><td><img src="./_images/'
                    + cell_filename
                    + '" alt="pic" style="max-width: 240px; height: auto;"></td>'
                )

                for stat in display_keys:
                    lbl, digits = stat
                    number = ("{0:." + str(digits) + "f}").format(
                        self.properties[lbl][idx]
                    )
                    number = str(Decimal(number))
                    number = (
                        number.rstrip("0").rstrip(".")
                        if "." in number
                        else number
                    )
                    lin = lin + "</td><td>" + number

                lin += "</td></tr>\n"
                selects.append(lin)

            if len(selects) > 1:
                report.extend(selects)
                report.append("</table>\n")

            report.append("</body>\n</html>")

        open(
            filename + "/html_report_" + ".html", "w", encoding="utf-16"
        ).writelines(report)

    def check_filename(self, filename):
        """Ensure a unique report directory by appending an index.

        Parameters
        ----------
        filename : str
            Base filename (without extension).

        Returns
        -------
        str
            Available filename not colliding with existing path.
        """
        if os.path.exists(filename):
            tmp = ""
            split_path = filename.split("_")
            tmp = "_".join(split_path[: len(split_path) - 1])
            tmp += "_" + str(int(split_path[-1]) + 1)
            return self.check_filename(tmp)

        else:
            return filename

    def prepare_report_dir(self, path, report_id=None):
        """Create the report output directory early for streaming image writes.

        Call this before the analysis loop so cell images can be written to
        disk frame-by-frame without accumulating them in memory. After the
        loop, call ``generate_report()`` with the same arguments to write the
        HTML and CSV; the directory will be reused.

        Parameters
        ----------
        path : str
            Output directory.
        report_id : str or None, optional
            Optional report identifier appended to directory name.

        Returns
        -------
        str
            Absolute path to the ``_images`` subdirectory where cell images
            should be written.
        """
        if report_id is None:
            filename = path + "/Report_1"
        else:
            filename = path + "/Report_" + report_id + "_1"
        filename = self.check_filename(filename)
        self.cell_data_filename = filename
        images_dir = filename + "/_images"
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)
        return images_dir

    def generate_report(self, path, report_id=None):
        """Generate HTML report and CSV with properties.

        Parameters
        ----------
        path : str
            Output directory.
        report_id : str or None, optional
            Optional report identifier appended to directory name.

        Side Effects
        ------------
        Creates directory structure, writes HTML and `Analysis.csv`, and
        sets `self.cell_data_filename`.
        """
        # If prepare_report_dir() was already called (streaming mode), reuse
        # the pre-created directory instead of creating a new one.
        if self.cell_data_filename is not None:
            filename = self.cell_data_filename
        elif report_id is None:
            filename = path + "/Report_1"
            filename = self.check_filename(filename)
            self.cell_data_filename = filename
            if not os.path.exists(filename + "/_images"):
                os.makedirs(filename + "/_images")
        else:
            filename = path + "/Report_" + report_id + "_1"
            filename = self.check_filename(filename)
            self.cell_data_filename = filename
            if not os.path.exists(filename + "/_images"):
                os.makedirs(filename + "/_images")

        self.html_report(filename)

        df = pd.DataFrame(self.properties)
        df.to_csv(os.path.join(filename, f"Analysis.csv"))

        # TODO SAVE PARS
