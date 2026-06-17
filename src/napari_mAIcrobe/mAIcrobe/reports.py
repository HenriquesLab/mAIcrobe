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
        """Write an HTML report composing cell thumbnails and stats.

        Parameters
        ----------
        filename : str
            Output directory path for the HTML report and images.

        Notes
        -----
        HTML content is written only when at least one cell montage is
        available.
        """
        cells = self.cells
        """generates an html report with the all the cell stats from the
        selected cells"""

        HTML_HEADER = """<!DOCTYPE html PUBLIC "-//W3C//DTD HTML 4.01//EN"
                        "http://www.w3.org/TR/html4/strict.dtd">
                    <html lang="en">
                      <head>
                        <meta http-equiv="content-type" content="text/html; charset=utf-8">
                        <title>title</title>
                        <link rel="stylesheet" type="text/css" href="style.css">
                        <script type="text/javascript" src="script.js"></script>
                      </head>
                      <body>\n"""

        report = [HTML_HEADER]

        if len(cells) > 0:
            header = "<table>\n<th>Cell ID</th><th>Images"
            for k in self.keys:
                label, digits = k
                header = header + "</th><th>" + label
            header += "</th>\n"
            selects = ["\n<h1>Selected cells:</h1>\n" + header + "\n"]

            print("Total Cells: " + str(len(cells)))

            for idx, cell in enumerate(cells):
                cell_filename = f"cell_{idx}.png"
                imsave(
                    filename + "/_images" + os.sep + cell_filename,
                    img_as_ubyte(cell),
                    check_contrast=False,
                )

                lin = (
                    "<tr><td>"
                    + str(self.properties["label"][idx])
                    + '</td><td><img src="./_images/'
                    + cell_filename
                    + '" alt="pic" style="max-width: 240px; height: auto;"></td>'
                )

                for stat in self.keys:
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

            report.append(
                "\n<h1>mAIcrobe Report - <a href='https://github.com/HenriquesLab/mAIcrobe/blob/main/docs/user-guide/getting-started.md' target='_blank'> https://github.com/HenriquesLab/mAIcrobe/blob/main/docs/user-guide/getting-started.md</a></h1>"
            )

            report.append(
                "\n<h3>Total cells: "
                + str(len(self.properties["label"]))
                + "</h3>"
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
        if report_id is None:
            filename = path + "/Report_1"
            filename = self.check_filename(filename)
            self.cell_data_filename = filename

            if not os.path.exists(filename + "/_images"):
                os.makedirs(filename + "/_images")
                # os.makedirs(filename + "/_images/membrane")
                # os.makedirs(filename + "/_images/dna")
                # os.makedirs(filename + "/_images/crops")
        else:
            filename = path + "/Report_" + report_id + "_1"
            filename = self.check_filename(filename)
            self.cell_data_filename = filename

            if not os.path.exists(filename + "/_images"):
                os.makedirs(filename + "/_images")
                # os.makedirs(filename + "/_images/membrane")
                # os.makedirs(filename + "/_images/dna")
                # os.makedirs(filename + "/_images/crops")

        self.html_report(filename)

        df = pd.DataFrame(self.properties)
        df.to_csv(os.path.join(filename, f"Analysis.csv"))

        # TODO SAVE PARS
