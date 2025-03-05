import os
import numpy as np
from scipy.stats import pearsonr

class ColocManager:

    def __init__(self):
        self.report = {}

    def save_report(self, reportID, sept=False):

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

        filtered_1 = (channel_1 * mask).flatten()
        filtered_1 = filtered_1[filtered_1 > 0.0] # removes 0s from entering pcc calculation
        filtered_2 = (channel_2 * mask).flatten()
        filtered_2 = filtered_2[filtered_2 > 0.0] # removes 0s from entering pcc calculation

        return pearsonr(filtered_1, filtered_2)

    def compute_pcc(self, fluor_image, optional_image, cells, parameters, reportID):
        self.report = {}

        for cell in cells:
            key = str(cell.label)
            self.report[key] = {}

            x0, y0, x1, y1 = cell.box

            fluor_box = fluor_image[x0:x1+1, y0:y1+1]
            optional_box = optional_image[x0:x1+1, y0:y1+1]

            try:
                self.report[key]["Channel 1"] = fluor_box
                self.report[key]["Channel 2"] = optional_box

                self.report[key]["Whole Cell"] = self.pearsons_score(fluor_box, optional_box, cell.cell_mask)[0]
                self.report[key]["Membrane"] = self.pearsons_score(fluor_box, optional_box, cell.perim_mask)[0]
                self.report[key]["Cytoplasm"] = self.pearsons_score(fluor_box, optional_box, cell.cyto_mask)[0]

                if parameters['find_septum']:
                    self.report[key]["Septum"] = self.pearsons_score(fluor_box, optional_box, cell.sept_mask)[0]
                    self.report[key]["MembSept"] = self.pearsons_score(fluor_box, optional_box, cell.membsept_mask)[0]
            except ValueError:
                del self.report[key]

        self.save_report(reportID, sept=parameters['find_septum'])

