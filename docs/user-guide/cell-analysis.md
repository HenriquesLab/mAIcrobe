# Cell Analysis Guide

This comprehensive guide covers all aspects of automated cell analysis in mAIcrobe, from basic morphometry to cell classification.

## 🎯 Overview

The **"Compute cells"** widget provides:

- 📏 **Morphological analysis** - Shape and size measurements
- 💡 **Intensity analysis** - Fluorescence quantification
- 🧠 **Cell classification** - Deep learning single-cell classification. Custom models or pre-trained.
- 🔗 **Colocalization analysis** - PCC analysis across channels
- 📊 **Report generation** - HTML and CSV output

---

## 🔬 Analysis Workflow

### Step 1: Prepare Your Data

Before running cell analysis, ensure you have:

1. 🏷️ **Segmented cells** - Labels layer either loaded externally or computed from the `compute_label` widget. (See [segmentation guide](segmentation-guide.md))
2. 🖼️ **Image channels** - Fluorescence channels to be analyzed (as needed)

### Step 2: Configure Analysis Parameters

#### ⚙️ Settings

**Image Selection:**
- **Label Image**: Segmentation results (required)
- **Fluorescence 1**: Fluorescence channel (e.g., Membrane stain)
- **Fluorescence 2**: Optional, in case it's not needed set to the same channel as Fluorescence 1 (e.g., Nuclear/nucleoid staining)
- **Pixel size**: Physical pixel size (e.g., 0.065 μm/pixel) (optional)

**Subcellular Segmentation:**
- **Inner mask thickness**: Membrane thickness for cytoplasmic measurements (default: 4)
- **Find septum**: Whether to detect division septa
- **Septum algorithm**: "Isodata" or "Box" thresholding
- **Find open septum**: Detect incomplete septa

**Fluorescence Analysis:**
- **Baseline margin**: Background region size (default: 30)

**Cell Cycle Classification:**

Check [cell classification guide](cell-classification.md) for details.

- **Classify cell cycle**: Enable single cell classification
- **Model**: Choose appropriate pre-trained model or choose a custom model
    - **Pre-trained models:**
        - S.aureus DNA+Membrane Epi
        - S.aureus DNA+Membrane SIM
        - S.aureus DNA Epi
        - S.aureus DNA SIM
        - S.aureus Membrane Epi
        - S.aureus Membrane SIM
        - E.coli DNA+Membrane AB phenotyping
    - **Custom**: Use your own trained model
- **Custom model path**: If you selected custom, provide the path to your own model file (.keras)
- **Custom model input**: Specify input type for custom model (Membrane, DNA, or Membrane+DNA)
- **Custom model max size**: Maximum cell size for custom model (default: 50 pixels)

**Other Options:**
- **Compute Colocalization**: Multi-channel colocalization analysis via PCC's. Only works if two different fluorescence channels are provided.
- **Generate Report**: Whether to create a report folder with an HTML report and CSV.
- **Report path**: Directory to save the report folder.
- **Compute Heatmap**: Spatial analysis visualization of a fluorescence channel. Corresponds to an average intensity heatmap over all cells aligned according to their major axis. Outputs a new Image layer.

### Timelapse datasets (2D+t)

mAIcrobe also supports timelapse analysis for `(T, Y, X)` datasets.

- Labels can be generated over all timepoints
- Optional reassignment can stabilize label IDs across consecutive frames
- Optional registration can correct drift before segmentation
- Cell properties include a `frame` column in timelapse workflows

For a complete walkthrough, see the [Timelapse Analysis Guide](timelapse-analysis.md).

---

## 📊 Outputs

### Morphological Measurements

mAIcrobe computes shape and size parameters using scikit-image [regionprops](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops). Furthermore it adds custom measurements relevant for bacterial cells such as subcellular segmentation (membrane, cytoplasm, and septum when requested).

#### Basic Shape Parameters

**Area and Size:**
- **Area**: Cell area (in pixels or μm² depending on the pixel size setting)
- **Perimeter**: Cell boundary length

**Shape Descriptors:**
- **Eccentricity**: Ellipse eccentricity (0=circle, 1=line)
    -  `ecc = sqrt(1 - (b²/a²))` where `a` is the semi-major axis and `b` is the semi-minor axis

### Intensity Analysis

Quantify fluorescence signals in subcellular compartments.

**Basic Statistics:**
- **Baseline intensity**: Local background signal in Fluorescence 1 channel.

**IMPORTANT**: This value is subtracted from all other intensity measurements

- **Cell Median intensity**: Median fluorescence within the entire cell in Fluorescence 1 channel
- **Membrane Median intensity**: Median fluorescence in the membrane region in Fluorescence 1 channel
- **Cytoplasm Median intensity**: Median fluorescence in the cytoplasmic region in Fluorescence 1 channel
- **Septum Median intensity**: Median fluorescence in the septum region (if detected and enabled otherwise 0) in Fluorescence 1 channel
- **Fluorescence Ratios 100%, 75%, 25%, 10% percentiles**: Ratios between septum and membrane (if septum detected and enabled otherwise 0) in Fluorescence 1 channel
- **DNA Ratio**: Relative DNA content compared to baseline background fluorescence (if Fluorescence 1 channel provided, otherwise 0)

### 🧠 Cell Classification

Use deep learning models to automatically classify cells. Classification is given in the form of a integer per cell corresponding to the predicted class.

mAIcrobe includes pretrained models or you can use your own custom trained models. Check the [Cell Classification Guide](cell-classification.md) for details on the available models. To train your own model check the [training notebook](../../notebooks/napari_mAIcrobe_cellcyclemodel.ipynb) and [Generate Training Data tutorial](../tutorials/generate_trainingdata.md) for details.


### 📈 Colocalization Analysis

Quantify spatial relationships between two fluorescence channels.

- **Pearson correlation coefficient**

---

## 🔍 Interactive Filtering

Use the **"Filter cells"** widget for real-time quality control:

1. 🏷️ **Select Labels layer** after compute_cells
2. ➕ **Add filters** for any measured feature
3. 👁️ **Preview filtered population** in real-time
4. ✅ **Use the filtered results for further analysis** The new layer "Filtered cells" contains only the selected cells.

**Tips**
- Hide the original Labels layer to visualize only the filtered results.
- Combine multiple filters to refine your selection. For example, filter by area to remove badly segmented cells and by cell classification to focus on a specific cell cycle phase.
- Compute septum option works best when combined with filtering by classification by cell cycle phase to focus on cells with actual division septa.


---

## 📚 Further Reading

- **[Cell Classification](cell-classification.md)** - Detailed cell classification guide
- **[Timelapse Analysis](timelapse-analysis.md)** - 2D+t workflows with relabeling
- **[Batch analysis workflow](../user-guide/batch-analysis-workflow.md)** - Multi-FoV processing
- **[API Reference](../api/api-reference.md)** - Programmatic analysis
- **[Basic workflow](../tutorials/basic-workflow.md)** - Step-by-step examples

### 🔗 Technical References
- **scikit-image regionprops**: [Documentation](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
- **napari-skimage-regionprops plugin**: [GitHub](https://github.com/haesleinhuepf/napari-skimage-regionprops) - mAIcrobe internally uses this plugin to add regionprops tables to the GUI.

---

**Next:** Explore deep learning cell classification in the **[Cell Classification Guide](cell-classification.md)** 🧠
