# Cell Analysis Guide

This comprehensive guide covers all aspects of automated cell analysis in napari-mAIcrobe, from basic morphometry to cell classification.

## 🎯 Overview

The **"Compute cells"** widget provides:

- 📏 **Morphological analysis** - Shape and size measurements
- 💡 **Intensity analysis** - Fluorescence quantification
- 🧠 **Cell classification** - Deep learning classification with default models for cell cycle phase determination in *S. aureus*
- 🔗 **Colocalization analysis** - Multi-channel correlation
- 📊 **Report generation** - Professional HTML output

---

## 🔬 Analysis Workflow

### Step 1: Prepare Your Data

Before running cell analysis, ensure you have:

1. 🏷️ **Segmented cells** - Labels layer from Segmentation step
2. 🖼️ **Image channels** - Phase contrast, membrane, DNA (as needed)

### Step 2: Configure Analysis Parameters

#### ⚙️ Essential Settings

**Image Selection:**
- **Label Image**: Segmentation results (required)
- **Membrane Image**: Fluorescence channel
- **DNA Image**: Nuclear/nucleoid staining
- **Pixel size**: Physical pixel size (e.g., 0.065 μm/pixel) (optional)

**Subcellular Segmentation:**
- **Inner mask thickness**: Membrane thickness for cytoplasmic measurements (default: 4)
- **Find septum**: Detect division septa
- **Septum algorithm**: "Isodata" or "Box" thresholding
- **Find open septum**: Detect incomplete septa

**Fluorescence Analysis:**
- **Baseline margin**: Background region size (default: 30)

**Cell Cycle Classification:**
- **Classify cell cycle**: Enable cell cycle classification
- **Model**: Choose appropriate pre-trained model or choose a custom model
    - **Pre-trained models:**
        - S.aureus DNA+Membrane Epi
        - S.aureus DNA+Membrane SIM
        - S.aureus DNA Epi
        - S.aureus DNA SIM
        - S.aureus Membrane Epi
        - S.aureus Membrane SIM
- **Custom model path**: If you selected custom, provide the path to your own model file (.keras)
- **Custom model input**: Specify input type for custom model (Membrane, DNA, or Membrane+DNA)
- **Custom model max size**: Maximum cell size for custom model (default: 50 pixels)

**Analysis Options:**
- **Compute Colocalization**: Multi-channel colocalization analysis via PCC's.
- **Generate Report**: Create HTML report.
- **Report path**: Directory to save the report.
- **Compute Heatmap**: Spatial analysis visualization of a fluorescence channel.

---

## 📊 Morphological Measurements

napari-mAIcrobe computes shape and size parameters using scikit-image regionprops.

### 📏 Basic Shape Parameters

**Area and Size:**
- **Area**: Cell area in pixels and μm²
- **Perimeter**: Cell boundary length

**Shape Descriptors:**
- **Eccentricity**: Ellipse eccentricity (0=circle, 1=line)

---

## 💡 Intensity Analysis

Quantify fluorescence signals in subcellular compartments.

### 📈 Channel Measurements

**Basic Statistics:**
- **Baseline intensity**: Local background signal. This value is subtracted from all other intensity measurements.
- **Cell Median intensity**: Median fluorescence within the entire cell
- **Membrane Median intensity**: Median fluorescence in the membrane region
- **Cytoplasm Median intensity**: Median fluorescence in the cytoplasmic region
- **Septum Median intensity**: Median fluorescence in the septum region (if detected and enabled otherwise 0)
- **Fluorescence Ratios 100%, 75%, 25%, 10% percentiles**: Ratios between septum and membrane (if septum detected and enabled otherwise 0)
- **DNA Ratio**: Relative DNA content compared to baseline background fluorescence (if DNA channel provided, otherwise 0)

---

## 🧠 Cell Classification

Use deep learning models to automatically classify cells.

### 🔬 Pre-trained Models

napari-mAIcrobe includes **6 specialized models** for cell cycle determination in *S. aureus*:

**🧬 DNA + Membrane Models:**
- **S.aureus DNA+Membrane Epi**: Epifluorescence imaging
- **S.aureus DNA+Membrane SIM**: Super-resolution SIM

**🧬 DNA Only Models:**
- **S.aureus DNA Epi**: Nuclear staining, epifluorescence
- **S.aureus DNA SIM**: Nuclear staining, super-resolution SIM

**🔴 Membrane Only Models:**
- **S.aureus Membrane Epi**: Membrane staining, epifluorescence
- **S.aureus Membrane SIM**: Membrane staining, super-resolution SIM

> **Note:** To build your own training dataset from annotated cells, export pickles via `Plugins > mAIcrobe > Compute pickles`. See the [Cell Classification Guide](cell-classification.md) for details.

---

## 📈 Colocalization Analysis

Quantify spatial relationships between two fluorescence channels.

### 🔗 Colocalization Metrics

**Correlation Coefficients:**
- **Pearson correlation coefficient**: Linear relationship strength

---

## 🔍 Interactive Filtering

Use the **"Filter cells"** widget for real-time quality control:

1. 🏷️ **Select Labels layer** after compute_cells
2. ➕ **Add filters** for any measured feature
3. 👁️ **Preview filtered population** in real-time
4. ✅ **Use the filtered results for further analysis** The new layer "Filtered cells" contains only the selected cells.

---

## 📚 Further Reading

- **[Cell Classification](cell-classification.md)** - Detailed cell classification guide
- **[API Reference](../api/api-reference.md)** - Programmatic analysis
- **[Tutorials](../tutorials/basic-workflow.md)** - Step-by-step examples

### 🔗 Technical References
- **scikit-image regionprops**: [Documentation](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)
- **napari-skimage-regionprops plugin**: [GitHub](https://github.com/haesleinhuepf/napari-skimage-regionprops) - napari-mAIcrobe internally uses this plugin to add regionprops tables to the GUI.

---

**Next:** Explore deep learning cell classification in the **[Cell Classification Guide](cell-classification.md)** 🧠
