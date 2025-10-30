# Cell Classification Guide

This guide provides comprehensive information about mAIcrobe's cell classification system, including model selection, usage, and custom model integration.

## 🧠 Overview

mAIcrobe uses deep learning models to automatically classify cells based on their morphological and fluorescence features. The plugin includes:
 - **6 pre-trained models** optimized for the cell cycle stage detection of _Staphylococcus aureus_ under various imaging conditions
 - **1 pre-trained model** for *E. coli* antibiotic phenotyping
 - **Support for user-trained custom models**

---

## 🔬 Pre-trained Models

mAIcrobe includes 7 specialized models optimized for different imaging conditions and channel availability.

### 🧬🔴 DNA + Membrane Models

#### **S.aureus DNA+Membrane Epi**
- **Imaging**: Epifluorescence microscopy
- **Channels**: DNA stain (e.g., Hoechst) + Membrane stain (e.g., NileRed)
- **Use case**: Cell cycle phase detection in *S. aureus* in standard fluorescence imaging with both channels

| Actual \ Predicted (%)| Class 1 (Phase 1) | Class 2 (Phase 2) | Class 3 (Phase 3)|
|-----------------------|-------------------|-------------------|------------------|
| **Phase 1**           | 89                | 11                | 0                |
| **Phase 2**           | 8                 | 86                | 6                |
| **Phase 3**           | 0                 | 6                 | 94               |

#### **S.aureus DNA+Membrane SIM**
- **Imaging**: Structured illumination microscopy (SIM)
- **Channels**: DNA stain + Membrane stain
- **Use case**: Cell cycle phase detection in super-resolution imaging

| Actual \ Predicted (%)| Class 1 (Phase 1) | Class 2 (Phase 2) | Class 3 (Phase 3)|
|-----------------------|-------------------|-------------------|------------------|
| **Phase 1**           | 89                | 11                | 0                |
| **Phase 2**           | 12                | 81                | 8                |
| **Phase 3**           | 1                 | 8                 | 91               |


#### **E.coli DNA+Membrane AB phenotyping**
- **Imaging**: Epifluorescence microscopy
- **Channels**: DNA stain + Membrane stain
- **Use case**: Phenotyping antibiotic-treated _E.coli_ cells. Distinguishes between control cells and those treated with Mecillinam or Nalidixate.

| Actual \ Predicted (%) | Class 1 (control) | Class 2 (Mecillinam) | Class 3 (Nalidixate) |
|------------------------|-------------------|----------------------|----------------------|
| **Phase 1**            | 88                | 0                    | 12                   |
| **Phase 2**            | 14                | 82                   | 4                    |
| **Phase 3**            | 20                | 0                    | 80                   |

---

### 🧬 DNA Only Models

#### **S.aureus DNA Epi**
- **Imaging**: Epifluorescence microscopy
- **Channels**: DNA stain only
- **Use case**: Cell cycle detection in *S. aureus* when membrane staining is not available/desired
- **Accuracy**: Lower than dual-channel models or membrane-only models

| Actual \ Predicted (%)| Class 1 (Phase 1) | Class 2 (Phase 2) | Class 3 (Phase 3)|
|-----------------------|-------------------|-------------------|------------------|
| **Phase 1**           | 92                | 8                 | 0                |
| **Phase 2**           | 12                | 79                | 10               |
| **Phase 3**           | 0                 | 8                 | 92               |

#### **S.aureus DNA SIM**
- **Imaging**: Structured illumination microscopy
- **Channels**: DNA stain only
- **Use case**: Cell cycle detection in *S. aureus* when membrane staining is not available/desired
- **Accuracy**: Lower than dual-channel models or membrane-only models

| Actual \ Predicted (%)| Class 1 (Phase 1) | Class 2 (Phase 2) | Class 3 (Phase 3)|
|-----------------------|-------------------|-------------------|------------------|
| **Phase 1**           | 78                | 20                | 2                |
| **Phase 2**           | 20                | 61                | 19               |
| **Phase 3**           | 3                 | 16                | 81               |

---

### 🔴 Membrane Only Models

#### **S.aureus Membrane Epi**
- **Imaging**: Epifluorescence microscopy
- **Channels**: Membrane stain only
- **Use case**: Cell cycle detection in *S. aureus* when DNA staining is not available/desired
- **Accuracy**: Comparable to dual-channel models, often better than DNA-only models

| Actual \ Predicted (%)| Class 1 (Phase 1) | Class 2 (Phase 2) | Class 3 (Phase 3)|
|-----------------------|-------------------|-------------------|------------------|
| **Phase 1**           | 93                | 7                 | 0                |
| **Phase 2**           | 8                 | 87                | 5                |
| **Phase 3**           | 0                 | 7                 | 93               |

#### **S.aureus Membrane SIM**
- **Imaging**: Structured illumination microscopy
- **Channels**: Membrane stain only
- **Use case**: Cell cycle detection in *S. aureus* when DNA staining is not available/desired
- **Accuracy**: Comparable to dual-channel models, often better than DNA-only models

| Actual \ Predicted (%)| Class 1 (Phase 1) | Class 2 (Phase 2) | Class 3 (Phase 3)|
|-----------------------|-------------------|-------------------|------------------|
| **Phase 1**           | 88                | 12                | 0                |
| **Phase 2**           | 12                | 79                | 9                |
| **Phase 3**           | 0                 | 12                | 87               |

> **Note:** Values represent the percentage of samples classified into each category. Diagonal values indicate correct classifications, while off-diagonal values represent misclassifications.

---

## 🎨 Custom Model Integration

Load and use your own trained TensorFlow models.

### 📋 Model Requirements

**Supported Formats:**
- Keras models (.keras)

### 🔬 Model Training Guidelines

**Training Data Requirements:**
- Manually annotated cell images
- Balanced representation of all classes
- Consistent imaging conditions

To train your own models and assure seamless integration with the plugin, refer to the jupyter notebook: [Cell Cycle Model Training](../../notebooks/napari_mAIcrobe_cellcyclemodel.ipynb)

### 🥒 Build Your Own Training Data (Pickles)

#### For more detailed information refer to the tutorial: [Generate Training Data for Cell Classification](../tutorials/generate_trainingdata.md)

Use the Compute pickles widget to export standardized per-cell crops:

#### 🛠️ Workflow:
1. **Labels layer**: Ensure you have a Labels layer (cells). This is used to detect individual cells.
2. **Image layers**: Ensure you have one or two Image layers. These are used to extract the training crops from.
3. **Points layer**: Create a Points layer and name it as a positive integer class id (e.g., "1"); add one point per cell to assign that class. Make sure to repeat for other classes with different integer names.
4. **Export**: Open `Plugins > mAIcrobe > Compute pickles`, select the layers and output folder, then click "Save Pickle".

#### ⚠️ Important Notes:
- The Points layer name must be a positive integer (class id). This is required.
- Each point in the Points layer assigns the corresponding cell to that class. Make sure to add only one point per cell and one cell per class.

#### 💾 What gets saved:
- `Class_<id>_source.p`: list of masked, padded, and resized crops (100×100; 100×200 if two channels concatenated).
- `Class_<id>_target.p`: list with the class id repeated for each crop.

These files integrate with the training notebook: [Cell Cycle Model Training](../../notebooks/napari_mAIcrobe_cellcyclemodel.ipynb).

---

## 📖 Further Reading

- **[Cell Analysis Guide](cell-analysis.md)** - Complete analysis workflows
- **[Getting Started](getting-started.md)** - Basic usage tutorial
- **[API Reference](../api/api-reference.md)** - Programmatic control

---

**Next:** Explore programmatic usage in the [API Reference](../api/api-reference.md).
