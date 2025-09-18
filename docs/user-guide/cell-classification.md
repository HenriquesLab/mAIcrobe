# Cell Classification Guide

This guide provides comprehensive information about napari-mAIcrobe's cell classification system, including model selection, usage, and custom model integration.

## 🧠 Overview

napari-mAIcrobe uses deep learning models to automatically classify cells phases based on morphological and fluorescence features. The plugin includes 6 pre-trained models optimized for the cell cycle stage detection of _Staphylococcus aureus_ under various imaging conditions, but it also has support for user-trained custom models.

## 🔬 Pre-trained Models

napari-mAIcrobe includes 6 specialized models optimized for different imaging conditions and channel availability.

### DNA + Membrane Models
---
**S.aureus DNA+Membrane Epi**
- **Imaging**: Epifluorescence microscopy
- **Channels**: DNA stain (e.g., Hoechst) + Membrane stain (e.g., NileRed)
- **Use case**: Standard fluorescence imaging with both channels

| Actual \ Predicted (%)| Class 1 | Class 2 | Class 3 |
|-----------------------|---------|---------|---------|
| **Phase 1**           | 89      | 11      | 0       |
| **Phase 2**           | 8       | 86      | 6       |
| **Phase 3**           | 0       | 6       | 94      |


**S.aureus DNA+Membrane SIM**
- **Imaging**: Structured illumination microscopy (SIM)
- **Channels**: DNA stain + Membrane stain
- **Use case**: Super-resolution imaging for detailed morphology

| Actual \ Predicted (%)| Class 1 | Class 2 | Class 3 |
|-----------------------|---------|---------|---------|
| **Phase 1**           | 89      | 11      | 0       |
| **Phase 2**           | 12      | 81      | 8       |
| **Phase 3**           | 1       | 8       | 91      |

### DNA Only Models
---
**S.aureus DNA Epi**
- **Imaging**: Epifluorescence microscopy
- **Channels**: DNA stain only
- **Use case**: When membrane staining is not available/desired
- **Accuracy**: Lower than dual-channel models or membrane-only models

| Actual \ Predicted (%)| Class 1 | Class 2 | Class 3 |
|-----------------------|---------|---------|---------|
| **Phase 1**           | 92      | 8       | 0       |
| **Phase 2**           | 12      | 79      | 10      |
| **Phase 3**           | 0       | 8       | 92      |

**S.aureus DNA SIM**
- **Imaging**: Structured illumination microscopy
- **Channels**: DNA stain only
- **Use case**: When membrane staining is not available/desired
- **Accuracy**: Lower than dual-channel models or membrane-only models

| Actual \ Predicted (%)| Class 1 | Class 2 | Class 3 |
|-----------------------|---------|---------|---------|
| **Phase 1**           | 78      | 20      | 2       |
| **Phase 2**           | 20      | 61      | 19      |
| **Phase 3**           | 3       | 16      | 81      |

### Membrane Only Models

**S.aureus Membrane Epi**
- **Imaging**: Epifluorescence microscopy
- **Channels**: Membrane stain only
- **Use case**: When DNA staining is not available/desired
- **Accuracy**: Comparable to dual-channel models, often better than DNA-only models

| Actual \ Predicted (%)| Class 1 | Class 2 | Class 3 |
|-----------------------|---------|---------|---------|
| **Phase 1**           | 93      | 7       | 0       |
| **Phase 2**           | 8       | 87      | 5       |
| **Phase 3**           | 0       | 7       | 93      |

**S.aureus Membrane SIM**
- **Imaging**: Structured illumination microscopy
- **Channels**: Membrane stain only
- **Use case**: When DNA staining is not available/desired
- **Accuracy**: Comparable to dual-channel models, often better than DNA-only models

| Actual \ Predicted (%)| Class 1 | Class 2 | Class 3 |
|-----------------------|---------|---------|---------|
| **Phase 1**           | 88      | 12      | 0       |
| **Phase 2**           | 12      | 79      | 9       |
| **Phase 3**           | 0       | 12      | 87      |

**Note:** Values represent the percentage of samples classified into each category. Diagonal values indicate correct classifications, while off-diagonal values represent misclassifications.

## 📊 Model Selection Guide

Choose the appropriate model based on your experimental setup:

### Decision Tree

```
Do you have both DNA and membrane staining?
├── Yes: DNA+Membrane models
│   ├── Super-resolution imaging? → S.aureus DNA+Membrane SIM
│   └── Standard resolution? → S.aureus DNA+Membrane Epi
└── No: Single-channel models
    ├── DNA staining only?
    │   ├── Super-resolution? → S.aureus DNA SIM
    │   └── Standard resolution? → S.aureus DNA Epi
    └── Membrane staining only?
        ├── Super-resolution? → S.aureus Membrane SIM
        └── Standard resolution? → S.aureus Membrane Epi
```

## 🎨 Custom Model Integration

Load and use your own trained TensorFlow models.

### Model Requirements

**Supported Formats:**
- Keras models (.keras)


### Model Training Guidelines

**Training Data Requirements:**
- Manually annotated cell images
- Balanced representation of all classes
- Consistent imaging conditions

To train your own models and assure seamless integration with the plugin, refer to the jupyter notebook: [Cell Cycle Model Training](../../notebooks/napari_mAIcrobe_cellcyclemodel.ipynb)

## 📖 Further Reading

- **[Cell Analysis Guide](cell-analysis.md)** - Complete analysis workflows
- **[Getting Started](getting-started.md)** - Basic usage tutorial
- **[API Reference](../api/api-reference.md)** - Programmatic control

---

**Next:** Explore programmatic usage in the [API Reference](../api/api-reference.md).
