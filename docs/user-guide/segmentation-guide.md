# Cell Segmentation Guide

This guide helps you choose the optimal segmentation method for your bacterial images and achieve the best possible cell detection results.

## 🎯 Overview

mAIcrobe offers four main segmentation approaches:

| Method | Type | Training Required | Speed | Accuracy |
|--------|------|-------------------|-------|----------|
| **🌟 StarDist** | Deep Learning | ✅ Custom model | Medium | High |
| **🔬 Cellpose** | Deep Learning | ❌ Pre-trained | Medium | High |
| **🧠 U-Net** | Deep Learning | ✅ Custom model | Medium | High |
| **⚡ Thresholding** | Classical | ❌ None | Fast | Medium |

---

## 🌟 StarDist Models

**Best for:** Star-convex shaped cells (most bacteria)

### Key Features
- 🎯 **Purpose**: Deep learning-based segmentation for star-convex shapes
- 📊 **Performance**: High accuracy for bacterial cells
- 🔧 **Requirement**: Custom trained model needed

### Getting Started
1. **Learn more**: Check the [StarDist paper](https://arxiv.org/abs/1806.03535) and [repository](https://github.com/stardist/stardist)
2. **Training**: Use our example notebook at [`notebooks/StarDistSegmentationTraining.ipynb`](../../notebooks/StarDistSegmentationTraining.ipynb)
3. **Examples**: See [StarDist training examples](https://github.com/stardist/stardist/tree/main/examples/2D)

> **Note**: mAIcrobe doesn't include pre-trained StarDist models - you must provide your own.

---

## 🔬 Cellpose Models

**Best for:** General cell segmentation across diverse cell types

### Key Features
- 🎯 **Purpose**: Universal deep learning segmentation model
- 🚀 **Ready to use**: Pre-trained cyto3 model included
- 🌐 **Versatile**: Trained on diverse cell types and imaging modalities

### Getting Started
1. **Learn more**: Check the [Cellpose paper](https://www.nature.com/articles/s41592-020-01018-x) and [repository](https://github.com/MouseLand/cellpose)
2. **First run**: Model weights download automatically on first use
3. **Usage**: Select "CellPose cyto3" in the segmentation widget

> **Tip**: Cellpose is great for getting started quickly without training custom models.

---

## 🧠 U-Net Models

**Best for:** Custom applications with specific imaging conditions

### Key Features
- 🎯 **Purpose**: Convolutional neural network for precise segmentation
- 🔧 **Format**: Requires Keras model files (`.keras`)
- 🎨 **Flexible**: Can be trained for specific cell types and conditions

### Model Requirements
Your U-Net model should output:
- **0**: Background
- **1**: Cell boundary
- **2**: Cell interior

mAIcrobe converts this to individual cell labels using watershed segmentation.

### Getting Started
1. **Learn more**: Read the [U-Net paper](https://arxiv.org/abs/1505.04597)
2. **Training**: Use [ZeroCostDL4Mic](https://github.com/HenriquesLab/ZeroCostDL4Mic)
3. **Technical details**: See [watershed documentation](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed)

---

## ⚡ Thresholding-Based Methods

**Best for:** Quick analysis without training requirements

### Key Features
- 🚀 **Speed**: Fastest segmentation method
- 🔧 **No training**: Classical image processing
- ⚖️ **Trade-off**: Lower accuracy for complex images

### Available Methods

#### 📊 Isodata Thresholding
- **Type**: Global automatic threshold
- **How it works**: Analyzes image histogram to find optimal threshold
- **Best for**: Images with clear intensity separation
- **Reference**: [scikit-image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.filters.html#skimage.filters.threshold_isodata)

#### 🎯 Local Average Thresholding
- **Type**: Adaptive local threshold
- **How it works**: Computes threshold based on local neighborhood
- **Best for**: Images with uneven illumination
- **Reference**: [scikit-image documentation](https://scikit-image.org/docs/0.25.x/api/skimage.filters.html#skimage.filters.threshold_local)

### Processing Pipeline
1. **Threshold** → Binary image
2. **Distance transform** → Separate touching cells
3. **Watershed** → Individual cell labels

---

## 📏 Validation and Quality Control

### ✅ Manual Validation Checklist

Always validate segmentation results:

- [ ] **Sample size**: Check 50-100 cells randomly
- [ ] **Visual inspection**: Look for common segmentation errors:
  - Under-segmentation (multiple cells as one)
  - Over-segmentation (one cell split into multiple)
  - Boundary accuracy
  - Missing cells

### 📊 Automated Quality Metrics

**Key indicators to monitor:**

| Metric | Good Range | Red Flags |
|--------|------------|-----------|
| Cell count consistency | ±10% between similar images | >20% variation |
| Size distribution | Normal/log-normal shape | Many outliers |
| Circularity | Species-appropriate | Too many non-circular shapes |

---

## 🏆 Choosing the Right Method

| Your Situation | Recommended Method |
|----------------|-------------------|
| 🚀 **Quick start, no training** | Cellpose cyto3 |
| ⚡ **Very fast, simple images** | Isodata thresholding |
| 🎯 **Best accuracy, have training data** | StarDist (custom) |
| 🔬 **Specific imaging conditions** | U-Net (custom) |
| 🌈 **Uneven illumination** | Local average thresholding |

---

## 📚 Further Reading

- **[Cell Analysis Guide](cell-analysis.md)** - What to do after segmentation
- **[Cell Classification Guide](cell-classification.md)** - Cell classification workflows
- **[API Reference](../api/api-reference.md)** - Programmatic control

### 📖 Scientific References
- **StarDist**: [Schmidt et al., MICCAI 2018](https://arxiv.org/abs/1806.03535)
- **Cellpose**: [Stringer et al., Nature Methods 2021](https://doi.org/10.1038/s41592-020-01018-x)
- **U-Net**: [Ronneberger et al., MICCAI 2015](https://arxiv.org/abs/1505.04597)

### 🛠️ Technical Documentation
- **Watershed segmentation**: [scikit-image docs](https://scikit-image.org/docs/stable/api/skimage.segmentation.html#skimage.segmentation.watershed)
- **Image filters**: [scikit-image docs](https://scikit-image.org/docs/stable/api/skimage.filters.html)

---

**Next:** Learn how to analyze your segmented cells in the **[Cell Analysis Guide](cell-analysis.md)** 🔬
