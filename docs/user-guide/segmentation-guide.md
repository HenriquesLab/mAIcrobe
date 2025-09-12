# Cell Segmentation Guide

This guide helps you choose the optimal segmentation method for your bacterial images and achieve the best possible cell detection results.

## 🎯 Overview

napari-mAIcrobe offers three main segmentation approaches:

1. **StarDist2D** - Star-convex shape detection (recommended for bacteria)
2. **Cellpose** - Deep learning-based universal segmentation
3. **Custom U-Net Models** - User-trained segmentation networks

## 🌟 StarDist2D (Recommended)

StarDist2D is optimized for detecting star-convex objects, making it ideal for bacterial cells.

### When to Use StarDist2D

- **Rod-shaped bacteria** like _S. aureus_, _E. coli_
- **Dense cell populations** with touching cells
- **Phase contrast images** with clear cell boundaries
- **Consistent cell morphology** across the image

### StarDist2D Parameters

**Key Settings:**
- **Probability Threshold** (0.3-0.7): Higher values = fewer false positives
- **NMS Threshold** (0.3-0.5): Controls overlap detection
- **Normalize Input**: Usually keep enabled for consistent results

### Optimizing StarDist2D

**For Dense Populations:**
```python
# Lower NMS threshold to separate touching cells
nms_threshold = 0.3
probability_threshold = 0.5
```

**For Sparse Populations:**
```python
# Higher thresholds for cleaner detection
nms_threshold = 0.4
probability_threshold = 0.6
```

**For Poor Contrast:**
```python
# Lower probability threshold to catch dim cells
probability_threshold = 0.3
normalize_input = True
```

## 🔬 Cellpose

Cellpose uses deep learning to segment cells of various shapes and sizes.

### When to Use Cellpose

- **Irregular cell shapes** not well-suited for StarDist
- **Mixed cell populations** with varying morphologies
- **Fluorescence images** with membrane staining
- **Human cells** or other non-bacterial samples

### Cellpose Parameters

**Model Selection:**
- **cyto**: General cytoplasm model (good for bacteria)
- **nuclei**: For nuclear segmentation
- **cyto2**: Improved cytoplasm model
- **Custom models**: Your own trained models

**Key Settings:**
- **Diameter** (pixels): Expected cell diameter
- **Flow threshold** (0.4): Controls segmentation sensitivity
- **Cellprob threshold** (-6 to 6): Probability threshold for cells

### Optimizing Cellpose

**For Small Bacteria:**
```python
model_type = "cyto"
diameter = 15  # Adjust based on your cell size
flow_threshold = 0.4
cellprob_threshold = 0.0
```

**For Larger Cells:**
```python
model_type = "cyto2"
diameter = 30
flow_threshold = 0.6
cellprob_threshold = -2.0
```

## 🎨 Custom U-Net Models

Load your own trained segmentation models for specialized applications.

### When to Use Custom Models

- **Specialized cell types** not covered by standard models
- **Unique imaging conditions** requiring custom training
- **Research applications** with specific segmentation requirements

### Loading Custom Models

```python
# In the Compute label widget:
# 1. Select "Custom" from model dropdown
# 2. Browse to your .h5 or .keras model file
# 3. Configure input preprocessing if needed
```

### Custom Model Requirements

**Supported Formats:**
- TensorFlow SavedModel (.pb)
- Keras models (.h5, .keras)
- ONNX models (.onnx)

**Input Requirements:**
- Single-channel grayscale images
- Normalized pixel values (0-1)
- Compatible input dimensions

## 📊 Comparing Segmentation Methods

| Method | Best For | Speed | Accuracy | Customization |
|--------|----------|-------|----------|--------------|
| **StarDist2D** | Rod-shaped bacteria | Fast | High | Medium |
| **Cellpose** | Variable shapes | Medium | High | High |
| **Custom Models** | Specialized cases | Varies | Variable | Maximum |

## 🔧 Troubleshooting Segmentation

### Common Issues and Solutions

#### Poor Cell Detection

**Symptoms:** Missing cells, incomplete boundaries
**Solutions:**
- Lower probability/flow thresholds
- Check image contrast and quality
- Try different models
- Adjust diameter (Cellpose)

#### Over-segmentation

**Symptoms:** Single cells split into multiple objects
**Solutions:**
- Increase probability threshold
- Increase NMS threshold (StarDist)
- Increase diameter (Cellpose)
- Apply smoothing filters

#### Under-segmentation

**Symptoms:** Multiple cells merged into single objects
**Solutions:**
- Decrease NMS threshold (StarDist)
- Decrease flow threshold (Cellpose)
- Use watershed post-processing
- Improve image contrast

#### Background Noise

**Symptoms:** False positive detections in background
**Solutions:**
- Increase probability thresholds
- Enable input normalization
- Pre-process images to reduce noise
- Use size filters in post-processing

## 📈 Preprocessing Tips

### Image Enhancement

**Contrast Enhancement:**
```python
# Using scikit-image
from skimage import exposure
enhanced = exposure.equalize_adapthist(image)
```

**Noise Reduction:**
```python
# Gaussian filtering
from skimage import filters
denoised = filters.gaussian(image, sigma=0.5)
```

**Background Subtraction:**
```python
# Rolling ball background subtraction
from skimage.morphology import disk
from skimage.filters import rank
background = rank.mean(image, disk(50))
corrected = image - background
```

## 🎯 Workflow Recommendations

### Phase Contrast Images

1. **Start with StarDist2D**
2. **Probability threshold**: 0.5
3. **NMS threshold**: 0.4
4. **Enable normalization**

### Fluorescence Membrane Images

1. **Try Cellpose first** (cyto model)
2. **Diameter**: Measure typical cell size
3. **Flow threshold**: 0.4
4. **Consider custom preprocessing**

### Mixed/Complex Images

1. **Test both StarDist2D and Cellpose**
2. **Compare results visually**
3. **Consider ensemble approaches**
4. **Manual validation on subset**

## 📏 Validation and Quality Control

### Manual Validation

Always validate segmentation on a representative subset:

1. **Random sampling**: Check 50-100 cells
2. **Visual inspection**: Look for common errors
3. **Quantitative metrics**: Count false positives/negatives
4. **Parameter adjustment**: Based on validation results

### Automated Quality Metrics

**Segmentation Quality Indicators:**
- Cell count consistency across similar images
- Size distribution reasonableness
- Shape parameter distributions
- Boundary completeness scores

### Best Practices

**Consistent Parameters:**
- Use identical settings for comparative studies
- Document all parameter choices
- Test on control/reference images first

**Batch Processing:**
- Process similar images with same parameters
- Monitor quality metrics across batches
- Flag outlier images for manual review

## 🚀 Advanced Techniques

### Multi-scale Segmentation

For images with cells of varying sizes:

1. **Run multiple scales** with different diameter settings
2. **Combine results** using confidence scores
3. **Post-process** to resolve conflicts

### Ensemble Methods

Combine multiple models for improved accuracy:

1. **Run StarDist2D and Cellpose**
2. **Compare outputs** and select best regions
3. **Merge results** using overlap analysis

### Custom Training

For specialized applications:

1. **Collect training data** (images + annotations)
2. **Use existing frameworks** (StarDist, Cellpose)
3. **Validate thoroughly** on independent test set
4. **Document model performance** and limitations

## 📚 Further Reading

- **[Cell Analysis Guide](cell-analysis.md)** - What to do after segmentation
- **[AI Models Guide](ai-models.md)** - Cell cycle classification
- **[API Reference](../api/api-reference.md)** - Programmatic control
- **StarDist Paper**: [Schmidt et al., MICCAI 2018](https://arxiv.org/abs/1806.03535)
- **Cellpose Paper**: [Stringer et al., Nature Methods 2021](https://doi.org/10.1038/s41592-020-01018-x)

---

**Next:** Learn how to analyze your segmented cells in the [Cell Analysis Guide](cell-analysis.md).