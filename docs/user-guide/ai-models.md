# AI Models Guide

This guide provides comprehensive information about napari-mAIcrobe's AI-powered cell cycle classification system, including model selection, usage, and custom model integration.

## 🧠 Overview

napari-mAIcrobe uses deep learning models to automatically classify bacterial cell cycle phases based on morphological and fluorescence features. This AI-powered approach provides:

- **Automated classification** into G1, S, G2, and Division phases
- **Quantitative confidence scores** for each prediction
- **Specialized models** for different imaging conditions
- **Custom model support** for research-specific applications

## 🎯 Cell Cycle Phases

### Phase Definitions

**G1 Phase (Gap 1):**
- Pre-replication phase
- Cell growth and normal metabolic activities
- Single, compact nucleoid
- Characteristic morphology: elongated but not dividing

**S Phase (Synthesis):**
- DNA replication phase  
- Nucleoid begins to expand and separate
- Increased cell length
- Intermediate morphological features

**G2 Phase (Gap 2):**
- Post-replication, pre-division phase
- Two distinct nucleoid regions
- Cell elongation continues
- Preparation for division

**Division Phase:**
- Active cell division
- Formation of septum/constriction
- Two cells in process of separation
- Characteristic dumbbell or peanut shape

### Visual Identification

Each phase has distinct morphological and fluorescence characteristics:

| Phase | Nucleoid Pattern | Cell Shape | Membrane Pattern |
|-------|------------------|------------|------------------|
| **G1** | Single, compact | Elongated rod | Uniform boundaries |
| **S** | Expanding, diffuse | Longer rod | Slight indentations |
| **G2** | Two distinct regions | Extended rod | Pre-septum formation |
| **Division** | Separated nucleoids | Constricted/split | Clear septum |

## 🔬 Pre-trained Models

napari-mAIcrobe includes 6 specialized models optimized for different imaging conditions and channel availability.

### DNA + Membrane Models

**S.aureus DNA+Membrane Epi**
- **Imaging**: Epifluorescence microscopy
- **Channels**: DNA stain (e.g., DAPI) + Membrane stain (e.g., FM 4-64)
- **Resolution**: Standard diffraction-limited imaging
- **Best for**: Most common fluorescence setups

**S.aureus DNA+Membrane SIM**
- **Imaging**: Structured illumination microscopy (SIM)
- **Channels**: DNA stain + Membrane stain
- **Resolution**: Super-resolution (~100-120 nm)
- **Best for**: High-resolution studies requiring fine detail

### DNA Only Models

**S.aureus DNA Epi**
- **Imaging**: Epifluorescence microscopy
- **Channels**: DNA stain only
- **Use case**: When membrane staining is not available/desired
- **Accuracy**: Good, but lower than dual-channel models

**S.aureus DNA SIM**
- **Imaging**: Structured illumination microscopy
- **Channels**: DNA stain only
- **Resolution**: Super-resolution
- **Best for**: High-resolution nucleoid analysis

### Membrane Only Models

**S.aureus Membrane Epi**
- **Imaging**: Epifluorescence microscopy
- **Channels**: Membrane stain only
- **Use case**: Studies focusing on cell boundary dynamics
- **Applications**: Membrane protein localization studies

**S.aureus Membrane SIM**
- **Imaging**: Structured illumination microscopy
- **Channels**: Membrane stain only
- **Resolution**: Super-resolution
- **Best for**: High-resolution membrane studies

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

### Performance Comparison

| Model Type | Accuracy | Speed | Data Required |
|------------|----------|-------|--------------|
| DNA+Membrane | Highest (>90%) | Medium | Both channels |
| DNA Only | High (>85%) | Fast | DNA channel |
| Membrane Only | Good (>80%) | Fast | Membrane channel |

## 🔧 Model Configuration

### Basic Setup

In the "Compute cells" widget:

1. **Enable cell cycle classification**: Check the box
2. **Select model**: Choose appropriate pre-trained model
3. **Configure parameters**: Set pixel size and analysis options
4. **Run analysis**: Execute classification

### Advanced Parameters

**Pixel Size Calibration:**
```python
pixel_size = 0.065  # μm/pixel - critical for accurate analysis
```

**Size Limits:**
```python
custom_model_maxsize = 50  # maximum cell diameter in pixels
```
- Cells larger than this limit are excluded from classification
- Prevents processing of cell clusters or artifacts
- Default: 50 pixels (suitable for typical bacterial cells)

## 🎨 Custom Model Integration

Load and use your own trained TensorFlow models.

### Model Requirements

**Technical Specifications:**
- **Format**: TensorFlow SavedModel, Keras (.keras), or HDF5 (.h5)
- **Input**: Normalized cell image patches
- **Output**: Probability scores for each cell cycle phase
- **Preprocessing**: Consistent with training data

**Input Configurations:**
- **"Membrane"**: Single-channel membrane fluorescence
- **"DNA"**: Single-channel nuclear/nucleoid staining  
- **"Membrane+DNA"**: Two-channel composite images

### Loading Custom Models

```python
# In the Compute cells widget:
model = "custom"
custom_model_path = "/path/to/your/model.keras"
custom_model_input = "Membrane+DNA"  # Match your training data
custom_model_maxsize = 50
```

### Model Training Guidelines

**Training Data Requirements:**
- Manually annotated cell images
- Balanced representation of all phases
- Consistent imaging conditions
- Sufficient sample size (>1000 cells per phase)

**Preprocessing Pipeline:**
```python
# Example preprocessing for consistency
def preprocess_cell(cell_image, target_size=64):
    # Resize to standard dimensions
    resized = resize(cell_image, (target_size, target_size))
    # Normalize to [0, 1] range
    normalized = (resized - resized.min()) / (resized.max() - resized.min())
    return normalized
```

**Model Architecture Suggestions:**
- Convolutional Neural Networks (CNNs)
- ResNet or EfficientNet backbones
- Transfer learning from pre-trained models
- Data augmentation for robustness

## 📈 Classification Output

### Prediction Results

For each cell, the model provides:

**Phase Prediction:**
- Most likely phase (G1, S, G2, Division)
- Based on highest probability score

**Confidence Scores:**
- Probability for each phase (0.0 - 1.0)
- Sum of all probabilities = 1.0
- Higher scores indicate more confident predictions

**Example Output:**
```csv
cell_id,predicted_phase,confidence_G1,confidence_S,confidence_G2,confidence_Division
1,G1,0.85,0.10,0.03,0.02
2,Division,0.05,0.15,0.25,0.55
3,S,0.20,0.65,0.12,0.03
```

### Confidence Interpretation

**High Confidence (>0.8):**
- Strong morphological features
- Clear phase characteristics
- Reliable classification

**Medium Confidence (0.5-0.8):**
- Ambiguous features
- Transition between phases
- May require manual verification

**Low Confidence (<0.5):**
- Unclear morphology
- Poor image quality
- Consider exclusion from analysis

## 🔍 Quality Control

### Automated Quality Assessment

**Size Filtering:**
- Exclude cells outside expected size range
- Remove artifacts and cell clusters
- Focus analysis on individual bacteria

**Confidence Thresholding:**
```python
# Filter by confidence score
min_confidence = 0.6
high_confidence_cells = results[results['max_confidence'] > min_confidence]
```

**Morphological Validation:**
- Check shape parameters consistency
- Identify potential segmentation errors
- Flag outliers for manual review

### Manual Validation

**Sample Review Process:**
1. Randomly select 5-10% of classified cells
2. Visually inspect original images
3. Compare AI predictions with expert annotation
4. Calculate accuracy metrics
5. Adjust confidence thresholds as needed

**Validation Metrics:**
- Overall accuracy per phase
- Confusion matrix analysis
- Precision and recall per phase
- False positive/negative rates

## 📊 Statistical Analysis

### Population Analysis

**Phase Distribution:**
```python
# Calculate phase proportions
phase_counts = results['predicted_phase'].value_counts()
phase_proportions = phase_counts / len(results)
```

**Temporal Analysis:**
- Track phase distributions over time
- Calculate cell cycle period
- Identify synchrony or arrest points

### Comparative Studies

**Treatment Effects:**
- Compare phase distributions between conditions
- Statistical testing (Chi-square, Fisher's exact)
- Effect size quantification

**Model Performance:**
- Cross-validation on independent datasets
- Comparison with manual annotation
- Robustness across imaging conditions

## 🚀 Advanced Applications

### Time-lapse Analysis

**Cell Cycle Tracking:**
1. Track individual cells across time points
2. Monitor phase transitions
3. Calculate cycle timing
4. Identify division synchrony

**Implementation:**
```python
# Pseudo-code for time-lapse analysis
for timepoint in timeseries:
    cells = segment_cells(timepoint)
    phases = classify_phases(cells)
    track_cells(cells, previous_timepoint)
    calculate_transitions(phases)
```

### High-Content Screening

**Drug Screening Applications:**
- Identify cell cycle inhibitors
- Quantify arrest points
- Dose-response analysis
- Mechanism classification

**Quality Control Pipeline:**
1. Automated image quality assessment
2. Cell count and morphology validation
3. Classification confidence filtering
4. Statistical outlier detection

### Multi-Condition Studies

**Experimental Design:**
- Multiple treatment conditions
- Time-course experiments
- Dose-response studies
- Environmental condition testing

**Analysis Workflow:**
1. Consistent imaging parameters
2. Identical analysis settings
3. Batch processing with quality control
4. Statistical comparison of results

## 🛠️ Troubleshooting

### Common Issues

**Low Classification Accuracy:**
- Verify correct model selection for imaging conditions
- Check pixel size calibration
- Review image quality (contrast, focus, artifacts)
- Consider confidence threshold adjustment

**Memory/Performance Issues:**
- Reduce batch size for large datasets
- Use CPU-only TensorFlow installation
- Process images individually rather than in batch
- Close unnecessary applications

**Model Loading Errors:**
- Verify model file format compatibility
- Check file path accessibility
- Ensure TensorFlow version compatibility
- Review custom model input specifications

### Optimization Tips

**Improve Accuracy:**
- Use highest quality images possible
- Ensure proper staining protocols
- Calibrate pixel size accurately
- Choose appropriate model for imaging setup

**Enhance Performance:**
- Use appropriate confidence thresholds
- Filter out poor quality cells early
- Optimize image preprocessing
- Consider model-specific optimizations

## 📚 Model Development Resources

### Training Your Own Models

**Recommended Frameworks:**
- TensorFlow/Keras for deep learning
- scikit-learn for classical ML
- PyTorch (with conversion to TensorFlow)

**Data Annotation Tools:**
- ImageJ with ROI manager
- QuPath for large image analysis
- Custom napari widgets
- Collaborative annotation platforms

**Training Resources:**
- Transfer learning tutorials
- Cell image analysis papers
- Open source datasets
- napari-mAIcrobe training notebooks

### Contributing Models

**Model Sharing:**
1. Train and validate your model
2. Document performance metrics
3. Provide training data description
4. Submit via GitHub pull request

**Quality Standards:**
- Minimum accuracy thresholds
- Cross-validation results
- Imaging condition specifications
- Reproducibility documentation

## 📖 Further Reading

- **[Cell Analysis Guide](cell-analysis.md)** - Complete analysis workflows
- **[Getting Started](getting-started.md)** - Basic usage tutorial
- **[API Reference](../api/api-reference.md)** - Programmatic control
- **TensorFlow Documentation**: [tensorflow.org](https://www.tensorflow.org/)
- **Cell Cycle Biology**: Understanding biological context

---

**Next:** Explore programmatic usage in the [API Reference](../api/api-reference.md).