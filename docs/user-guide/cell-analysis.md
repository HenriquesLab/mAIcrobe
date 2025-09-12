# Cell Analysis Guide

This comprehensive guide covers all aspects of automated cell analysis in napari-mAIcrobe, from basic morphometry to advanced AI-powered cell cycle classification.

## 🎯 Overview

The "Compute cells" widget provides:

- **Morphological analysis** - Shape and size measurements
- **Intensity analysis** - Fluorescence quantification
- **Cell cycle classification** - AI-powered phase determination
- **Colocalization analysis** - Multi-channel correlation
- **Quality control** - Interactive filtering and validation
- **Report generation** - Professional HTML output

## 🔬 Analysis Workflow

### Step 1: Prepare Your Data

Before running cell analysis, ensure you have:

1. **Segmented cells** - Labels layer from segmentation step
2. **Image channels** - Phase contrast, membrane, DNA (as needed)
3. **Pixel size** - Accurate calibration for meaningful measurements
4. **Parameter settings** - Configured for your experimental conditions

### Step 2: Configure Analysis Parameters

#### Essential Settings

**Image Selection:**
- **Label Image**: Segmentation results (required)
- **Membrane Image**: Fluorescence channel for boundaries
- **DNA Image**: Nuclear/nucleoid staining
- **Pixel size**: μm/pixel (critical for accurate measurements)

**Morphological Analysis:**
- **Inner mask thickness**: Pixels for cytoplasmic measurements (default: 4)
- **Baseline margin**: Background region size (default: 30)

#### Advanced Options

**Septum Detection:**
- **Find septum**: Detect division septa
- **Septum algorithm**: "Isodata" or "Box" thresholding
- **Find open septum**: Detect incomplete septa

**Cell Cycle Classification:**
- **Classify cell cycle**: Enable AI-powered phase detection
- **Model**: Choose appropriate pre-trained model
- **Custom model path**: Load your own TensorFlow model

**Analysis Options:**
- **Compute Colocalization**: Multi-channel correlation analysis
- **Generate Report**: Create HTML report with statistics
- **Compute Heatmap**: Spatial analysis visualization

## 📊 Morphological Measurements

napari-mAIcrobe computes comprehensive shape and size parameters using scikit-image regionprops.

### Basic Shape Parameters

**Area and Size:**
- **Area**: Cell area in pixels and μm²
- **Perimeter**: Cell boundary length
- **Equivalent diameter**: Diameter of circle with same area
- **Major axis length**: Length of major ellipse axis
- **Minor axis length**: Length of minor ellipse axis

**Shape Descriptors:**
- **Circularity**: 4π × Area / Perimeter²
- **Aspect ratio**: Major axis / Minor axis
- **Solidity**: Area / Convex area
- **Extent**: Area / Bounding box area
- **Eccentricity**: Ellipse eccentricity (0=circle, 1=line)

### Advanced Morphology

**Spatial Properties:**
- **Centroid**: Center of mass coordinates
- **Orientation**: Angle of major axis
- **Bounding box**: Minimal rectangle containing cell

**Texture Analysis:**
- **Compactness**: Relationship between area and perimeter
- **Convexity**: Ratio of perimeter to convex perimeter
- **Roughness**: Surface texture quantification

### Custom Measurements

**Inner Mask Analysis:**
Analyze cytoplasmic regions by creating inner masks:

```python
# Inner mask thickness determines analyzed region
inner_mask_thickness = 4  # pixels from cell boundary
```

**Applications:**
- Exclude membrane signal from cytoplasmic measurements
- Focus on nuclear regions for DNA analysis
- Separate membrane and cytoplasmic compartments

## 💡 Intensity Analysis

Quantify fluorescence signals across multiple channels with precision.

### Per-Channel Measurements

**Basic Statistics:**
- **Mean intensity**: Average signal within cell
- **Median intensity**: Robust central tendency
- **Standard deviation**: Signal variability
- **Min/Max intensity**: Signal extremes
- **Integrated density**: Total signal (mean × area)

**Advanced Intensity Metrics:**
- **Background-corrected intensity**: Subtract local background
- **Signal-to-noise ratio**: Signal quality assessment
- **Coefficient of variation**: Normalized signal variability

### Background Correction

Accurate background subtraction is crucial for quantitative analysis:

**Local Background:**
- Measured in region around each cell
- Size controlled by "baseline margin" parameter
- Automatically excluded from cell areas

**Implementation:**
```python
baseline_margin = 30  # pixels around cell for background
corrected_intensity = raw_intensity - local_background
```

## 🧠 AI-Powered Cell Cycle Classification

Use deep learning models to automatically classify cell cycle phases.

### Pre-trained Models

napari-mAIcrobe includes 6 specialized models:

**DNA + Membrane Models:**
- **S.aureus DNA+Membrane Epi**: Epifluorescence imaging
- **S.aureus DNA+Membrane SIM**: Super-resolution SIM

**DNA Only Models:**
- **S.aureus DNA Epi**: Nuclear staining, epifluorescence
- **S.aureus DNA SIM**: Nuclear staining, super-resolution

**Membrane Only Models:**
- **S.aureus Membrane Epi**: Boundary staining, epifluorescence  
- **S.aureus Membrane SIM**: Boundary staining, super-resolution

### Model Selection Guide

**Choose based on your imaging setup:**

| Imaging Method | Channels Available | Recommended Model |
|----------------|-------------------|-------------------|
| Epifluorescence | DNA + Membrane | S.aureus DNA+Membrane Epi |
| SIM | DNA + Membrane | S.aureus DNA+Membrane SIM |
| Epifluorescence | DNA only | S.aureus DNA Epi |
| SIM | DNA only | S.aureus DNA SIM |
| Epifluorescence | Membrane only | S.aureus Membrane Epi |
| SIM | Membrane only | S.aureus Membrane SIM |

### Classification Output

**Cell Cycle Phases:**
- **G1**: Pre-replication phase
- **S**: DNA synthesis phase  
- **G2**: Pre-division phase
- **Division**: Active cell division

**Confidence Scores:**
- Probability for each phase (0-1)
- Model confidence in classification
- Quality control metric

### Custom Model Integration

Load your own TensorFlow models:

```python
# Configure custom model
custom_model_path = "/path/to/your/model.keras"
custom_model_input = "Membrane"  # or "DNA" or "Membrane+DNA"
custom_model_maxsize = 50  # maximum cell size for analysis
```

**Model Requirements:**
- TensorFlow/Keras format (.keras, .h5)
- Input: normalized cell images
- Output: probability scores for each phase

## 📈 Colocalization Analysis

Quantify spatial relationships between fluorescence channels.

### Colocalization Metrics

**Correlation Coefficients:**
- **Pearson correlation**: Linear relationship strength
- **Spearman correlation**: Monotonic relationship
- **Cosine similarity**: Vector angle similarity

**Overlap Coefficients:**
- **Manders M1**: Fraction of channel 1 overlapping with channel 2
- **Manders M2**: Fraction of channel 2 overlapping with channel 1
- **Overlap coefficient**: Proportional overlap

### Applications

**Protein Colocalization:**
- Membrane proteins with membrane markers
- DNA-binding proteins with nucleoid regions
- Metabolic enzymes with cellular compartments

**Quality Control:**
- Verify channel alignment
- Assess staining specificity
- Detect imaging artifacts

## 🔧 Quality Control and Filtering

Ensure analysis accuracy through systematic quality control.

### Automated Quality Metrics

**Size Filters:**
- Minimum/maximum cell area
- Aspect ratio limits
- Circularity thresholds

**Intensity Filters:**
- Signal-to-noise ratio cutoffs
- Background intensity limits
- Saturation detection

**Shape Filters:**
- Edge cell exclusion
- Incomplete cell detection
- Artifact identification

### Interactive Filtering

Use the "Filter cells" widget for real-time quality control:

1. **Load analysis results** from Compute cells
2. **Set filter parameters** for any measured feature
3. **Preview filtered population** in real-time
4. **Export filtered results** for further analysis

### Manual Validation

**Best Practices:**
- Randomly sample 5-10% of cells for visual inspection
- Check for systematic errors across the dataset
- Validate classification results with known controls
- Document quality control procedures

## 📊 Statistical Analysis and Reporting

### HTML Report Generation

Enable comprehensive reporting:

```python
generate_report = True
report_path = "/path/to/output/directory"
```

**Report Contents:**
- Population statistics and distributions
- Cell cycle phase proportions
- Morphological parameter histograms
- Colocalization analysis results
- Quality control metrics
- Experimental metadata

### Data Export

**CSV Output:**
All measurements exported as comma-separated values:
- One row per cell
- All morphological and intensity measurements
- Cell cycle classifications and confidence scores
- Quality control flags

**Data Structure:**
```csv
cell_id,area_um2,perimeter_um,circularity,mean_membrane,mean_dna,cell_cycle_phase,confidence
1,2.34,8.45,0.82,145.2,89.3,G1,0.91
2,2.78,9.12,0.75,134.6,156.7,S,0.85
```

## 🎯 Experimental Applications

### Antimicrobial Testing

**Protocol:**
1. Treat bacteria with antimicrobial agents
2. Image at multiple time points
3. Analyze cell morphology and viability
4. Track population changes over time

**Key Measurements:**
- Cell count and viability
- Morphological changes (area, shape)
- Cell cycle arrest points
- Membrane integrity (intensity analysis)

### Cell Cycle Dynamics

**Experimental Design:**
1. Synchronize cell populations
2. Image at regular intervals
3. Track cell cycle progression
4. Quantify phase durations

**Analysis Focus:**
- Cell cycle phase proportions over time
- Individual cell tracking
- Division timing variability
- Synchrony maintenance

### Drug Mechanism Studies

**Applications:**
- Target identification through morphological profiling
- Dose-response curve generation
- Time-course analysis
- Mechanism of action classification

## 🚀 Advanced Analysis Techniques

### Batch Processing

Process multiple images with consistent parameters:

```python
# Process all images in directory
import glob
for image_path in glob.glob("*.tif"):
    # Load image
    # Run analysis with identical parameters  
    # Save results with systematic naming
```

### Time-lapse Analysis

Track cells over time:

1. **Cell tracking**: Link cells across time points
2. **Lineage analysis**: Follow cell divisions
3. **Dynamic measurements**: Growth rates, cycle times
4. **Population dynamics**: Birth/death rates

### High-Content Analysis

Large-scale screening applications:

- **Automated quality control**: Flag problematic images
- **Statistical analysis**: Population comparisons
- **Hit identification**: Outlier detection
- **Data visualization**: Multi-dimensional plots

## 📚 Troubleshooting

### Common Analysis Issues

**No cells detected in analysis:**
- Check that Labels layer is selected correctly
- Ensure segmentation completed successfully
- Verify pixel size is set appropriately

**Cell cycle classification fails:**
- Confirm both membrane and DNA channels are loaded
- Check model selection matches your imaging conditions
- Verify cells are within size limits (< 50 pixels diameter)

**Intensity measurements seem incorrect:**
- Verify background correction settings
- Check for image saturation or clipping
- Ensure proper channel assignment

**Memory errors with large datasets:**
- Process images individually rather than in batch
- Reduce image resolution if scientifically appropriate
- Close unnecessary applications

### Performance Optimization

**Speed Improvements:**
- Use smaller baseline margins for background calculation
- Disable unnecessary analysis options
- Process smaller regions of interest

**Memory Management:**
- Clear previous analysis results before new runs
- Process images in smaller batches
- Monitor system memory usage

## 📖 Further Reading

- **[AI Models Guide](ai-models.md)** - Detailed cell cycle classification
- **[API Reference](../api/api-reference.md)** - Programmatic analysis
- **[Tutorials](../tutorials/basic-workflow.md)** - Step-by-step examples
- **scikit-image regionprops**: [Documentation](https://scikit-image.org/docs/stable/api/skimage.measure.html#skimage.measure.regionprops)

---

**Next:** Explore AI-powered cell cycle classification in the [AI Models Guide](ai-models.md).